---
name: DiskANN-dr-distance
description: Use when working with the distance module of DiskANN — distance/similarity computation, SIMD optimization, and PQ distance lookups
---

# DiskANN Distance Module

## 1. Module Purpose & Capabilities

The distance module provides all vector similarity/distance computation for DiskANN. It defines an abstract `Distance<T>` base class with virtual `compare()`, then provides concrete implementations for L2, cosine, inner product, and "fast L2" (inner-product-based L2) metrics across three data types (`float`, `int8_t`, `uint8_t`). It also provides SIMD-accelerated helper intrinsics (`simd_utils.h`), Windows-only cosine SIMD kernels (`cosine_similarity.h`), and a product-quantization (PQ) distance layer (`QuantizedDistance` / `PQL2Distance`) that computes approximate distances from precomputed PQ centroid lookup tables.

**Key capabilities exposed externally:**
- `get_distance_function<T>(Metric m)` — factory that returns the best `Distance<T>*` for a given metric, auto-selecting SIMD tier at runtime.
- `Distance<T>::compare(a, b, length)` — pairwise distance between two raw vectors.
- `AVXNormalizedCosineDistanceFloat` — cosine via pre-normalization + inner product (requires `preprocessing_required() == true`).
- `PQL2Distance<data_t>` — PQ-compressed approximate L2 distance (load pivot tables, preprocess query, then batch-lookup distances during graph walk).
- `QuantizedDistance<data_t>` — abstract interface for any quantized distance scheme.

---

## 2. Core Design Logic

### 2.1 Virtual Dispatch Over Templates

Distance classes use a **virtual-dispatch** strategy (base class `Distance<T>` with virtual `compare()`) rather than compile-time CRTP or function pointers. This is a deliberate choice because:

1. The distance object is created **once** at index construction/load time via `get_distance_function<T>()`, then reused for millions of comparisons. The vtable overhead per call is negligible relative to the actual computation.
2. It lets the factory select the implementation at **runtime** based on CPU feature detection (`Avx2SupportedCPU`, `AvxSupportedCPU` globals in [src/utils.cpp](src/utils.cpp)).
3. The `Index` class stores a `Distance<T>*` and does not need to be templated on the distance metric.

### 2.2 SIMD Strategy: Tiered Fallback

The SIMD strategy is **compile-time guard + runtime selection**:

- **Compile-time:** `#ifdef USE_AVX2` / `#ifdef __AVX__` / `#ifdef __SSE2__` guards wrap the actual intrinsics. On Linux, GCC-style `#pragma omp simd` is the baseline fallback.
- **Runtime:** `get_distance_function<T>()` checks global booleans `Avx2SupportedCPU` / `AvxSupportedCPU` (set once at static init via CPUID) to pick the fastest available class:
  - **AVX2 path:** `DistanceL2Float` (uses `_mm256_load_ps`, `_mm256_fmadd_ps`, `_mm256_reduce_add_ps`), `DistanceL2Int8` (uses `_mm256_subs_epi8` + custom `_mm256_mul_epi8`). The `_mm256_mul_epi8` helper sign-extends 8-bit lanes to 16-bit via `_mm256_unpacklo_epi8` / `_mm256_unpackhi_epi8`, multiplies-and-adds pairs via `_mm256_madd_epi16` on both the low and high halves separately, then **adds the low and high half results** together via `_mm256_add_epi32`, and finally converts to float via `_mm256_cvtepi32_ps`. The single-operand overload squares the input (`X*X`), while the two-operand overload computes the dot product of `X` and `Y`.
  - **AVX-only path:** `AVXDistanceL2Float` / `AVXDistanceL2Int8` (128-bit SSE loops).
  - **Scalar fallback:** `SlowDistanceL2<T>` (plain C loop).

For **inner product** and **cosine**, only AVX2-level implementations exist; there is no tiered fallback (the code always uses the AVX2 path or a scalar loop under `#ifdef` guards).

### 2.3 Cosine = Normalize + Inner Product

`AVXNormalizedCosineDistanceFloat` implements cosine distance as:

1. `preprocessing_required()` returns `true`.
2. `preprocess_base_points()` normalizes every base vector in-place.
3. `preprocess_query()` copies and normalizes the query vector.
4. `compare()` = `1.0f + innerProduct.compare(a, b, length)` — the inner product returns `-dot(a,b)`, so `1 + (-dot)` = `1 - dot` = cosine distance for unit vectors.

This avoids computing norms at query time. The trade-off: base data is **modified in place** during index build.

### 2.4 Platform Split: Windows vs Linux

`cosine_similarity.h` is entirely `#ifdef _WINDOWS`. On Windows, cosine for int8 and float uses dedicated SIMD kernels (`NormScalarProductSIMD`, `NormScalarProductSIMD2`) that compute dot product and norms simultaneously. On Linux, cosine for int8/float uses scalar loops in `distance.cpp`.

The `AVXDistanceL2Int8` and `AVXDistanceL2Float` classes have **real implementations only on Windows** (inside `#ifdef _WINDOWS`). On Linux they return 0 — they are only used as a fallback tier, and the Linux build assumes AVX2 is available (the flag `Avx2SupportedCPU` is hardcoded `true` in [src/utils.cpp](src/utils.cpp) line 69 for non-Windows).

### 2.5 PQ Distance: Two-Phase Lookup

`PQL2Distance` implements asymmetric distance estimation (ADC) for Product Quantization:

1. **Offline:** `load_pivot_data()` reads 256 centroids × ndims table from file, plus chunk offsets and centroid. Transposes table into `_tables_tr` (column-major, 256 × ndims) for cache-friendly access during lookup. The transpose formula is: `_tables_tr[j * 256 + i] = _tables[i * ndims + j]` — this reorders from row-major (centroid × dim) to column-major (dim × centroid) so that the per-dimension centroid scan in `prepopulate_chunkwise_distances()` reads contiguous memory.
2. **Per-query:** `preprocess_query()` subtracts global centroid, optionally applies OPQ rotation matrix, then calls `prepopulate_chunkwise_distances()` to build a `[num_chunks × 256]` distance table (squared L2 from rotated query to each centroid per chunk).
3. **Per-candidate:** `preprocessed_distance()` calls `pq_dist_lookup()` (in [src/pq.cpp](src/pq.cpp)) which sums up the pre-computed chunk distances for each candidate's PQ code — a simple table lookup with `_mm_prefetch` hints.

This design separates the expensive per-query precomputation from the cheap per-candidate lookup, which is critical for performance during graph traversal where thousands of candidates are evaluated per query.

---

## 3. State Flow

### 3.1 Full-Precision Distance Computation

```
User calls get_distance_function<T>(metric)
  → Factory checks Avx2SupportedCPU / AvxSupportedCPU globals
  → Returns concrete Distance<T>* (e.g. DistanceL2Float)

User calls distance->compare(a, b, dim)
  → Virtual dispatch to concrete implementation
  → SIMD loop (AVX2 → 8-wide float ops, or SSE → 4-wide, or scalar)
  → Returns float distance value
```

For **cosine with float** (AVXNormalizedCosineDistanceFloat):
```
Index build time:
  preprocessing_required() → true
  preprocess_base_points(data, dim, N) → normalizes all N vectors in-place

Query time:
  preprocess_query(query, dim, scratch) → normalizes query into scratch
  compare(scratch, base_vec, dim) → 1.0 + inner_product(scratch, base_vec)
```

### 3.2 PQ Distance Computation

```
Index load:
  PQL2Distance(num_chunks, use_opq)
  load_pivot_data(pq_table_file, num_chunks)
    → Loads: _tables (256 × ndims), _centroid (ndims), _chunk_offsets (num_chunks+1)
    → If OPQ: loads _rotmat_tr (ndims × ndims)
    → Computes _tables_tr (transposed, column-major)

Per-query:
  preprocess_query(aligned_query, dim, pq_scratch)
    → Converts query to float, copies to scratch
    → Subtracts _centroid
    → If OPQ: multiplies by _rotmat_tr
    → Calls prepopulate_chunkwise_distances(rotated_query, dist_scratch)
      → For each chunk, for each of 256 centroids:
           dist[chunk][centroid] = Σ (query[j] - center[j])² for dims in chunk

Per-candidate (during graph walk):
  preprocessed_distance(pq_scratch, id_count, dists_out)
    → Calls pq_dist_lookup() [in src/pq.cpp]
      → For each candidate: dist = Σ_chunk dist_table[chunk][pq_code[chunk]]
```

### 3.3 SIMD Dispatch Logic

The `_mm256_reduce_add_ps()` helper in `simd_utils.h` is the core horizontal-add used by `DistanceL2Float`. The flow inside `DistanceL2Float::compare()`:

```
for j in 0..size/8:
  a_vec = _mm256_load_ps(a + 8*j)     // aligned load
  b_vec = _mm256_load_ps(b + 8*j)     // aligned load  
  diff  = _mm256_sub_ps(a_vec, b_vec)
  sum   = _mm256_fmadd_ps(diff, diff, sum)  // FMA: sum += diff²
result = _mm256_reduce_add_ps(sum)     // horizontal reduction
```

For `DistanceInnerProduct<float>::inner_product()`:
```
AVX_DOT macro: load 8 floats from each, multiply, accumulate
Loop unrolled 2× (16 floats per iteration)
Final: horizontal sum of 256-bit accumulator
```

---

## 4. Common Modification Scenarios

### Scenario 1: Adding a New Distance Metric (e.g., Hamming)

1. **Define enum value**: Add `HAMMING = 4` to the `Metric` enum in [include/distance.h](include/distance.h) (line 10-15).
2. **Create concrete class**: Add `class DistanceHammingUInt8 : public Distance<uint8_t>` in the same header.
3. **Implement `compare()`**: Add the implementation in [src/distance.cpp](src/distance.cpp).
4. **Register in factory**: Add an `else if (m == diskann::Metric::HAMMING)` branch in the appropriate `get_distance_function<uint8_t>()` specialization (around line 692 of distance.cpp).
5. **No changes needed** to `QuantizedDistance` unless PQ-based approximate Hamming is desired.

### Scenario 2: Adding SIMD Support for a New Architecture (e.g., ARM NEON)

1. **simd_utils.h** ([include/simd_utils.h](include/simd_utils.h)): Add `#ifdef __ARM_NEON` blocks with NEON equivalents of `_mm256_mul_epi8`, `_mm256_reduce_add_ps`, etc.
2. **CPU detection**: In [src/utils.cpp](src/utils.cpp), add a `cpuHasNeonSupport()` function and a global `NeonSupportedCPU` boolean.
3. **New distance classes** (optional): Create `NeonDistanceL2Float` etc. in `distance.h` / `distance.cpp`, or gate existing classes with `#ifdef __ARM_NEON`.
4. **Factory update**: Add NEON tier in `get_distance_function<float>()` before the AVX2 check.
5. **cosine_similarity.h**: This file is Windows-only; for ARM you would add NEON cosine kernels in `distance.cpp` under a new `#ifdef`.

### Scenario 3: Adding a New Quantized Distance Type (e.g., PQ Inner Product)

1. **Create new class** inheriting `QuantizedDistance<data_t>` (use [include/quantized_distance.h](include/quantized_distance.h) as the interface).
2. **Model after `PQL2Distance`** in [include/pq_l2_distance.h](include/pq_l2_distance.h) / [src/pq_l2_distance.cpp](src/pq_l2_distance.cpp):
   - Override `preprocess_query()` to compute inner-product-based distance tables instead of L2.
   - Override `prepopulate_chunkwise_distances()` to use `Σ query[j] * center[j]` per chunk instead of `Σ (query[j] - center[j])²`.
   - `brute_force_distance()` needs matching logic.
3. **Register**: The `PQFlashIndex` and `Index` classes instantiate `PQL2Distance` directly; change to use the factory or conditional construction.

### Scenario 4: Improving PQ Distance Lookup Performance

The hot path is `pq_dist_lookup()` in [src/pq.cpp](src/pq.cpp) (lines 289-350). It uses scalar loops with `_mm_prefetch` hints. To optimize:
- SIMD gather: use `_mm256_i32gather_ps` to load 8 chunk distances at once for 8 candidates.
- The inner loop iterates `n_pts × n_chunks`; reorganizing to batch 8 or 16 candidates at a time with AVX2 gathers could yield significant speedup.
- The `prepopulate_chunkwise_distances()` function in [src/pq_l2_distance.cpp](src/pq_l2_distance.cpp) (line 258) is also scalar and could benefit from SIMD.

### Scenario 5: Supporting a New Data Type (e.g., float16)

1. **Add `Distance<float16>` specialization** or use a conversion wrapper.
2. **Create concrete classes**: `DistanceL2Float16`, etc. in `distance.h`.
3. **Template instantiation**: Add `template DISKANN_DLLEXPORT class ...` lines at the bottom of `distance.cpp` and ensure `get_distance_function<float16>()` is specialized.
4. **Alignment**: Adjust `_alignment_factor` in `Distance<T>` if the new type needs different alignment.
5. **PQ layer**: Instantiate `PQL2Distance<float16>` — the template already exists for `int8_t`, `uint8_t`, `float`.

---

## 5. Key Architectural Invariants

- **Alignment requirement**: `Distance<T>::get_required_alignment()` returns `_alignment_factor = 8` (meaning 8 elements = 32 bytes for float, matching AVX2). All input vectors must be aligned to this boundary. `DistanceL2Float::compare()` uses `_mm256_load_ps` (aligned load), which will segfault on unaligned data.
- **Distance semantics**: All `compare()` functions return a **distance** (lower = more similar). For inner product, `compare()` returns `-dot(a,b)` (negated). For cosine, it returns `1 - cosine_similarity`.
- **PQ invariant**: `preprocess_query()` must be called exactly once per query before any `preprocessed_distance()` calls. The scratch object (`PQScratch`) is caller-managed and must have properly sized buffers.
- **Template instantiations**: At the bottom of `distance.cpp` and `pq_l2_distance.cpp`, explicit template instantiations are listed. Forgetting to add one for a new type will cause linker errors.
