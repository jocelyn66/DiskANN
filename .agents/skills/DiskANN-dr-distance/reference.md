# DiskANN Distance Module — Reference

## File Map

| File | Role |
|------|------|
| `include/distance.h` | Core `Distance<T>` base class + all concrete distance class declarations |
| `src/distance.cpp` | All `compare()` implementations, SIMD kernels, `get_distance_function<T>()` factory |
| `include/cosine_similarity.h` | Windows-only SIMD cosine kernels (`NormScalarProductSIMD`, `NormScalarProductSIMD2`) |
| `include/simd_utils.h` | Low-level SIMD helper intrinsics for int8 multiply and float horizontal add |
| `include/quantized_distance.h` | Abstract `QuantizedDistance<data_t>` interface |
| `include/pq_l2_distance.h` | `PQL2Distance<data_t>` class declaration (PQ-based L2) |
| `src/pq_l2_distance.cpp` | `PQL2Distance` implementation: load pivot data, preprocess, distance lookup |
| `include/pq_common.h` | PQ constants (`NUM_PQ_CENTROIDS = 256`) and filename helpers |
| `include/pq_scratch.h` | `PQScratch<T>` scratch buffer structure |

---

## Enums

### `diskann::Metric` — `include/distance.h:10`

```cpp
enum Metric {
    L2 = 0,
    INNER_PRODUCT = 1,
    COSINE = 2,
    FAST_L2 = 3
};
```

---

## Classes and Interfaces

### `Distance<T>` — `include/distance.h:17`

Abstract base class for all full-precision distance computation. Template parameter `T` is the element type (`float`, `int8_t`, `uint8_t`).

| Method | Signature | Default Behavior |
|--------|-----------|------------------|
| `compare` | `virtual float compare(const T *a, const T *b, uint32_t length) const = 0` | Pure virtual |
| `compare` (with norms) | `virtual float compare(const T *a, const T *b, float normA, float normB, uint32_t length) const` | Throws `logic_error` |
| `post_normalization_dimension` | `virtual uint32_t post_normalization_dimension(uint32_t orig_dimension) const` | Returns `orig_dimension` |
| `get_metric` | `virtual Metric get_metric() const` | Returns `_distance_metric` |
| `preprocessing_required` | `virtual bool preprocessing_required() const` | Returns `false` |
| `preprocess_base_points` | `virtual void preprocess_base_points(T *original_data, size_t orig_dim, size_t num_points)` | No-op |
| `preprocess_query` | `virtual void preprocess_query(const T *query_vec, size_t query_dim, T *scratch_query)` | `memcpy` |
| `get_required_alignment` | `virtual size_t get_required_alignment() const` | Returns `_alignment_factor` (8) |

**Fields:** `Metric _distance_metric`, `size_t _alignment_factor = 8`.

---

### Concrete Distance Classes

All declared in `include/distance.h`, implemented in `src/distance.cpp`.

#### L2 Distance

| Class | Data Type | Metric | SIMD Level | Notes |
|-------|-----------|--------|------------|-------|
| `DistanceL2Float` | `float` | L2 | AVX2 (`_mm256_load_ps`, `_mm256_fmadd_ps`) | Primary L2 for float. Uses aligned loads. Marked `__attribute__((hot))` on Linux. |
| `AVXDistanceL2Float` | `float` | L2 | SSE (`_mm_loadu_ps`) | Fallback for AVX-only CPUs. **Windows-only real impl; returns 0 on Linux.** |
| `SlowDistanceL2<T>` | any `T` | L2 | Scalar | Template fallback. Plain `(a[i]-b[i])²` loop. |
| `DistanceL2Int8` | `int8_t` | L2 | AVX2 on Windows (`_mm256_subs_epi8`), `omp simd` on Linux | Uses `#pragma omp simd reduction` on Linux. |
| `AVXDistanceL2Int8` | `int8_t` | L2 | SSE on Windows | Windows-only real impl; returns 0 on Linux. |
| `DistanceL2UInt8` | `uint8_t` | L2 | `omp simd` | No dedicated SIMD path. |

#### Cosine Distance

| Class | Data Type | Metric | SIMD Level | Notes |
|-------|-----------|--------|------------|-------|
| `DistanceCosineFloat` | `float` | COSINE | Windows: `NormScalarProductSIMD` (AVX2/SSE). Linux: scalar | Computes `1 - dot/(||a|| * ||b||)` |
| `DistanceCosineInt8` | `int8_t` | COSINE | Windows: `NormScalarProductSIMD2` (AVX2/SSE). Linux: scalar | Same formula, integer arithmetic |
| `SlowDistanceCosineUInt8` | `uint8_t` | COSINE | Scalar | Used for uint8 on all platforms |
| `AVXNormalizedCosineDistanceFloat` | `float` | COSINE | AVX2 (via inner product) | **Pre-normalizes data.** `compare() = 1.0f + innerProduct.compare()` |

#### Inner Product

| Class | Data Type | Metric | SIMD Level | Notes |
|-------|-----------|--------|------------|-------|
| `DistanceInnerProduct<T>` | any `T` | IP | AVX2/SSE2/scalar (cascading `#ifdef`) | Returns `-dot(a,b)`. Only works for float despite template. |
| `AVXDistanceInnerProductFloat` | `float` | IP | AVX2 (`AVX_DOT` macro) | Returns `-dot(a,b)`. Cross-platform. |

#### Fast L2

| Class | Data Type | Metric | SIMD Level | Notes |
|-------|-----------|--------|------------|-------|
| `DistanceFastL2<T>` | any `T` | FAST_L2 | Inherits from `DistanceInnerProduct<T>` | `compare(a, b, norm, size)` = `-2*dot(a,b) + norm`. Caller pre-computes `||a||² + ||b||²`. |

---

### `QuantizedDistance<data_t>` — `include/quantized_distance.h`

Abstract interface for PQ-based approximate distance. Non-copyable.

| Method | Signature | Description |
|--------|-----------|-------------|
| `is_opq` | `virtual bool is_opq() const = 0` | Whether OPQ rotation is used |
| `get_quantized_vectors_filename` | `virtual string get_quantized_vectors_filename(const string &prefix) const = 0` | Path to compressed vectors file |
| `get_pivot_data_filename` | `virtual string get_pivot_data_filename(const string &prefix) const = 0` | Path to PQ pivots file |
| `get_rotation_matrix_suffix` | `virtual string get_rotation_matrix_suffix(const string &pq_pivots_filename) const = 0` | Rotation matrix filename suffix |
| `load_pivot_data` | `virtual void load_pivot_data(const string &pq_table_file, size_t num_chunks) = 0` | Load PQ centroid tables from file |
| `get_num_chunks` | `virtual uint32_t get_num_chunks() const = 0` | Number of PQ chunks |
| `preprocess_query` | `virtual void preprocess_query(const data_t *query_vec, uint32_t query_dim, PQScratch<data_t> &pq_scratch) = 0` | Build per-query distance tables |
| `preprocessed_distance` | `virtual void preprocessed_distance(PQScratch<data_t> &pq_scratch, uint32_t id_count, float *dists_out) = 0` | Batch distance from precomputed tables |
| `brute_force_distance` | `virtual float brute_force_distance(const float *query_vec, uint8_t *base_vec) = 0` | Exact PQ distance (for disk index) |

---

### `PQL2Distance<data_t>` — `include/pq_l2_distance.h`

Concrete implementation of `QuantizedDistance<data_t>` for L2 metric with PQ compression.

**Constructor:** `PQL2Distance(uint32_t num_chunks, bool use_opq = false)`

**Fields (protected):**

| Field | Type | Description |
|-------|------|-------------|
| `_tables` | `float*` | PQ centroid table, `[256 × ndims]`, row-major |
| `_tables_tr` | `float*` | Transposed table, `[ndims × 256]`, column-major — used for cache-friendly distance computation |
| `_ndims` | `uint64_t` | True dimensionality of vectors |
| `_num_chunks` | `uint64_t` | Number of PQ chunks (subspaces) |
| `_is_opq` | `bool` | Whether OPQ rotation is enabled |
| `_chunk_offsets` | `uint32_t*` | Array of `num_chunks+1` values delimiting which dims belong to each chunk |
| `_centroid` | `float*` | Global centroid (`ndims` values), subtracted from queries |
| `_rotmat_tr` | `float*` | OPQ rotation matrix (transposed), `ndims × ndims`. `nullptr` if not OPQ. |

**Key methods:**

- `load_pivot_data(file, num_chunks)` — Reads binary file with offsets → PQ table → centroid → chunk_offsets (→ rotation matrix if OPQ). Validates sizes. Computes `_tables_tr`.
- `preprocess_query(aligned_query, dim, scratch)` — Converts to float, subtracts centroid, applies OPQ rotation, fills `scratch.aligned_pqtable_dist_scratch` via `prepopulate_chunkwise_distances()`.
- `prepopulate_chunkwise_distances(query_vec, dist_vec)` — For each chunk `c` and each centroid `idx` (0..255): `dist_vec[256*c + idx] += (query[j] - center[j])²` for dims `j` in chunk `c`.
- `preprocessed_distance(scratch, n_ids, dists_out)` — Delegates to `pq_dist_lookup()` in `src/pq.cpp`.
- `brute_force_distance(query_vec, base_vec)` — Direct per-chunk L2 from query to PQ centroid of the base vector's code. Used in disk index exact reranking.

**Template instantiations:** `int8_t`, `uint8_t`, `float`.

---

### `PQScratch<T>` — `include/pq_scratch.h`

Scratch buffer for PQ distance computation, allocated per-thread.

| Field | Type | Min Size | Purpose |
|-------|------|----------|---------|
| `aligned_pqtable_dist_scratch` | `float*` | `256 × num_chunks` | Per-query distance table (chunk × centroid) |
| `aligned_dist_scratch` | `float*` | `MAX_DEGREE` | Output distances for candidates |
| `aligned_pq_coord_scratch` | `uint8_t*` | `num_chunks × MAX_DEGREE` | PQ codes of candidates, gathered before lookup |
| `rotated_query` | `float*` | `ndims` | Query after centroid subtraction and OPQ rotation |
| `aligned_query_float` | `float*` | `ndims` | Query converted to float |

---

## PQ Constants — `include/pq_common.h`

```cpp
#define NUM_PQ_BITS 8
#define NUM_PQ_CENTROIDS (1 << NUM_PQ_BITS)   // = 256
#define MAX_OPQ_ITERS 20
#define NUM_KMEANS_REPS_PQ 12
#define MAX_PQ_TRAINING_SET_SIZE 256000
#define MAX_PQ_CHUNKS 512
```

---

## SIMD Intrinsic Helpers — `include/simd_utils.h`

All functions are `static inline` in namespace `diskann`.

| Function | Signature | Description |
|----------|-----------|-------------|
| `_mm256_mul_epi8(__m256i X)` | `→ __m256` | Squared magnitude of 32 int8 elements: sign-extend to int16, `madd`, convert to float |
| `_mm256_mul_epi8(__m256i X, __m256i Y)` | `→ __m256` | Dot product of 32 int8 pairs: sign-extend both, `madd` lo+hi, convert to float |
| `_mm256_mul32_pi8(__m128i X, __m128i Y)` | `→ __m256` | Dot product of first 4 int8 pairs (zero-fills upper) via `_mm256_cvtepi8_epi16` |
| `_mm_mul_epi8(__m128i X, __m128i Y)` | `→ __m128` | 128-bit version: dot product of 16 int8 pairs |
| `_mm_mul_epi8(__m128i X)` | `→ __m128` | 128-bit version: squared magnitude of 16 int8 elements |
| `_mm_mulhi_epi8(__m128i X)` | `→ __m128` | Squared magnitude of upper 8 int8 elements only |
| `_mm_mulhi_epi8_shift32(__m128i X)` | `→ __m128` | Shift right by 32 bits first, then squared magnitude of upper 8 |
| `_mm_mul32_pi8(__m128i X, __m128i Y)` | `→ __m128` | Dot product of first 4 int8 pairs (128-bit) |
| `_mm256_reduce_add_ps(__m256 x)` | `→ float` | Horizontal sum of 8 floats: extract high 128, add, `movehl`, add, `shuffle`, add |

### SIMD technique for int8 dot product

The int8 dot product pattern (used throughout) is:
1. Sign-extend int8 → int16 via `_mm256_unpacklo_epi8` / `_mm256_unpackhi_epi8` with sign mask from `_mm256_cmpgt_epi8(zero, X)`
2. Multiply-add pairs of int16 → int32 via `_mm256_madd_epi16`
3. Sum lo and hi halves via `_mm256_add_epi32`
4. Convert int32 → float via `_mm256_cvtepi32_ps`

This processes 32 int8 elements per 256-bit iteration.

---

## Cosine SIMD Kernels (Windows only) — `include/cosine_similarity.h`

Wrapped in `#ifdef _WINDOWS`.

| Function | Signature | Description |
|----------|-----------|-------------|
| `NormScalarProductSIMD2(const int8_t*, const int8_t*, uint32_t)` | `→ float` | Cosine similarity for int8. AVX2 path processes 32 bytes/iter; SSE path 16 bytes/iter. Computes dot product and both norms simultaneously. |
| `NormScalarProductSIMD(const float*, const float*, uint32_t)` | `→ float` | Cosine similarity for float. AVX2 path processes 8 floats/iter. |
| `NormScalarProductSIMD2(const float*, const float*, uint32_t)` | `→ float` | Redirects to `NormScalarProductSIMD` |
| `CosineSimilarity2<T>(p1, p2, qty)` | `→ float` | Returns `max(0, 1 - NormScalarProductSIMD2(p1, p2, qty))` — i.e. cosine distance clamped to [0, ∞) |
| `CosineSimilarityNormalize<T>(pVector, qty)` | `→ void` | In-place L2 normalization. **Throws for int8, int16, int** (not meaningful for integer types). |

---

## Global CPU Feature Flags — `src/utils.cpp`

```cpp
// Windows: runtime CPUID detection
bool AvxSupportedCPU = cpuHasAvxSupport();
bool Avx2SupportedCPU = cpuHasAvx2Support();

// Linux/macOS: hardcoded (assumes AVX2 available)
bool Avx2SupportedCPU = true;
bool AvxSupportedCPU = false;   // not used as fallback on Linux
```

These are declared `extern bool` and checked by `get_distance_function()` and `cosine_similarity.h`.

---

## Factory Function — `src/distance.cpp`

```cpp
template <typename T> Distance<T> *get_distance_function(Metric m);
```

Specialized for `float`, `int8_t`, `uint8_t`. Selection logic:

**`float`:**
| Metric | AVX2 | AVX only | Scalar |
|--------|------|----------|--------|
| L2 | `DistanceL2Float` | `AVXDistanceL2Float` | `SlowDistanceL2<float>` |
| COSINE | `DistanceCosineFloat` | (same) | (same) |
| INNER_PRODUCT | `AVXDistanceInnerProductFloat` | (same) | (same) |
| FAST_L2 | `DistanceFastL2<float>` | (same) | (same) |

**`int8_t`:**
| Metric | AVX2 | AVX only | Scalar |
|--------|------|----------|--------|
| L2 | `DistanceL2Int8` | `AVXDistanceL2Int8` | `SlowDistanceL2<int8_t>` |
| COSINE | `DistanceCosineInt8` | (same) | (same) |

**`uint8_t`:**
| Metric | Implementation |
|--------|----------------|
| L2 | `DistanceL2UInt8` |
| COSINE | `SlowDistanceCosineUInt8` |

Note: Inner product is only supported for `float`. `uint8_t` has no SIMD-optimized distance classes.

---

## Utility Functions Used by Distance Module — `include/utils.h`

```cpp
template <typename T> inline float get_norm(T *arr, size_t dim);
// Returns sqrt(Σ arr[i]²)

template <typename T = float> inline void normalize(T *arr, size_t dim);
// In-place: arr[i] /= get_norm(arr, dim)
```

Used by `AVXNormalizedCosineDistanceFloat::preprocess_base_points()` and `normalize_and_copy()`.

---

## Template Instantiations

### `src/distance.cpp` (bottom)
```cpp
template class DistanceInnerProduct<float>;
template class DistanceInnerProduct<int8_t>;
template class DistanceInnerProduct<uint8_t>;
template class DistanceFastL2<float>;
template class DistanceFastL2<int8_t>;
template class DistanceFastL2<uint8_t>;
template class SlowDistanceL2<float>;
template class SlowDistanceL2<int8_t>;
template class SlowDistanceL2<uint8_t>;
template Distance<float> *get_distance_function(Metric m);
template Distance<int8_t> *get_distance_function(Metric m);
template Distance<uint8_t> *get_distance_function(Metric m);
```

### `src/pq_l2_distance.cpp` (bottom)
```cpp
template class PQL2Distance<int8_t>;
template class PQL2Distance<uint8_t>;
template class PQL2Distance<float>;
```

---

## Key `#define` Guards

| Guard | Where | Effect |
|-------|-------|--------|
| `_WINDOWS` | Throughout | Enables Windows-specific SIMD paths, `__declspec(align)`, `m256_f32[]` direct access |
| `USE_AVX2` | `distance.cpp` | Enables AVX2 codepath in `DistanceL2Float`, `DistanceL2Int8` |
| `__AVX__` | `distance.cpp` | Enables AVX codepath in `DistanceFastL2::norm()` |
| `__SSE2__` | `distance.cpp` | Enables SSE2 codepath in `DistanceInnerProduct::inner_product()` |
| `__GNUC__` | `distance.cpp` | Gates GCC/Clang-specific SIMD code |
| `EXEC_ENV_OLS` | `pq_l2_distance.h/cpp` | Switches to `MemoryMappedFiles` based loading instead of disk I/O |
