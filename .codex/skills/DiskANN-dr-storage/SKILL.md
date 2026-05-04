---
name: DiskANN-dr-storage
description: Use when working with the storage module of DiskANN — vector data stores (in-memory and PQ-compressed) and graph adjacency stores
---

# DiskANN Storage Module

## Module Purpose & Capabilities

The storage module provides the persistent data layer for DiskANN's index structures. It defines abstract interfaces and concrete implementations for two orthogonal storage concerns: **vector data** (the high-dimensional points being indexed) and **graph adjacency** (the Vamana graph edges connecting those points). The module exposes two data store backends — `InMemDataStore` for full-precision in-memory vectors and `PQDataStore` for PQ-compressed vectors — plus one graph store backend, `InMemGraphStore`. All are accessed through abstract base classes (`AbstractDataStore<data_t>`, `AbstractGraphStore`) that allow the core index logic in `index.cpp` and `pq_flash_index.cpp` to be agnostic of the storage strategy.

Key capabilities exposed externally:
- Load/save vectors and graphs to/from DiskANN binary file formats
- Populate stores from raw pointers or files, with automatic alignment and metric-specific preprocessing (e.g., normalization for cosine)
- Per-vector random access: get, set, prefetch, move, copy
- Distance computation (full-precision via `Distance<data_t>`, or PQ-approximate via `QuantizedDistance<data_t>`)
- Dynamic resize (expand/shrink) for streaming insert/delete workloads
- Graph neighbor list manipulation (get, set, add, clear, swap)
- Medoid calculation for choosing the search entry point

---

## Core Design Logic

### Why Abstract Interfaces?

The index classes (`Index<data_t>` in `index.cpp`, `PQFlashIndex` in `pq_flash_index.cpp`) interact with data and graph stores through abstract base classes. This **strategy pattern** decouples the search/build algorithms from the storage representation. A single `iterate_to_fixed_point` implementation works identically whether the underlying data is full-precision or PQ-compressed — the data store's `get_distance()` and `preprocess_query()` methods absorb the differences.

Design decision: `AbstractDataStore<data_t>` is a **class template** parameterized on the element type (`float`, `int8_t`, `uint8_t`), while `AbstractGraphStore` is **not** templated because graph adjacency lists are always `std::vector<location_t>` (i.e., `uint32_t` indices) regardless of vector type.

### Distance Function Co-location

`InMemDataStore` owns a `std::unique_ptr<Distance<data_t>>` and `PQDataStore` owns both a `Distance<data_t>` (full-precision, used by callers who need `get_dist_fn()`) and a `QuantizedDistance<data_t>` (PQ-approximate, used internally for `get_distance()`). Embedding the distance function inside the data store avoids copying data out for every distance computation and enables the store to do metric-specific preprocessing (normalization) at ingestion time via `Distance::preprocessing_required()` and `preprocess_base_points()`.

### Alignment Requirements

SIMD distance functions (AVX2/SSE) require vectors to be aligned to specific boundaries. `InMemDataStore` pads the stored dimension to `_aligned_dim = ROUND_UP(dim, distance_fn->get_required_alignment())` (typically alignment factor 8, so a 100-dim vector becomes 104-dim with 4 zero-padded elements). Memory is allocated with `alloc_aligned()` using `8 * sizeof(data_t)` byte alignment. This alignment is invisible to callers who always use `get_dims()` for the logical dimension and `get_aligned_dim()` for the internal padded dimension.

`PQDataStore` does **not** require alignment (its `get_alignment_factor()` returns 1) because PQ distances are computed via table lookups on `uint8_t` chunk codes, not raw vector SIMD.

### PQDataStore Is Partially Implemented

Many `PQDataStore` methods throw `std::logic_error("Not implemented yet")`:
- `populate_data(const data_t*, ...)` — from raw pointer
- `get_vector()` — has an inverted condition bug: the if-branch (`i < capacity()`) throws `logic_error("Not implemented yet")` for valid indices, while the else-branch (`i >= capacity()`) throws `ANNException` for out-of-range. Valid calls always hit "not implemented."
- `set_vector()` — compression not implemented
- `move_vectors()` — streaming not yet supported for PQ
- `expand()`, `shrink()` — dynamic resize not supported
- `get_distance()` single-point overloads — only batch overloads with scratch are implemented
- `calculate_medoid()` — returns a random point instead of true medoid

This is intentional: PQDataStore is primarily used during search in `pq_flash_index.cpp` where only `preprocess_query()` and batch `get_distance()` are needed. The `REFACTOR TODO` comments indicate plans to complete these.

### Graph File Format

The Vamana graph is stored as a binary file with a 24-byte header:
```
[8 bytes: total file size] [4 bytes: max_observed_degree] [4 bytes: start/entry point] [8 bytes: num_frozen_points]
```
Followed by per-node adjacency lists:
```
[4 bytes: neighbor count k] [k * 4 bytes: neighbor IDs]
```
This is read/written by `InMemGraphStore::load_impl()` / `save_graph()`.

---

## State Flow

### Data Store Lifecycle

1. **Construction**: `InMemDataStore(capacity, dim, distance_fn)` allocates an aligned, zero-initialized flat array of `capacity * aligned_dim` elements. `PQDataStore(dim, capacity, num_chunks, dist_fn, pq_dist_fn)` stores parameters but does not allocate `_quantized_data` until `load()` or `populate_data()`.

2. **Population** (static index build):
   - `populate_data(vectors, num_pts)` — copies vectors into aligned storage, then calls `_distance_fn->preprocess_base_points()` if needed (e.g., normalization for cosine). Uses `memmove` per-vector to handle dim→aligned_dim padding.
   - `populate_data(filename, offset)` — reads from a `.bin` file via `copy_aligned_data_from_file()`.
   - For PQ: `populate_data(filename, offset)` runs the full PQ pipeline — training pivots via `generate_quantized_data()`, then loading the compressed vectors and pivot tables.

3. **Streaming insert** (dynamic index): `set_vector(loc, vector)` writes a single vector at a specific slot, zeroing alignment padding and applying preprocessing. `resize()` delegates to `expand()`/`shrink()` which reallocate the flat array.

4. **Search access**: `get_distance(query, loc)` computes distance between a query and a stored vector. `prefetch_vector(loc)` issues CPU prefetch hints. `preprocess_query()` copies the query into scratch space (InMem) or computes PQ chunk-distance tables (PQ). PQ's `preprocess_query()` checks both `scratch` and `scratch->pq_scratch()` for null, throwing `ANNException` if either is null.

5. **Persistence**: `save(filename, num_pts)` writes `num_pts` vectors in base (unaligned) dimensions via `save_data_in_base_dimensions()`. `load(filename)` reads and re-aligns.

### Graph Store Lifecycle

1. **Construction**: `InMemGraphStore(total_pts, reserve_graph_degree)` creates `_graph` as `vector<vector<uint32_t>>` of size `total_pts`, each inner vector reserved to `reserve_graph_degree`.

2. **Build**: `add_neighbour(i, id)` appends to `_graph[i]` and checks `_graph[i].size()` (the stored list size after appending) to update `_max_observed_degree`. `set_neighbours(i, neighbors)` bulk-assigns and checks `neighbours.size()` (the input parameter size, not `_graph[i].size()`) to update `_max_observed_degree`.

3. **Search**: `get_neighbours(i)` returns `const vector<location_t>&` — zero-copy, no allocation.

4. **Persistence**: `store()` → `save_graph()` writes the Vamana binary format. It initializes `index_size = 24` (the header size), accumulates per-node sizes during the write loop, then seeks back to rewrite the final `index_size` and recomputed `max_degree` in the header. `load()` → `load_impl()` reads it back, resizing `_graph` if needed.

### Memory Management

- `InMemDataStore::_data` is allocated with `alloc_aligned()` and freed with `aligned_free()`. On non-Windows platforms, `expand()` allocates a new buffer, copies, and frees the old one (no `realloc_aligned`). On Windows, `realloc_aligned` is used.
- `PQDataStore::_quantized_data` is similarly `alloc_aligned()` / `aligned_free()`.
- `InMemGraphStore::_graph` is `std::vector<std::vector<uint32_t>>` — standard heap management.

### Thread Safety

The stores themselves are **not thread-safe**. Comments in `AbstractGraphStore` explicitly state: "not synchronised, user should use lock when necessary." The `Index` class in `index.cpp` handles locking externally (per-node locks for graph mutations, a global lock for resize operations). `InMemDataStore` includes `<shared_mutex>` in its header but does not use it internally — the intent is that the caller manages concurrency.

---

## Common Modification Scenarios

### Scenario 1: Add a New Data Store Backend (e.g., Disk-Based or GPU)

**Files to modify:**
1. Create `include/my_data_store.h` and `src/my_data_store.cpp` inheriting from `AbstractDataStore<data_t>` ([include/abstract_data_store.h](include/abstract_data_store.h)).
2. Implement all pure virtual methods: `load()`, `save()`, `get_aligned_dim()`, `populate_data()` (both overloads), `extract_data_to_bin()`, `get_vector()`, `set_vector()`, `prefetch_vector()`, `move_vectors()`, `copy_vectors()`, `preprocess_query()`, all `get_distance()` overloads, `calculate_medoid()`, `get_dist_fn()`, `get_alignment_factor()`, `expand()`, `shrink()`.
3. Add the `.cpp` to `src/CMakeLists.txt`.
4. Wire it into `IndexFactory` ([include/index_factory.h](include/index_factory.h), [src/index_factory.cpp](src/index_factory.cpp)) so that `Index` can instantiate your store based on config.
5. Add explicit template instantiations for `float`, `int8_t`, `uint8_t` at the bottom of the `.cpp` file.

**Key contract to honor:** `get_distance()` must work with preprocessed queries (the query passed through `preprocess_query()` first). The batch `get_distance()` overloads receive an `AbstractScratch<data_t>*` that may contain metric-specific scratch data.

### Scenario 2: Add a New Graph Store Backend (e.g., Memory-Mapped or Distributed)

**Files to modify:**
1. Create `include/my_graph_store.h` and `src/my_graph_store.cpp` inheriting from `AbstractGraphStore` ([include/abstract_graph_store.h](include/abstract_graph_store.h)).
2. Implement: `load()`, `store()`, `get_neighbours()`, `add_neighbour()`, `clear_neighbours()`, `swap_neighbours()`, `set_neighbours()`, `resize_graph()`, `clear_graph()`, `get_max_observed_degree()`, `get_max_range_of_graph()`.
3. The graph file format is defined by `load()`/`store()` — you can use a different format internally but must be able to read/write the Vamana binary format for compatibility.
4. Update `src/CMakeLists.txt` and the `Index` constructor in `src/index.cpp` to accept your store.

**Critical:** `get_neighbours()` returns `const std::vector<location_t>&`. If your backend doesn't use `std::vector` internally, you'll need a caching/conversion layer.

### Scenario 3: Add a New Distance Metric to an Existing Data Store

**No changes to the storage module are needed.** The data store delegates all distance computation to its `Distance<data_t>` object. Steps:
1. Add a new `Distance<data_t>` subclass in [include/distance.h](include/distance.h) / [src/distance.cpp](src/distance.cpp).
2. If the new metric requires preprocessing (like cosine normalization), implement `preprocessing_required()` returning `true` and `preprocess_base_points()`.
3. If it requires different alignment, override `get_required_alignment()`.
4. The `InMemDataStore` will automatically align to the new alignment factor and call preprocessing during `populate_data()` and `set_vector()`.

### Scenario 4: Enable Dynamic Resize for PQDataStore

Currently `PQDataStore::expand()` and `shrink()` throw `std::logic_error`. To enable streaming with PQ:
1. Implement `expand()` in [src/pq_data_store.cpp](src/pq_data_store.cpp) — allocate a new `uint8_t*` buffer of `new_size * _num_chunks`, copy existing data, free old buffer.
2. Implement `shrink()` similarly.
3. Implement `set_vector()` — this requires quantizing a single full-precision vector to PQ codes, which means the PQ codebook must be accessible.
4. Implement `move_vectors()` using `memmove` on `_quantized_data` with stride `_num_chunks`.

### Scenario 5: Change the Graph Serialization Format

Modify `InMemGraphStore::save_graph()` and `load_impl()` in [src/in_mem_graph_store.cpp](src/in_mem_graph_store.cpp). The current format is sequential with variable-length adjacency lists. A fixed-width format (padding all lists to `max_degree`) would enable memory-mapped access but waste space. The header structure (24 bytes) encodes `file_size`, `max_observed_degree`, `start` (entry point), and `num_frozen_points` — any format change must preserve these semantics or update all callers in `index.cpp`.
