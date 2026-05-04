---
name: DiskANN-dr-infrastructure
description: Use when working with the infrastructure module of DiskANN — IO, scratch spaces, data structures, utilities, math, logging, parameters, and type definitions
---

# DiskANN Infrastructure Module — Skill Document

This module provides the foundational plumbing for the entire DiskANN codebase: file I/O with binary format conventions, aligned memory allocation, scratch space pooling for thread-safe search, neighbor data structures for graph traversal, logging abstraction, math utilities (MKL-based k-means/distance), parameter management, custom exceptions, and platform-abstraction layers.

## 1. Module Purpose & Capabilities

### File I/O & Binary Format (`include/utils.h`, `src/utils.cpp`)

DiskANN uses a **standard binary format** for all vector data files:

```
[4 bytes: int32 npts][4 bytes: int32 ndims][npts * ndims * sizeof(T) bytes: data]
```

Key functions:
- **`get_bin_metadata(filename, nrows, ncols)`** — reads first 8 bytes to extract npts/ndims
- **`load_bin<T>(filename, data, npts, dim)`** — allocates and loads full binary file
- **`save_bin<T>(filename, data, npts, ndims)`** — writes data in standard binary format
- **`load_aligned_bin<T>(filename, data, npts, dim, rounded_dim)`** — loads with `ROUND_UP(dim, 8)` alignment, zero-padding each row
- **`copy_aligned_data_from_file()`** — copies into pre-allocated aligned buffer
- **`load_truthset()`** / **`load_range_truthset()`** — loads ground truth (ids + optional distances)
- **`save_Tvecs()`** — saves in `.fvecs`/`.ivecs` format (dim_u32 per row, then data)
- **`normalize_data_file()`** / **`block_convert()`** — normalizes float vectors to unit norm

Graph files have a different header: `[8B expected_file_size][4B max_degree][4B start_node][8B frozen_pts]`.

### Aligned Memory Allocation (`include/utils.h`)

- **`alloc_aligned(void** ptr, size, align)`** — `aligned_alloc` on Linux, `_aligned_malloc` on Windows. Size must be a multiple of align.
- **`aligned_free(ptr)`** — platform-correct free.
- **`realloc_aligned()`** — Windows only (`_aligned_realloc`).

Macros: `ROUND_UP(X,Y)`, `ROUND_DOWN(X,Y)`, `IS_ALIGNED(X,Y)`, `IS_512_ALIGNED`, `IS_4096_ALIGNED`.

### Recall Calculation (`src/utils.cpp`)

- **`calculate_recall(num_queries, gold_std, gs_dist, dim_gs, our_results, dim_or, recall_at)`** — computes recall@k with tie-breaking on ground-truth distances.
- **`calculate_range_search_recall()`** — for range search ground truth (variable-length result lists).

### Neighbor Data Structures (`include/neighbor.h`)

- **`Neighbor`** — `{unsigned id, float distance, bool expanded}`. Ordered by distance (tie-break on id). Equality by id only.
- **`NeighborPriorityQueue`** — fixed-capacity sorted array with cursor-based expansion tracking. Key operations:
  - `insert(nbr)` — binary-search insert, drops if at capacity and worse than worst. O(capacity) due to memmove.
  - `closest_unexpanded()` — returns and marks the nearest unexpanded neighbor, advances cursor.
  - `has_unexpanded_node()` — checks if cursor < size.

This is the **core search data structure** — used as the "best L nodes" working set during greedy search.

### Scratch Space System (`include/scratch.h`, `include/abstract_scratch.h`)

Thread-safe pre-allocated working memory for search/index operations.

**Hierarchy:**
- `AbstractScratch<T>` — holds `_aligned_query_T` (aligned query copy) and `_pq_scratch` (PQ distance scratch).
- `InMemQueryScratch<T>` — for in-memory index search/build. Constructor calls `resize_for_new_L(max(search_l, indexing_l))`. Contains:
  - `_pool` (vector of explored Neighbors, init 3L+R)
  - `_best_l_nodes` (NeighborPriorityQueue, capacity L)
  - `_occlude_factor` (floats for pruning, size maxc)
  - `_inserted_into_pool_rs` (robin_set for visited tracking)
  - `_inserted_into_pool_bs` (dynamic_bitset, same purpose, O(1) lookup)
  - `_id_scratch` / `_dist_scratch` (buffers for neighbor retrieval, reserved to `ceil(1.5 * defaults::GRAPH_SLACK_FACTOR * R)`)
  - `_expanded_nodes_set/vec`, `_occlude_list_output` (for delete/consolidate)
- `SSDQueryScratch<T>` — for disk index. Has `coord_scratch`, `sector_scratch` (aligned sector buffer), `visited` set, `retset`, `full_retset`.
- `SSDThreadData<T>` — wraps SSDQueryScratch + IOContext.

**`ScratchStoreManager<T>`** — RAII wrapper that pops a scratch from a `ConcurrentQueue<T*>`, returns it on destruction with `clear()`. Pattern:
```cpp
ScratchStoreManager<InMemQueryScratch<T>> manager(scratch_pool);
auto scratch = manager.scratch_space();
// use scratch...
// ~ScratchStoreManager returns scratch to pool
```

### Parameters (`include/parameters.h`, `include/defaults.h`)

- **`IndexWriteParameters`** — immutable struct: `search_list_size` (L), `max_degree` (R), `saturate_graph`, `max_occlusion_size` (C), `alpha`, `num_threads`, `filter_list_size` (Lf).
- **`IndexWriteParametersBuilder`** — fluent builder. Required: L and R. Optional with defaults from `defaults.h`. Note: `with_filter_list_size(0)` applies `_filter_list_size = filter_list_size == 0 ? _search_list_size : filter_list_size`, so passing 0 defaults Lf to L.
- **`IndexSearchParams`** — `initial_search_list_size`, `num_search_threads`.

Key defaults: `ALPHA=1.2`, `MAX_OCCLUSION_SIZE=750`, `MAX_DEGREE=64`, `BUILD_LIST_SIZE=100`, `SECTOR_LEN=4096`, `MAX_N_SECTOR_READS=128`.

### Math Utilities (`include/math_utils.h`, `src/math_utils.cpp`)

MKL-dependent. Uses `cblas_sgemm` for batched distance computation.

- **`math_utils::calc_distance()`** — naive L2 squared distance
- **`math_utils::compute_vecs_l2sq()`** — parallel L2 norm via `cblas_snrm2`
- **`math_utils::compute_closest_centers()`** — finds k-nearest centers using MKL matrix multiply for batch distance: `D = norms_data + norms_centers - 2 * data * centers^T`. Internally `compute_closest_centers_in_block` issues three `cblas_sgemm` calls:
  1. `docs_l2sq × ones_a^T`, alpha=1.0, beta=0.0 → initializes dist_matrix with per-point norms
  2. `ones_b × centers_l2sq^T`, alpha=1.0, beta=1.0 → accumulates per-center norms
  3. `data × centers^T`, alpha=-2.0, beta=1.0 → accumulates cross-term
- **`math_utils::rotate_data_randomly()`** — `cblas_sgemm` rotation
- **`math_utils::process_residuals()`** — subtracts/adds nearest center from each vector
- **`kmeans::run_lloyds()`** — full k-means with early stopping on residual convergence
- **`kmeans::kmeanspp_selecting_pivots()`** — k-means++ initialization (up to 8M points)

### Logging (`include/logger.h`, `include/logger_impl.h`, `src/logger.cpp`)

Two modes controlled by `ENABLE_CUSTOM_LOGGER`:
- **Standard mode** (default on Linux): `diskann::cout` and `diskann::cerr` are aliases for `std::cout`/`std::cerr`.
- **Custom mode** (OLS environment): `ANNStreamBuf` wraps stdout/stderr as `std::basic_streambuf`. Characters accumulate in a 1024-byte buffer; on `sync()`/overflow, invokes `g_logger(logLevel, str)`. Set via `SetCustomLogger()`.

LogLevel: `LL_Info` (stdout), `LL_Error` (stderr).

### Custom Exceptions (`include/ann_exception.h`, `include/exceptions.h`, `src/ann_exception.cpp`)

- **`ANNException`** extends `std::runtime_error`. Two constructors:
  - Simple: `(message, errorCode)`
  - Detailed: `(message, errorCode, funcSig, fileName, lineNum)` — formats as `[FUNC: ...][FILE: ...][LINE: ...] message`
- **`FileException`** extends ANNException — for file I/O errors with system_error code.
- **`NotImplementedException`** extends `std::logic_error`.

`__FUNCSIG__` is mapped to `__PRETTY_FUNCTION__` on non-Windows.

### Types & Wrappers (`include/types.h`, `include/any_wrappers.h`)

DiskANN uses **type-erased types** for runtime flexibility:
- `location_t` = `uint32_t` — internal point index
- `DataType` / `TagType` / `LabelType` = `std::any`
- `TagVector` / `DataVector` / `Labelvector` = `AnyWrapper::AnyVector` (wraps `std::vector<T>` ref in `std::any`)
- `TagRobinSet` = `AnyWrapper::AnyRobinSet`

`AnyWrapper::AnyReference` stores a pointer via `std::any`, no ownership. `get<T>()` casts back.

### Timer (`include/timer.h`)

`diskann::Timer` — wraps `std::chrono::high_resolution_clock`. Methods: `reset()`, `elapsed()` (microseconds), `elapsed_seconds()`, `elapsed_seconds_for_step(step_name)`.

### Concurrent Data Structures

**`ConcurrentQueue<T>`** (`include/concurrent_queue.h`):
- Mutex-protected `std::queue` with condition variable notifications.
- `push()`, `pop()` (returns null_T if empty), `wait_for_push_notify()`, `wait_for_pop_notify()`. Both wait functions default to a 10-microsecond timeout (`chrono_us_t{10}`).
- Used for scratch space pooling.

**`natural_number_map<Key, Value>`** (`include/natural_number_map.h`, `src/natural_number_map.cpp`):
- Dense map for consecutive natural number keys. Backed by `vector<Value>` + `dynamic_bitset` for presence.
- O(1) set/get/erase/contains. Memory = O(max_key).
- Iteration via `find_first()` / `find_next()` using bitset scanning.
- Used for location-to-tag mapping.

**`natural_number_set<T>`** (`include/natural_number_set.h`, `src/natural_number_set.cpp`):
- Dense set for natural numbers. `vector<T>` + `dynamic_bitset`.
- `insert()`, `pop_any()` (LIFO from vector), `is_in_set()` (O(1) via bitset).
- `pop_any()` throws `diskann::ANNException("No values available", -1, __FUNCSIG__, __FILE__, __LINE__)` when the set is empty.
- Used for tracking delete candidates.

### Locking (`include/locking.h`, `include/windows_slim_lock.h`)

Platform abstraction:
- Linux: `non_recursive_mutex` = `std::mutex`, `LockGuard` = `std::lock_guard<std::mutex>`
- Windows: `windows_exclusive_slim_lock` (8 bytes, wraps SRWLOCK) + matching guard. Used for per-vector locking where std::mutex (80 bytes) is too heavy.

### Query Statistics (`include/percentile_stats.h`)

`QueryStats` — per-query metrics: `total_us`, `io_us`, `cpu_us`, `n_4k/8k/12k`, `n_ios`, `read_size`, `n_cmps`, `n_cache_hits`, `n_hops`.

`get_percentile_stats()` / `get_mean_stats()` — aggregate over query arrays with lambda accessors.

### Other Files

- **`tag_uint128.h`** — 128-bit tag struct with Murmur-inspired `Hash128to64` specialization for `std::hash`.
- **`program_options_utils.hpp`** — CLI description string constants for all app parameters.
- **`boost_dynamic_bitset_fwd.h`** — forward declaration to avoid boost dependency in public headers.
- **`windows_customizations.h`** — `DISKANN_DLLEXPORT` macro (dllexport/dllimport on Windows, empty on Linux).
- **`common_includes.h`** — aggregated standard library includes.

---

## 2. Core Design Logic

### Why Binary Format with 8-byte Header?
The `[npts_i32][ndims_i32]` header enables quick metadata reads without loading data. All DiskANN tools expect this format. The graph file has a larger header (24 bytes) because it needs `expected_file_size` for validation, `max_degree` for buffer sizing, `start_node` for search entry, and `frozen_pts` count.

### Why Scratch Space Pooling?
Search and index-build are **multi-threaded** but each thread needs large working buffers (neighbor lists, visited sets, PQ tables, sector buffers). Allocating per-call is expensive. Instead:
1. At index creation, N scratch objects are created and pushed into a `ConcurrentQueue`.
2. Each search thread pops a scratch via `ScratchStoreManager` (RAII).
3. On completion, the scratch is `clear()`-ed and returned to the pool.
4. If pool is empty, the thread busy-waits with `wait_for_push_notify()`.

This avoids allocation overhead and keeps memory bounded.

### Why Two Visited-Tracking Structures?
`InMemQueryScratch` has both `_inserted_into_pool_rs` (robin_set) and `_inserted_into_pool_bs` (dynamic_bitset):
- The bitset gives O(1) lookup by point ID but requires memory proportional to the max point ID.
- The robin_set is used as a fallback/complement for scenarios where the point space is sparse.

### Why NeighborPriorityQueue Instead of std::priority_queue?
The sorted-array design with cursor tracking supports the specific access pattern of greedy search:
1. Insert candidates maintaining sort order.
2. Pop closest unexpanded (cursor-based, no actual removal).
3. Fixed capacity means worst candidates are silently dropped.
This is more cache-friendly than a heap and supports the "beam search" pattern naturally.

### Why natural_number_map Instead of std::unordered_map?
For the location-to-tag map where keys are dense consecutive integers (0..N), a vector+bitset is:
- More memory efficient (no hash table overhead per entry)
- Faster (direct array index vs. hash computation)
- Supports ordered iteration via bitset scanning

### Why Type-Erased Types (std::any)?
`DataType`, `TagType`, `LabelType` are `std::any` to support **runtime-selected** data types (float, int8, uint8) and tag types (uint32, uint64, uint128) without templating the entire API surface. The `index_factory` pattern uses these to construct appropriately-typed indices.

### Platform Abstraction Strategy
The codebase uses:
- `DISKANN_DLLEXPORT` for Windows DLL boundaries
- `__FUNCSIG__` / `__PRETTY_FUNCTION__` mapping
- Slim SRWLOCK vs std::mutex selection
- `aligned_alloc` vs `_aligned_malloc`
- `#ifdef _WINDOWS` guards throughout

---

## 3. Core Data Structures

See [reference.md](reference.md) for the complete type catalog with all fields.

Key relationships:
- `InMemQueryScratch` contains `NeighborPriorityQueue` (best_l_nodes) and vectors of `Neighbor`
- `ScratchStoreManager` manages `ConcurrentQueue<InMemQueryScratch<T>*>` or `ConcurrentQueue<SSDThreadData<T>*>`
- `natural_number_map` is templated as `<uint32_t, TagType>` for the location-to-tag mapping in Index
- `IndexWriteParametersBuilder` produces `IndexWriteParameters`
- `ANNException` → `FileException` inheritance

---

## 4. State Flow

### Scratch Lifecycle (In-Memory Search)

```
Index construction:
  for i in 0..num_threads:
    create InMemQueryScratch(L, indexing_L, R, maxc, dim, aligned_dim, align)
    push into ConcurrentQueue<InMemQueryScratch<T>*>

Per-query:
  ScratchStoreManager ctor:
    scratch = queue.pop()       // blocks if empty via wait_for_push_notify()
  
  // Search uses scratch->best_l_nodes(), scratch->pool(), etc.
  
  ScratchStoreManager dtor:
    scratch->clear()            // resets all internal buffers
    queue.push(scratch)
    queue.push_notify_all()     // wake waiting threads
```

### Logging Flow

```
Standard mode (Linux):
  diskann::cout << "msg" → std::cout << "msg"

Custom mode (OLS):
  diskann::cout << "msg"
  → ANNStreamBuf::overflow() accumulates in _buf
  → ANNStreamBuf::sync() on endl/flush
    → flush() → logImpl(str, len)
      → g_logger(LL_Info, str)    // user-provided callback
```

### File I/O Pattern (Load Binary)

```
load_bin<T>(filename, data, npts, dim):
  open file binary
  read 4 bytes → npts_i32
  read 4 bytes → dim_i32
  data = new T[npts * dim]
  read npts*dim*sizeof(T) bytes → data

load_aligned_bin<T>(filename, data, npts, dim, rounded_dim):
  rounded_dim = ROUND_UP(dim, 8)
  alloc_aligned(&data, npts * rounded_dim * sizeof(T), 8*sizeof(T))
  for each point:
    read dim elements
    zero-pad to rounded_dim
```

### K-Means Flow (`src/math_utils.cpp`)

```
run_lloyds(data, num_points, dim, centers, num_centers, max_reps):
  compute docs_l2sq
  repeat up to max_reps:
    lloyds_iter:
      compute_closest_centers(data, centers)  // MKL batched
      update centers as cluster means
      compute residual
    if residual change < 0.001%: break
```

---

## 5. Common Modification Scenarios

### Scenario 1: Adding a New Scratch Buffer Field

If a new algorithm needs additional per-thread workspace:

1. Add the field to `InMemQueryScratch<T>` in `include/scratch.h` (private section).
2. Add an accessor method in the public section.
3. Initialize in the constructor (allocate appropriately sized buffer).
4. Add cleanup in `clear()` method (reset, not deallocate — the scratch is reused).
5. If it needs to resize with L, update `resize_for_new_L()`.
6. Deallocate in the destructor.

Files to modify: `include/scratch.h`, `src/scratch.cpp` (not listed but contains implementations).

### Scenario 2: Adding a New Default Parameter

1. Add the constant in `include/defaults.h` under `diskann::defaults`.
2. If it's a build parameter, add to `IndexWriteParameters` struct fields.
3. Add a `with_<param>()` method to `IndexWriteParametersBuilder`.
4. Update the `build()` method and constructor to pass the new field.
5. Update CLI descriptions in `include/program_options_utils.hpp`.
6. Update the app-level argument parsing (in `apps/` files).

### Scenario 3: Adding a New Distance Metric to Math Utils

1. Add the function declaration in `include/math_utils.h`.
2. Implement in `src/math_utils.cpp`. Use MKL BLAS for batched operations where possible.
3. If it needs to plug into the main distance framework, also update `include/distance.h` and `src/distance.cpp` (distance module, not infrastructure).

### Scenario 4: Adding a New Exception Type

1. Declare in `include/ann_exception.h` extending `ANNException`.
2. Implement constructor in `src/ann_exception.cpp` following `FileException` pattern.
3. Use `__FUNCSIG__`, `__FILE__`, `__LINE__` at throw sites for diagnostics.

### Scenario 5: Supporting a New Tag Type

1. Define the type (like `tag_uint128` in `include/tag_uint128.h`).
2. Provide `operator==`, `operator=`, and `std::hash` specialization.
3. Add explicit template instantiation in `src/natural_number_map.cpp` for `natural_number_map<uint32_t, YourType>`.
4. The `std::any`-based `TagType` in `types.h` will wrap it automatically.
