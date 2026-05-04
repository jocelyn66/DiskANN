# DiskANN Infrastructure Module — Reference

Complete type and function catalog for the infrastructure module.

---

## File: `include/types.h`

```cpp
namespace diskann {
  typedef uint32_t location_t;      // Internal point index type
  using DataType  = std::any;       // Runtime-selected vector element type
  using TagType   = std::any;       // Runtime-selected external ID type
  using LabelType = std::any;       // Runtime-selected filter label type
  using TagVector    = AnyWrapper::AnyVector;
  using DataVector   = AnyWrapper::AnyVector;
  using Labelvector  = AnyWrapper::AnyVector;
  using TagRobinSet  = AnyWrapper::AnyRobinSet;
}
```

---

## File: `include/any_wrappers.h`

```
AnyWrapper::AnyReference
  - Stores a pointer via std::any (no ownership)
  - template<T> AnyReference(T& reference)  // wraps &reference
  - template<T> T& get()                     // casts back

AnyWrapper::AnyRobinSet : AnyReference
  - Wraps tsl::robin_set<T>

AnyWrapper::AnyVector : AnyReference
  - Wraps std::vector<T>
```

---

## File: `include/defaults.h`

```cpp
namespace diskann::defaults {
  ALPHA                    = 1.2f
  NUM_THREADS              = 0      // 0 → omp_get_num_procs()
  MAX_OCCLUSION_SIZE       = 750
  HAS_LABELS               = false
  FILTER_LIST_SIZE         = 0
  NUM_FROZEN_POINTS_STATIC = 0
  NUM_FROZEN_POINTS_DYNAMIC= 1
  GRAPH_SLACK_FACTOR       = 1.3f
  MAX_GRAPH_DEGREE         = 512    // SSD index max
  SECTOR_LEN               = 4096
  MAX_N_SECTOR_READS       = 128
  MAX_DEGREE               = 64
  BUILD_LIST_SIZE          = 100
  SATURATE_GRAPH           = false
  SEARCH_LIST_SIZE         = 100
}
```

---

## File: `include/parameters.h`

### `IndexWriteParameters` (immutable)
| Field | Type | Description |
|---|---|---|
| `search_list_size` | uint32_t | Build-time search list size (L) |
| `max_degree` | uint32_t | Max graph out-degree (R) |
| `saturate_graph` | bool | Whether to fill every node to max_degree |
| `max_occlusion_size` | uint32_t | Max candidates for occlusion pruning (C) |
| `alpha` | float | Graph density/diameter control |
| `num_threads` | uint32_t | Build parallelism |
| `filter_list_size` | uint32_t | Filtered search list size (Lf) |

### `IndexWriteParametersBuilder`
- Constructor: `(search_list_size, max_degree)` — both required
- `with_max_occlusion_size(uint32_t)` → default 750
- `with_saturate_graph(bool)` → default false
- `with_alpha(float)` → default 1.2
- `with_num_threads(uint32_t)` → default omp_get_num_procs()
- `with_filter_list_size(uint32_t)` → default = search_list_size
- `build()` → `IndexWriteParameters`

### `IndexSearchParams`
| Field | Type |
|---|---|
| `initial_search_list_size` | uint32_t |
| `num_search_threads` | uint32_t |

---

## File: `include/neighbor.h`

### `Neighbor`
```cpp
struct Neighbor {
  unsigned id;
  float distance;
  bool expanded;
  operator<  → by distance, then id
  operator== → by id only
};
```

### `NeighborPriorityQueue`
- Fixed-capacity sorted array with expansion cursor
- `_data`: `vector<Neighbor>` (capacity+1 to handle inserts)
- `_size`, `_capacity`, `_cur` (index of first unexpanded)

| Method | Complexity | Description |
|---|---|---|
| `insert(Neighbor)` | O(capacity) | Binary search + memmove. Deduplicates by id. Drops if full and worse than worst. |
| `closest_unexpanded()` | O(1) amortized | Returns _data[_cur], marks expanded, advances cursor |
| `has_unexpanded_node()` | O(1) | `_cur < _size` |
| `reserve(capacity)` | O(n) | Resize backing vector |
| `clear()` | O(1) | Reset size and cursor |
| `operator[]` | O(1) | Direct access |

---

## File: `include/scratch.h` + `include/abstract_scratch.h`

### `AbstractScratch<T>`
```
protected:
  T* _aligned_query_T = nullptr     // aligned copy of query vector
  PQScratch<T>* _pq_scratch = nullptr
```

### `InMemQueryScratch<T> : AbstractScratch<T>`

Constructor: `(search_l, indexing_l, r, maxc, dim, aligned_dim, alignment_factor, init_pq_scratch)`

| Field | Type | Init Size | Purpose |
|---|---|---|---|
| `_L` | uint32_t | search_l | Current search list size |
| `_R` | uint32_t | r | Max degree |
| `_maxc` | uint32_t | maxc | Max occlusion candidates |
| `_pool` | vector\<Neighbor\> | 3L+R | All explored neighbors |
| `_best_l_nodes` | NeighborPriorityQueue | L+1 | Best L candidates (working set) |
| `_occlude_factor` | vector\<float\> | maxc | Pruning factors |
| `_inserted_into_pool_rs` | robin_set\<uint32_t\> | 20L | Visited tracking (hash) |
| `_inserted_into_pool_bs` | dynamic_bitset* | - | Visited tracking (bit) |
| `_id_scratch` | vector\<uint32_t\> | R*GRAPH_SLACK_FACTOR | Neighbor ID buffer |
| `_dist_scratch` | vector\<float\> | R*GRAPH_SLACK_FACTOR | Distance buffer |
| `_expanded_nodes_set` | robin_set\<uint32_t\> | - | For delete operations |
| `_expanded_nghrs_vec` | vector\<Neighbor\> | - | For delete operations |
| `_occlude_list_output` | vector\<uint32_t\> | - | Pruning output |

Methods: `resize_for_new_L(new_L)`, `clear()`, accessors for all fields.

### `SSDQueryScratch<T> : AbstractScratch<T>`
| Field | Type | Purpose |
|---|---|---|
| `coord_scratch` | T* | Decompressed vector buffer (aligned_dim) |
| `sector_scratch` | char* | Sector read buffer (MAX_N_SECTOR_READS * SECTOR_LEN) |
| `sector_idx` | size_t | Next sector slot |
| `visited` | robin_set\<size_t\> | Visited node tracking |
| `retset` | NeighborPriorityQueue | Best candidates |
| `full_retset` | vector\<Neighbor\> | All results |

### `SSDThreadData<T>`
Wraps `SSDQueryScratch<T> scratch` + `IOContext ctx`.

### `ScratchStoreManager<T>` (RAII)
- Constructor: pops from `ConcurrentQueue<T*>`, blocks until available
- `scratch_space()` → T*
- Destructor: calls `scratch->clear()`, pushes back to pool, notifies waiters
- `destroy()` — drains pool and deletes all scratches (cleanup)

---

## File: `include/timer.h`

```cpp
class Timer {
  reset()                              // reset to now
  long long elapsed()                  // microseconds since last reset
  float elapsed_seconds()              // seconds (float)
  string elapsed_seconds_for_step(s)   // "Time for {s}: X.XX seconds"
};
```

---

## File: `include/utils.h` + `src/utils.cpp`

### Macros
```
ROUND_UP(X, Y)    — round X up to nearest multiple of Y
ROUND_DOWN(X, Y)  — round X down to nearest multiple of Y
DIV_ROUND_UP(X,Y) — ceiling integer division
IS_ALIGNED(X, Y)  — true if X % Y == 0
METADATA_SIZE      = 4096
BUFFER_SIZE_FOR_CACHED_IO = 1GB
```

### Inline Functions (in header)
| Function | Description |
|---|---|
| `file_exists(name, dirCheck)` | stat-based existence check |
| `open_file_to_write(writer, filename)` | Opens ofstream with exception handling |
| `get_file_size(fname)` | Returns file size via seekg(end) |
| `delete_file(fileName)` | remove() with error reporting |
| `get_bin_metadata(file, nrows, ncols)` | Reads first 8 bytes |
| `get_graph_num_frozen_points(graph_file)` | Reads graph header for frozen count |
| `load_bin<T>(file, data, npts, dim)` | Allocates and loads binary file |
| `save_bin<T>(file, data, npts, ndims)` | Writes binary with header |
| `load_aligned_bin<T>(file, data, npts, dim, rounded_dim)` | Loads with row alignment to 8*sizeof(T) |
| `copy_aligned_data_from_file(file, data, npts, dim, rounded_dim)` | Copies into pre-allocated aligned buf |
| `load_truthset(file, ids, dists, npts, dim)` | Loads ground truth (ids + optional distances) |
| `load_range_truthset(file, groundtruth, gt_num)` | Variable-length ground truth |
| `prune_truthset_for_range(file, range, gt, npts)` | Filters truth by distance threshold |
| `save_Tvecs(filename, data, npts, ndims)` | Writes in xvecs format |
| `save_data_in_base_dimensions(file, data, npts, ndims, aligned_dim)` | Strips padding before save |
| `alloc_aligned(ptr, size, align)` | Platform aligned allocation |
| `aligned_free(ptr)` | Platform aligned deallocation |
| `GenRandom(rng, addr, size, N)` | Generate random unique indices in [0, N) |
| `convert_types<In,Out>(src, dst, npts, dim)` | Element-wise type conversion (parallel) |
| `prepare_base_for_inner_products<T>(in, out)` | MIPS→L2 transformation (adds extra dim) |
| `prefetch_vector(vec, vecsize)` | _mm_prefetch L1 (64-byte stride) |
| `prefetch_vector_l2(vec, vecsize)` | _mm_prefetch L2 |
| `get_norm<T>(arr, dim)` | L2 norm |
| `normalize<T>(arr, dim)` | In-place L2 normalization |
| `print_progress(percentage)` | Terminal progress bar |
| `convert_labels_string_to_int(...)` | String labels → integer IDs with mapping file |
| `read_file_to_vector_of_strings(file, unique)` | Read text file line-by-line |
| `copy_file(in, out)` | Binary file copy |
| `validate_index_file_size(stream)` | Check actual vs expected size |
| `clean_up_artifacts(paths, suffixes)` | Bulk delete temporary files |

### Functions in `src/utils.cpp`
| Function | Description |
|---|---|
| `block_convert(writer, reader, buf, npts, ndims)` | Normalize float block and write |
| `normalize_data_file(inFile, outFile)` | Normalize entire binary float file |
| `calculate_recall(num_q, gold, gs_dist, dim_gs, results, dim_or, recall_at)` | Recall@k with tie-breaking |
| `calculate_recall(..., active_tags)` | Recall limited to active tag subset |
| `calculate_range_search_recall(num_q, gt, results)` | Range search recall percentage |

### AVX Detection (`src/utils.cpp`, Windows only)
- `cpuHasAvxSupport()` / `cpuHasAvx2Support()` — CPUID-based detection
- `AvxSupportedCPU`, `Avx2SupportedCPU` — global bools

---

## File: `include/math_utils.h` + `src/math_utils.cpp`

### `math_utils` namespace
| Function | Description |
|---|---|
| `calc_distance(v1, v2, dim)` | L2 squared distance (scalar loop) |
| `compute_vecs_l2sq(out, data, n, dim)` | Parallel L2² norms via cblas_snrm2 |
| `rotate_data_randomly(data, n, dim, rot, out, transpose)` | Matrix rotation via cblas_sgemm |
| `compute_closest_centers_in_block(data, n, dim, centers, nc, dl2, cl2, idx, dm, k)` | k-nearest centers using MKL: D = dl2 + cl2 - 2·data·centersᵀ |
| `compute_closest_centers(data, n, dim, pivots, nc, k, idx, inv_idx, norms)` | Block-wise closest center computation |
| `process_residuals(data, n, dim, pivots, nc, closest, subtract)` | Add/subtract nearest center from each vector |

### `kmeans` namespace
| Function | Description |
|---|---|
| `lloyds_iter(data, n, dim, centers, nc, dl2, docs, cc)` | One Lloyd's iteration: assign + update centers + compute residual |
| `run_lloyds(data, n, dim, centers, nc, max_reps, docs, cc)` | Full k-means with early stopping (Δresidual < 0.001%) |
| `selecting_pivots(data, n, dim, pivots, nc)` | Random pivot selection |
| `kmeanspp_selecting_pivots(data, n, dim, pivots, nc)` | k-means++ init (max 8M points) |

---

## File: `include/logger.h` + `include/logger_impl.h` + `src/logger.cpp`

### Public API
```cpp
namespace diskann {
  // When ENABLE_CUSTOM_LOGGER is defined:
  extern std::basic_ostream<char> cout;  // backed by ANNStreamBuf(stdout)
  extern std::basic_ostream<char> cerr;  // backed by ANNStreamBuf(stderr)
  void SetCustomLogger(std::function<void(LogLevel, const char*)>);

  // When not defined:
  using std::cout;
  using std::cerr;

  enum LogLevel { LL_Info = 0, LL_Error, LL_Count };
}
```

### `ANNStreamBuf` (internal, when ENABLE_CUSTOM_LOGGER)
- Extends `std::basic_streambuf<char>`
- 1024-byte internal buffer
- `overflow(c)` — lock, append char, flush if full
- `sync()` — lock, flush buffer to `g_logger`
- `logImpl(str, len)` — null-terminates and calls `g_logger(_logLevel, str)`

---

## File: `include/ann_exception.h` + `src/ann_exception.cpp`

```cpp
class ANNException : public std::runtime_error {
  ANNException(message, errorCode);
  ANNException(message, errorCode, funcSig, fileName, lineNum);
  // formats: [FUNC: f][FILE: f][LINE: n]  message
};

class FileException : public ANNException {
  FileException(filename, system_error, funcSig, fileName, lineNum);
};
```

## File: `include/exceptions.h`
```cpp
class NotImplementedException : public std::logic_error {
  NotImplementedException();  // "Function not yet implemented."
};
```

---

## File: `include/concurrent_queue.h`

```cpp
template<typename T> class ConcurrentQueue {
  ConcurrentQueue();
  ConcurrentQueue(T nullT);        // value returned on empty pop

  uint64_t size();                  // lock-protected
  bool empty();
  void push(T& val);               // lock + push
  template<class Iter> void insert(begin, end);  // bulk push
  T pop();                          // returns null_T if empty
  void wait_for_push_notify(us=10); // condition_variable wait
  void wait_for_pop_notify(us=10);
  void push_notify_one/all();
  void pop_notify_one/all();
};
```

---

## File: `include/natural_number_map.h` + `src/natural_number_map.cpp`

```cpp
template<typename Key, typename Value> class natural_number_map {
  struct position { size_t _key; size_t _keys_already_enumerated; bool is_valid(); };

  void reserve(count);
  size_t size();
  void set(Key, Value);            // auto-resizes vector+bitset
  void erase(Key);
  bool contains(Key);              // O(1) bitset test
  bool try_get(Key, Value&);
  Value get(position);
  position find_first();           // bitset scan
  position find_next(position);
  void clear();
};
// Instantiated: <uint32_t, int32_t>, <uint32_t, uint32_t>,
//               <uint32_t, int64_t>, <uint32_t, uint64_t>,
//               <uint32_t, tag_uint128>
```

---

## File: `include/natural_number_set.h` + `src/natural_number_set.cpp`

```cpp
template<typename T> class natural_number_set {
  bool is_empty();
  void reserve(count);
  void insert(T id);          // push_back + bitset set
  T pop_any();                // pop_back + bitset clear (LIFO)
  void clear();
  size_t size();
  bool is_in_set(T id);       // O(1) bitset test
};
// Instantiated: <unsigned>
```

---

## File: `include/locking.h`

```cpp
namespace diskann {
  // Linux:
  using non_recursive_mutex = std::mutex;
  using LockGuard = std::lock_guard<std::mutex>;
  // Windows:
  using non_recursive_mutex = windows_exclusive_slim_lock;  // 8 bytes
  using LockGuard = windows_exclusive_slim_lock_guard;
}
```

---

## File: `include/percentile_stats.h`

```cpp
struct QueryStats {
  float total_us, io_us, cpu_us;
  unsigned n_4k, n_8k, n_12k, n_ios, read_size;
  unsigned n_cmps_saved, n_cmps, n_cache_hits, n_hops;
};

template<T> T get_percentile_stats(QueryStats*, len, percentile, member_fn);
template<T> double get_mean_stats(QueryStats*, len, member_fn);
```

---

## File: `include/tag_uint128.h`

```cpp
#pragma pack(push, 1)
struct tag_uint128 {
  uint64_t _data1 = 0, _data2 = 0;
  operator==, operator= for tag_uint128 and uint64_t
};
#pragma pack(pop)

// std::hash<tag_uint128> → Hash128to64 (Murmur-inspired)
```

---

## File: `include/program_options_utils.hpp`

String constants for CLI argument descriptions:
- `DATA_TYPE_DESCRIPTION`, `DISTANCE_FUNCTION_DESCRIPTION`
- `INDEX_PATH_PREFIX_DESCRIPTION`, `QUERY_FILE_DESCRIPTION`
- `SEARCH_LIST_DESCRIPTION`, `INPUT_DATA_PATH`
- `FILTER_LABEL_DESCRIPTION`, `FILTERS_FILE_DESCRIPTION`
- `GROUND_TRUTH_FILE_DESCRIPTION`, `NUMBER_THREADS_DESCRIPTION`
- `MAX_BUILD_DEGREE`, `GRAPH_BUILD_COMPLEXITY`, `GRAPH_BUILD_ALPHA`
- `BUIlD_GRAPH_PQ_BYTES`, `USE_OPQ`, `LABEL_FILE`, `UNIVERSAL_LABEL`
- `BEAMWIDTH`, `NUMBER_OF_NODES_TO_CACHE`, `FAIL_IF_RECALL_BELOW`
- `make_program_description(exe_name, desc)` — formats usage string

---

## File: `include/boost_dynamic_bitset_fwd.h`

Forward declares `boost::dynamic_bitset<Block, Allocator>` to avoid including boost headers in public API.

## File: `include/windows_customizations.h`

```cpp
#ifdef _WINDLL
  #define DISKANN_DLLEXPORT __declspec(dllexport)
#elif _WINDOWS
  #define DISKANN_DLLEXPORT __declspec(dllimport)
#else
  #define DISKANN_DLLEXPORT  // empty
#endif
```

## File: `include/windows_slim_lock.h`

```cpp
class windows_exclusive_slim_lock {  // 8 bytes (SRWLOCK)
  lock(), try_lock(), unlock()
};
class windows_exclusive_slim_lock_guard {  // RAII guard
  windows_exclusive_slim_lock_guard(lock&);
  ~windows_exclusive_slim_lock_guard();
};
```

## File: `include/common_includes.h`

Aggregated standard includes: algorithm, atomic, cassert, chrono, climits, cmath, cstdio, cstring, ctime, fcntl.h, fstream, iostream, iomanip, omp.h, queue, random, set, shared_mutex, sys/stat.h, sstream, unordered_map, vector.

---

## Binary File Format Summary

### Vector Data (`.bin`)
```
Offset 0:  int32_t npts
Offset 4:  int32_t ndims
Offset 8:  T[npts * ndims]  (row-major)
```

### Graph File
```
Offset 0:   uint64_t expected_file_size
Offset 8:   uint32_t max_observed_degree
Offset 12:  uint32_t start_node
Offset 16:  uint64_t num_frozen_points
Offset 24:  per-node adjacency lists (variable length)
```

### xvecs Format (save_Tvecs)
```
For each point:
  uint32_t ndims
  T[ndims]
```

### Ground Truth (truthset)
```
int32_t npts, int32_t dim
uint32_t[npts * dim]  (neighbor ids)
[optional] float[npts * dim]  (distances)
```

### Range Ground Truth
```
int32_t npts, int32_t total_results
uint32_t[npts]  (count per query)
uint32_t[total_results]  (concatenated ids)
```
