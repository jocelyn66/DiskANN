# DiskANN Python Bindings — Detailed Reference

## File Inventory

### C++ Binding Layer

| File | Role |
|------|------|
| `python/CMakeLists.txt` | Build config: pybind11 module `_diskannpy`, links `diskann` lib, NumPy, pybind11 |
| `python/include/common.h` | Type aliases: `filterT`, `StaticIdType`, `DynamicIdType` (all `uint32_t`); `NeighborsAndDistances<IdType>` template |
| `python/include/builder.h` | Declares `build_disk_index<DT>()` and `build_memory_index<DT>()` |
| `python/include/static_memory_index.h` | Declares `StaticMemoryIndex<DT>` class |
| `python/include/static_disk_index.h` | Declares `StaticDiskIndex<DT>` class; typedefs `PlatformSpecificAlignedFileReader` |
| `python/include/dynamic_memory_index.h` | Declares `DynamicMemoryIndex<DT>` class |
| `python/src/module.cpp` | pybind11 module definition: `PYBIND11_MODULE(_diskannpy, m)`, `add_variant<T>()`, defaults submodule |
| `python/src/builder.cpp` | Implements `build_disk_index<DT>()`, `build_memory_index<DT>()`, `prepare_filtered_label_map()` |
| `python/src/static_memory_index.cpp` | Implements `StaticMemoryIndex<DT>` methods |
| `python/src/static_disk_index.cpp` | Implements `StaticDiskIndex<DT>` methods |
| `python/src/dynamic_memory_index.cpp` | Implements `DynamicMemoryIndex<DT>` methods |

### Python Wrapper Layer

| File | Role |
|------|------|
| `python/src/__init__.py` | Package init; type aliases (`DistanceMetric`, `VectorDType`, `VectorLike`, etc.); `QueryResponse`, `QueryResponseBatch` NamedTuples; module docstring |
| `python/src/_builder.py` | `build_disk_index()` and `build_memory_index()` top-level functions |
| `python/src/_builder.pyi` | Type stubs with overloads for builder functions |
| `python/src/_common.py` | Validation utilities, metric conversion, metadata I/O, dtype helpers |
| `python/src/_static_memory_index.py` | `StaticMemoryIndex` class |
| `python/src/_static_disk_index.py` | `StaticDiskIndex` class |
| `python/src/_dynamic_memory_index.py` | `DynamicMemoryIndex` class |
| `python/src/_files.py` | Binary file I/O: `vectors_to_file`, `vectors_from_file`, `tags_to_file`, `tags_from_file`, `vectors_metadata_from_file` |
| `python/src/defaults.py` | Re-exports C++ defaults with Python docstrings |
| `python/src/py.typed` | PEP 561 marker for typed package |

### Apps

| File | Purpose |
|------|---------|
| `python/apps/in-mem-static.py` | CLI app: build + search `StaticMemoryIndex` with recall measurement |
| `python/apps/in-mem-dynamic.py` | CLI app: insert, delete, consolidate, re-insert, search with `DynamicMemoryIndex` |
| `python/apps/insert-in-clustered-order.py` | CLI app: cluster data with k-means, insert cluster-by-cluster into `DynamicMemoryIndex` |
| `python/apps/cluster.py` | CLI: k-means clustering and permutation of vector file |
| `python/apps/utils.py` | Shared utilities: `bin_to_numpy`, `Timer`, `calculate_recall`, `cluster_and_permute` |
| `python/apps/cli/__main__.py` | Fire-based CLI dispatcher for dynamic/static/clustered workflows |

### Tests

| File | Purpose |
|------|---------|
| `python/tests/test_builder.py` | Tests for `build_disk_index` and `build_memory_index` parameter validation |
| `python/tests/test_static_memory_index.py` | Integration: build → load → search/batch_search with recall check vs sklearn |
| `python/tests/test_static_disk_index.py` | Integration: build disk → load → search/batch_search with recall check |
| `python/tests/test_dynamic_memory_index.py` | Integration: `from_file` load, fresh insert, batch_search, recall check |
| `python/tests/test_files.py` | Tests for `vectors_from_file` (in-memory and memmap modes) |
| `python/tests/fixtures/__init__.py` | Re-exports: `build_random_vectors_and_memory_index`, `random_vectors`, `vectors_as_temp_file`, `calculate_recall` |
| `python/tests/fixtures/build_memory_index.py` | Helper: generate random vectors, build memory index, return test tuple |
| `python/tests/fixtures/create_test_data.py` | `random_vectors()`, `write_vectors()`, `vectors_as_temp_file()` context manager |
| `python/tests/fixtures/recall.py` | `calculate_recall()` set-intersection based recall computation |

---

## Type Aliases (python/src/__init__.py)

```python
DistanceMetric = Literal["l2", "mips", "cosine"]
VectorDType = Union[Type[np.float32], Type[np.int8], Type[np.uint8]]
VectorLike = npt.NDArray[VectorDType]           # 1d query vector
VectorLikeBatch = npt.NDArray[VectorDType]      # 2d matrix of vectors
VectorIdentifier = np.uint32                     # single vector ID
VectorIdentifierBatch = npt.NDArray[np.uint32]  # array of vector IDs
```

### QueryResponse (NamedTuple)
```python
class QueryResponse(NamedTuple):
    identifiers: npt.NDArray[VectorIdentifier]  # 1d
    distances: npt.NDArray[np.float32]          # 1d
```

### QueryResponseBatch (NamedTuple)
```python
class QueryResponseBatch(NamedTuple):
    identifiers: npt.NDArray[VectorIdentifier]  # 2d (num_queries × k)
    distances: np.ndarray[np.float32]          # 2d (num_queries × k)
```

### Metadata (NamedTuple, python/src/_files.py)
```python
class Metadata(NamedTuple):
    num_vectors: int
    dimensions: int
```

---

## C++ Binding Types (python/include/common.h)

```cpp
namespace diskannpy {
    typedef uint32_t filterT;
    typedef uint32_t StaticIdType;
    typedef uint32_t DynamicIdType;
    
    template <class IdType>
    using NeighborsAndDistances = std::pair<py::array_t<IdType>, py::array_t<float>>;
}
```

---

## Builder Functions

### `diskannpy.build_disk_index()`

**Python:** `python/src/_builder.py` → **C++:** `python/src/builder.cpp` `build_disk_index<DT>()`

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `str \| VectorLikeBatch` | Path to .bin file or numpy 2d array |
| `distance_metric` | `DistanceMetric` | `"l2"` or `"mips"` (no cosine for disk) |
| `index_directory` | `str` | Existing directory for output files |
| `complexity` | `int` | Build list size (L), typically 75–200 |
| `graph_degree` | `int` | Max graph degree (R), typically 60–150 |
| `search_memory_maximum` | `float` | Max RAM in GB for search |
| `build_memory_maximum` | `float` | Max RAM in GB for build |
| `num_threads` | `int` | 0 = all processors |
| `pq_disk_bytes` | `int` | 0 = uncompressed; >0 = PQ compressed |
| `vector_dtype` | `VectorDType \| None` | Required if data is str |
| `index_prefix` | `str` | Default `"ann"` |

**Restrictions:** `mips` requires `np.float32`. `cosine` not supported for disk index.

**C++ implementation:** Constructs a params string `"R L final_ram indexing_ram threads [pq_bytes]"` and calls `diskann::build_disk_index<DT>()`.

### `diskannpy.build_memory_index()`

**Python:** `python/src/_builder.py` → **C++:** `python/src/builder.cpp` `build_memory_index<DT>()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `str \| VectorLikeBatch` | — | Vectors to index |
| `distance_metric` | `DistanceMetric` | — | |
| `index_directory` | `str` | — | Output directory |
| `complexity` | `int` | — | Build list size |
| `graph_degree` | `int` | — | Max degree |
| `num_threads` | `int` | — | |
| `alpha` | `float` | `1.2` | Pruning parameter ≥1 |
| `use_pq_build` | `bool` | `False` | PQ during build |
| `num_pq_bytes` | `int` | `0` | |
| `use_opq` | `bool` | `False` | Optimized PQ |
| `vector_dtype` | `VectorDType \| None` | `None` | Required if data is str |
| `tags` | `str \| VectorIdentifierBatch` | `""` | Required for DynamicMemoryIndex |
| `filter_labels` | `list[list[str]] \| None` | `None` | Per-vector category labels |
| `universal_label` | `str` | `""` | Label included in all searches |
| `filter_complexity` | `int` | `0` | Must be >0 if filters used |
| `index_prefix` | `str` | `"ann"` | |

**C++ implementation:**
- Builds `IndexWriteParameters` via builder pattern
- Reads bin file metadata for `data_num`, `data_dim`
- Constructs `diskann::Index<T, TagT, LabelT>` with appropriate flags
- Tags: loads from `.tags` file if path provided; calls `index.build(path, num, tags)` 
- Filters: calls `prepare_filtered_label_map()` → `convert_labels_string_to_int()`, then `index.build_filtered_index()`
- Saves with `index.save()`

---

## Search Classes — Detailed API

### StaticMemoryIndex

**Python:** `python/src/_static_memory_index.py`
**C++ binding:** `python/src/static_memory_index.cpp` → `diskann::Index<DT, StaticIdType, filterT>`

#### Constructor

```python
StaticMemoryIndex(
    index_directory: str,
    num_threads: int,
    initial_search_complexity: int,
    index_prefix: str = "ann",
    distance_metric: Optional[DistanceMetric] = None,
    vector_dtype: Optional[VectorDType] = None,
    dimensions: Optional[int] = None,
    enable_filters: bool = False
)
```

When `enable_filters=True`, reads `{prefix}_labels_map.txt` and `{prefix}_label_metadata.json`.

**C++ constructor** (`static_index_builder<DT>`): Creates `diskann::Index<DT>` with `nullptr` write params, `dynamic_index=false`, `enable_tags=false`.

#### `search(query, k_neighbors, complexity, filter_label="")`
- Returns `QueryResponse`
- If `filter_label != ""`: maps string → uint32 via `self._labels_map`, calls `search_with_filter()`
- C++: `_index.search()` or `_index.search_with_filters()` (note plural in core API)

#### `batch_search(queries, k_neighbors, complexity, num_threads)`
- Returns `QueryResponseBatch`
- C++: OMP parallel loop calling `_index.search()` per query row

---

### StaticDiskIndex

**Python:** `python/src/_static_disk_index.py`
**C++ binding:** `python/src/static_disk_index.cpp` → `diskann::PQFlashIndex<DT>`

#### Constructor

```python
StaticDiskIndex(
    index_directory: str,
    num_threads: int,
    num_nodes_to_cache: int,
    cache_mechanism: int = 1,
    distance_metric: Optional[DistanceMetric] = None,
    vector_dtype: Optional[VectorDType] = None,
    dimensions: Optional[int] = None,
    index_prefix: str = "ann"
)
```

**Cache mechanisms:**
- `1`: Load `_sample_data.bin`, generate cache from sample queries (`generate_cache_list_from_sample_queries`)
- `2`: BFS-level-based caching (`cache_bfs_levels`)
- Other: No caching

**C++ constructor:** Creates `PlatformSpecificAlignedFileReader` (Linux or Windows), constructs `PQFlashIndex`, calls `load()`, then caches.

#### `search(query, k_neighbors, complexity, beam_width=2)`
- Returns `QueryResponse`
- C++: `_index.cached_beam_search()` → returns `uint64_t` ids, converted to `uint32_t`

#### `batch_search(queries, k_neighbors, complexity, num_threads, beam_width=2)`
- Returns `QueryResponseBatch`
- C++: OMP parallel `cached_beam_search`, then cast `uint64_t[]` → `uint32_t` 2d array

---

### DynamicMemoryIndex

**Python:** `python/src/_dynamic_memory_index.py`
**C++ binding:** `python/src/dynamic_memory_index.cpp` → `diskann::Index<DT, DynamicIdType, filterT>`

#### Constructor

```python
DynamicMemoryIndex(
    distance_metric: DistanceMetric,
    vector_dtype: VectorDType,
    dimensions: int,
    max_vectors: int,
    complexity: int,
    graph_degree: int,
    saturate_graph: bool = False,
    max_occlusion_size: int = 750,
    alpha: float = 1.2,
    num_threads: int = 0,
    filter_complexity: int = 0,
    num_frozen_points: int = 1,
    initial_search_complexity: int = 0,
    search_threads: int = 0,
    concurrent_consolidation: bool = True
)
```

**C++ constructor:** `dynamic_index_write_parameters()` builds `IndexWriteParameters`; `dynamic_index_builder<DT>()` creates `diskann::Index` with `dynamic_index=true`, `enable_tags=true`.

**Python state tracking:**
- `self._num_vectors` — count of live vectors
- `self._removed_num_vectors` — count of soft-deleted vectors
- `self._points_deleted` — flag for pending deletions

#### `from_file(index_directory, max_vectors, complexity, graph_degree, ...)` (classmethod)
- Verifies `.tags` file exists
- Reads metadata if available
- Constructs empty index, then calls `_index.load(prefix_path)`

#### `insert(vector, vector_id)`
- Auto-consolidates if at capacity with pending deletes
- C++: `_index.insert_point(vector.data(), id)` → status 0 = success

#### `batch_insert(vectors, vector_ids, num_threads=0)`
- C++: OMP parallel `insert_point()` per row; returns `py::array_t<int>` statuses
- Python: checks all statuses, raises RuntimeError with failed IDs if any ≠ 0

#### `mark_deleted(vector_id)`
- Soft delete. C++: `_index.lazy_delete(id)`
- Increments `_removed_num_vectors`, sets `_points_deleted = True`

#### `consolidate_delete()`
- Hard delete. C++: `_index.consolidate_deletes(_write_parameters)`
- Resets counters: `_num_vectors -= _removed_num_vectors`, `_removed_num_vectors = 0`

#### `save(save_path, index_prefix="ann")`
- Auto-consolidates if `_points_deleted` (with warning)
- C++: `_index.save(path, compact_before_save=True)`
- Writes metadata file

#### `search(query, k_neighbors, complexity)` → `QueryResponse`
- C++: `_index.search_with_tags(query, knn, complexity, ids, dists, empty_vector)`

#### `batch_search(queries, k_neighbors, complexity, num_threads)` → `QueryResponseBatch`
- C++: OMP parallel `search_with_tags()` per query

#### `num_points()` → `size_t`
- C++: `_index.get_num_points()`

---

## Validation Utilities (python/src/_common.py)

| Function | Purpose |
|----------|---------|
| `valid_dtype(dtype)` | Validates and canonicalizes to `np.float32`, `np.int8`, or `np.uint8` |
| `_valid_metric(metric: str)` | Maps `"l2"` → `_native_dap.L2`, `"mips"` → `INNER_PRODUCT`, `"cosine"` → `COSINE` |
| `_assert(bool, msg)` | Raises `ValueError` if false |
| `_assert_dtype(dtype)` | Checks `np.can_cast(dtype, valid)` for any valid dtype |
| `_castable_dtype_or_raise(data, expected)` | Returns `data.astype(expected, casting="safe")` or raises `TypeError` |
| `_assert_2d(arr, name)` | Ensures array is 2-dimensional |
| `_assert_is_positive_uint32(val, name)` | `0 < val < 2^32` |
| `_assert_is_nonnegative_uint32(val, name)` | `-1 < val < 2^32` |
| `_assert_existing_directory(path, name)` | Path exists and is directory |
| `_assert_existing_file(path, name)` | Path exists and is file |
| `_write_index_metadata(prefix, dtype, metric, n, d)` | Writes 4 × uint64 to `{prefix}_metadata.bin` |
| `_read_index_metadata(prefix)` | Reads metadata, returns `(dtype, metric_str, num_points, dims)` or None |
| `_ensure_index_metadata(prefix, ...)` | Reads metadata or validates user-provided values |
| `_valid_index_prefix(dir, prefix)` | Validates directory exists, returns `os.path.join(dir, prefix)` |

### Internal Enums

```python
class _DataType(Enum):
    FLOAT32 = 0; INT8 = 1; UINT8 = 2
    
class _Metric(Enum):
    L2 = 0; MIPS = 1; COSINE = 2
```

---

## File I/O (python/src/_files.py)

### Binary Format
DiskANN vector binary files:
```
[num_points: int32][dimensions: int32][vector_data: dtype × num_points × dimensions]
```

Tag files use same format with `dimensions = 1`.

### Functions

| Function | Signature | Notes |
|----------|-----------|-------|
| `vectors_metadata_from_file` | `(path: str) → Metadata` | Reads first 8 bytes |
| `vectors_to_file` | `(path: str, vectors: 2d NDArray)` | Validates dtype and 2d shape |
| `vectors_from_file` | `(path: str, dtype, use_memmap=False, mode="r")` | Returns ndarray or memmap |
| `tags_to_file` | `(path: str, tags: 1d NDArray[uint32])` | Handles 2d→1d reshape with warning |
| `tags_from_file` | `(path: str) → NDArray[uint32]` | Returns 1d uint32 array |

---

## Defaults (python/src/defaults.py → C++ include/defaults.h)

| Python Name | C++ Source | Value | Description |
|-------------|-----------|-------|-------------|
| `ALPHA` | `defaults::ALPHA` | `1.2f` | Pruning parameter |
| `NUM_THREADS` | `defaults::NUM_THREADS` | `0` | 0 = all processors |
| `MAX_OCCLUSION_SIZE` | `defaults::MAX_OCCLUSION_SIZE` | `750` | Max occlusion neighbors |
| `FILTER_COMPLEXITY` | `defaults::FILTER_LIST_SIZE` | `0` | Filter search list size |
| `NUM_FROZEN_POINTS_STATIC` | `defaults::NUM_FROZEN_POINTS_STATIC` | `0` | |
| `NUM_FROZEN_POINTS_DYNAMIC` | `defaults::NUM_FROZEN_POINTS_DYNAMIC` | `1` | |
| `SATURATE_GRAPH` | `defaults::SATURATE_GRAPH` | `False` | |
| `GRAPH_DEGREE` | `defaults::MAX_DEGREE` | `64` | Max graph degree (R) |
| `COMPLEXITY` | `defaults::BUILD_LIST_SIZE` | `100` | Build/search list size (L) |
| `PQ_DISK_BYTES` | hardcoded | `0` | 0 = uncompressed on SSD |
| `USE_PQ_BUILD` | hardcoded | `False` | |
| `NUM_PQ_BYTES` | hardcoded | `0` | |
| `USE_OPQ` | hardcoded | `False` | |

---

## pybind11 Module Registration (python/src/module.cpp)

The `Variant` struct maps type-specific names:
```cpp
struct Variant {
    std::string disk_builder_name;    // e.g. "build_disk_float_index"
    std::string memory_builder_name;  // e.g. "build_memory_float_index"
    std::string dynamic_memory_index_name;  // e.g. "DynamicMemoryFloatIndex"
    std::string static_memory_index_name;   // e.g. "StaticMemoryFloatIndex"
    std::string static_disk_index_name;     // e.g. "StaticDiskFloatIndex"
};
```

Three variants registered: `FloatVariant`, `UInt8Variant`, `Int8Variant`.

The `add_variant<T>(m, variant)` template function registers:
1. Two module-level functions (`build_disk_*_index`, `build_memory_*_index`)
2. Three `py::class_<>` definitions with constructors and methods

Opaque vector types declared to prevent automatic conversion:
```cpp
PYBIND11_MAKE_OPAQUE(std::vector<uint32_t>);
PYBIND11_MAKE_OPAQUE(std::vector<float>);
PYBIND11_MAKE_OPAQUE(std::vector<int8_t>);
PYBIND11_MAKE_OPAQUE(std::vector<uint8_t>);
```

The `defaults` submodule re-exports C++ default values to Python.

---

## Distance Metric / Dtype Compatibility Matrix

### Memory Index (StaticMemoryIndex, DynamicMemoryIndex)

| Metric \ Dtype | np.float32 | np.uint8 | np.int8 |
|----------------|------------|----------|---------|
| L2             | ✅         | ✅       | ✅      |
| MIPS           | ✅         | ❌       | ❌      |
| Cosine         | ✅         | ✅       | ✅      |

### Disk Index (StaticDiskIndex)

| Metric \ Dtype | np.float32 | np.uint8 | np.int8 |
|----------------|------------|----------|---------|
| L2             | ✅         | ✅       | ✅      |
| MIPS           | ✅         | ❌       | ❌      |
| Cosine         | ❌         | ❌       | ❌      |

---

## OMP Parallelization Pattern

All batch operations use the same OMP pattern:
```cpp
omp_set_num_threads(num_threads != 0 ? num_threads : omp_get_num_procs());
#pragma omp parallel for schedule(dynamic, 1) default(none) shared(...)
for (int64_t i = 0; i < (int64_t)num_queries; i++) {
    // per-query work
}
```

Dynamic scheduling with chunk size 1 is used because individual query/insert times vary.
