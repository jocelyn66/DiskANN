---
name: DiskANN-dr-python
description: Use when working with the Python bindings of DiskANN — pybind11 C++ wrappers and Python API for building/searching memory and disk indices
---

# DiskANN Python Bindings (`diskannpy`)

## 1. Module Purpose

The `diskannpy` module provides a Python API for DiskANN's approximate nearest neighbor (ANN) index capabilities: building indices, searching, and dynamic insert/delete operations. It bridges Python and the C++ core via **pybind11**, exposing three index types and two builder functions.

The native C++ extension is compiled as `_diskannpy` (a single shared library) via `python/CMakeLists.txt`. A pure-Python wrapper layer (`python/src/`) adds validation, type coercion, metadata management, and ergonomic API on top.

**Key entry points:**
- `diskannpy.build_memory_index()` — build a Vamana in-memory index from vectors (file or numpy array) → calls `_diskannpy.build_memory_{float,uint8,int8}_index`
- `diskannpy.build_disk_index()` — build a PQ-compressed disk index → calls `_diskannpy.build_disk_{float,uint8,int8}_index`
- `diskannpy.StaticMemoryIndex` — load & search an immutable in-memory index → wraps `_diskannpy.StaticMemory{Float,UInt8,Int8}Index`
- `diskannpy.StaticDiskIndex` — load & search an immutable SSD-backed index → wraps `_diskannpy.StaticDisk{Float,UInt8,Int8}Index`
- `diskannpy.DynamicMemoryIndex` — mutable in-memory index with insert/delete/search → wraps `_diskannpy.DynamicMemory{Float,UInt8,Int8}Index`

**Supported dtypes:** `np.float32`, `np.int8`, `np.uint8`
**Supported metrics:** `"l2"`, `"mips"` (float32 only), `"cosine"` (memory indices only; not disk)

---

## 2. Core Design Logic

### 2.1 Dual-Layer Architecture

```
Python user code
    ↓
Pure-Python wrappers (python/src/_builder.py, _static_memory_index.py, etc.)
  - parameter validation (_common.py assertions)
  - dtype routing (select float/int8/uint8 variant)
  - numpy ↔ file conversions (_files.py)
  - metadata read/write (_common.py _write_index_metadata / _read_index_metadata)
    ↓
pybind11 C++ bindings (python/src/module.cpp → builder.cpp, static_memory_index.cpp, etc.)
  - thin wrappers accepting py::array_t<T> numpy arrays
  - delegate to core diskann::Index<T>, diskann::PQFlashIndex<T>
    ↓
DiskANN C++ core (src/index.cpp, src/disk_utils.cpp, etc.)
```

**Why dual-layer?**
- The C++ binding layer (`python/src/*.cpp`, `python/include/*.h`) is intentionally minimal: it converts numpy arrays to raw pointers, constructs `diskann::Index` objects, and returns `py::array_t` results. No validation or user-facing logic lives here.
- The Python layer (`python/src/*.py`) handles all validation, dtype coercion (`_castable_dtype_or_raise`), metric string-to-enum mapping (`_valid_metric`), metadata file I/O, and API documentation. This separation keeps the C++ bindings simple and the Python API ergonomic.

### 2.2 Type Variant Strategy

Each C++ class is templated on `<DT>` (data type). The `module.cpp` `add_variant<T>()` function template registers three variants per class/function, using a `Variant` struct to name them:

- `FloatVariant` → `StaticMemoryFloatIndex`, `build_memory_float_index`, etc.
- `UInt8Variant` → `StaticMemoryUInt8Index`, etc.
- `Int8Variant` → `StaticMemoryInt8Index`, etc.

The Python wrappers select the right native class based on `vector_dtype`:
```python
# from _static_memory_index.py __init__
if vector_dtype == np.uint8:
    _index = _native_dap.StaticMemoryUInt8Index
elif vector_dtype == np.int8:
    _index = _native_dap.StaticMemoryInt8Index
else:
    _index = _native_dap.StaticMemoryFloatIndex
```

### 2.3 Type Mapping (numpy ↔ C++)

| Python (numpy)        | C++ pybind11 parameter                                       | C++ core type |
|-----------------------|--------------------------------------------------------------|---------------|
| `np.float32` array    | `py::array_t<float, py::array::c_style \| py::array::forcecast>` | `float`       |
| `np.uint8` array      | `py::array_t<uint8_t, ...>`                                  | `uint8_t`     |
| `np.int8` array       | `py::array_t<int8_t, ...>`                                   | `int8_t`      |
| `np.uint32` (ids)     | `py::array_t<uint32_t, ...>`                                 | `uint32_t`    |

Return types use `NeighborsAndDistances<IdType>` = `std::pair<py::array_t<IdType>, py::array_t<float>>`, defined in `python/include/common.h`.

The `py::array::forcecast` flag means pybind11 will attempt to cast input arrays to the required type, but the Python layer pre-validates with `_castable_dtype_or_raise()` in `python/src/_common.py` to give clearer error messages.

### 2.4 Memory Management

- **numpy arrays own their memory.** The C++ binding layer accesses numpy array data via `.data()` / `.mutable_data()` pointers during the call. Returned arrays (`py::array_t<>`) are allocated on the C++ side and their ownership is transferred to Python/numpy.
- **Index objects** are stored as member variables in the Python wrapper classes (e.g., `self._index`). The C++ `diskann::Index` / `diskann::PQFlashIndex` lifetime is tied to the pybind11 wrapper object, which is tied to the Python wrapper's `self._index` reference.
- **AlignedFileReader** for disk indices: `StaticDiskIndex` creates a `std::shared_ptr<AlignedFileReader>` (`_reader`) that outlives the `PQFlashIndex` it's passed to (`python/src/static_disk_index.cpp` constructor).

### 2.5 Metadata File Convention

Builder functions write a `{prefix}_metadata.bin` file (4 × uint64 values: dtype enum, metric enum, num_points, dimensions) via `_write_index_metadata()` in `python/src/_common.py`. Search classes read it via `_ensure_index_metadata()` to auto-detect dtype, metric, etc., making constructor parameters optional when metadata exists.

---

## 3. Core Data Structures

### 3.1 Python Layer Classes

| Class | File | Purpose | Wraps |
|-------|------|---------|-------|
| `StaticMemoryIndex` | `python/src/_static_memory_index.py` | Immutable in-memory index for search | `_diskannpy.StaticMemory{Float,UInt8,Int8}Index` |
| `StaticDiskIndex` | `python/src/_static_disk_index.py` | Immutable SSD-backed index for search | `_diskannpy.StaticDisk{Float,UInt8,Int8}Index` |
| `DynamicMemoryIndex` | `python/src/_dynamic_memory_index.py` | Mutable in-memory index (insert/delete/search) | `_diskannpy.DynamicMemory{Float,UInt8,Int8}Index` |
| `QueryResponse` | `python/src/__init__.py` | NamedTuple: `(identifiers: NDArray[uint32], distances: NDArray[float32])` | — |
| `QueryResponseBatch` | `python/src/__init__.py` | NamedTuple: same but 2d arrays | — |
| `Metadata` | `python/src/_files.py` | NamedTuple: `(num_vectors, dimensions)` from bin file header | — |

### 3.2 C++ Binding Classes (in `diskannpy` namespace)

| Class | Header | .cpp | Core dependency |
|-------|--------|------|-----------------|
| `StaticMemoryIndex<DT>` | `python/include/static_memory_index.h` | `python/src/static_memory_index.cpp` | `diskann::Index<DT, StaticIdType, filterT>` |
| `StaticDiskIndex<DT>` | `python/include/static_disk_index.h` | `python/src/static_disk_index.cpp` | `diskann::PQFlashIndex<DT>` |
| `DynamicMemoryIndex<DT>` | `python/include/dynamic_memory_index.h` | `python/src/dynamic_memory_index.cpp` | `diskann::Index<DT, DynamicIdType, filterT>` |
| `build_disk_index<DT>()` | `python/include/builder.h` | `python/src/builder.cpp` | `diskann::build_disk_index<DT>()` |
| `build_memory_index<DT>()` | `python/include/builder.h` | `python/src/builder.cpp` | `diskann::Index<DT>::build()` / `build_filtered_index()` |

### 3.3 Key Type Aliases (`python/include/common.h`)

```cpp
typedef uint32_t filterT;
typedef uint32_t StaticIdType;
typedef uint32_t DynamicIdType;
template <class IdType>
using NeighborsAndDistances = std::pair<py::array_t<IdType>, py::array_t<float>>;
```

### 3.4 Utility Functions

| Function | File | Purpose |
|----------|------|---------|
| `vectors_to_file()` | `python/src/_files.py` | Write 2d numpy array → DiskANN binary vector file |
| `vectors_from_file()` | `python/src/_files.py` | Read DiskANN binary vector file → numpy array (or memmap) |
| `vectors_metadata_from_file()` | `python/src/_files.py` | Read (num_points, dimensions) header from binary file |
| `tags_to_file()` | `python/src/_files.py` | Write uint32 tag array → DiskANN binary tag file |
| `tags_from_file()` | `python/src/_files.py` | Read DiskANN binary tag file → uint32 numpy array |
| `valid_dtype()` | `python/src/_common.py` | Validate and canonicalize vector dtype |

### 3.5 File Layout

DiskANN binary vector files have an 8-byte header: `[num_points: int32, dimensions: int32]` followed by raw vector data. Tag files use the same format with `dimensions=1`.

Index files on disk (produced by builders) include:
- Memory index: `{prefix}` (graph), `{prefix}.data`, optionally `{prefix}.tags`, `{prefix}_metadata.bin`
- Disk index: `{prefix}_disk.index`, `{prefix}_pq_compressed.bin`, `{prefix}_pq_pivots.bin`, `{prefix}_sample_data.bin`, `{prefix}_sample_ids.bin`, `{prefix}_mem.index.data`, `{prefix}_metadata.bin`

---

## 4. State Flow

### 4.1 Build Memory Index Flow

```
User: diskannpy.build_memory_index(data=vectors, distance_metric="l2", ...)
  │
  ├─ _builder.py: validate params, coerce dtype
  ├─ If data is np.ndarray: vectors_to_file() → temp .bin file
  ├─ If tags provided: tags_to_file() → {prefix}.tags
  ├─ If filter_labels provided: write {prefix}_pylabels.txt + {prefix}_label_metadata.json
  ├─ Select builder: _native_dap.build_memory_{dtype}_index
  │
  └─ builder.cpp: build_memory_index<T>()
       ├─ Construct IndexWriteParameters via builder pattern
       ├─ diskann::get_bin_metadata() → (data_num, data_dim)
       ├─ Construct diskann::Index<T, TagT, LabelT>
       ├─ If tags + filters: index.build_filtered_index()
       │  Elif tags: index.build() with tags
       │  Elif filters: index.build_filtered_index() 
       │  Else: index.build()
       └─ index.save(index_output_path)
  │
  └─ _builder.py: _write_index_metadata() → {prefix}_metadata.bin
```

### 4.2 Build Disk Index Flow

```
User: diskannpy.build_disk_index(data=..., ...)
  │
  ├─ _builder.py: validate (cosine not supported for disk), validate mips float32 only
  ├─ Select builder: _native_dap.build_disk_{dtype}_index
  │
  └─ builder.cpp: build_disk_index<T>()
       ├─ Construct params string: "graph_degree complexity final_ram indexing_ram threads [pq_bytes]"
       └─ diskann::build_disk_index<DT>(data_path, index_prefix, params, metric)
  │
  └─ _builder.py: _write_index_metadata()
```

### 4.3 Static Memory Search Flow

```
User: index = diskannpy.StaticMemoryIndex(index_directory=..., ...)
  │
  ├─ _static_memory_index.py:
  │    ├─ If enable_filters=True: loads `{prefix}_labels_map.txt` and `{prefix}_label_metadata.json`
  │    │    └─ **Known bug:** the `except:` clause (bare, with `# noqa: E722`) raises `RuntimeException(...)` which is NOT a valid Python built-in — should be `RuntimeError`
  │    └─ _ensure_index_metadata() → (dtype, metric, num_points, dims)
  ├─ Select native class: StaticMemory{Float,UInt8,Int8}Index
  └─ Construct → static_memory_index.cpp constructor
       ├─ static_index_builder<DT>() → diskann::Index<DT> with search params only
       └─ _index.load(prefix, num_threads, initial_search_complexity)

User: response = index.search(query, k_neighbors=10, complexity=64)
  │
  ├─ _static_memory_index.py: validate query shape/dtype, coerce, warn if k > complexity
  └─ self._index.search(query, knn, complexity)
       └─ static_memory_index.cpp: allocate py::array_t<StaticIdType>, py::array_t<float>
            └─ _index.search(query.data(), knn, complexity, ids, dists)
  │
  └─ Return QueryResponse(identifiers=neighbors, distances=distances)
```

### 4.4 Static Disk Search Flow

```
User: index = diskannpy.StaticDiskIndex(index_directory=..., num_nodes_to_cache=1000)
  │
  └─ static_disk_index.cpp constructor:
       ├─ Create AlignedFileReader (platform-specific)
       ├─ PQFlashIndex::load()
       ├─ cache_mechanism==1: cache_sample_paths() from _sample_data.bin
       └─ cache_mechanism==2: cache_bfs_levels()

User: response = index.search(query, k_neighbors=10, complexity=64, beam_width=2)
  │
  └─ static_disk_index.cpp: _index.cached_beam_search(query, knn, complexity, u64_ids, dists, beam_width)
       └─ Convert uint64 ids → uint32 StaticIdType
  │
  └─ Return QueryResponse(identifiers, distances)
```

### 4.5 Dynamic Memory Index Flow

```
User: index = diskannpy.DynamicMemoryIndex(distance_metric="l2", vector_dtype=np.float32,
                                            dimensions=128, max_vectors=100000, complexity=64, graph_degree=32)
  │
  └─ dynamic_memory_index.cpp constructor:
       ├─ Member initializer list: `_initial_search_complexity(initial_search_complexity != 0 ? initial_search_complexity : complexity)` — when 0, falls back to `complexity`
       ├─ dynamic_index_write_parameters() → IndexWriteParameters
       └─ dynamic_index_builder<DT>(m, _write_parameters, dims, max_vectors, _initial_search_complexity, initial_search_threads, ...)
            └─ `_initial_search_threads = initial_search_threads != 0 ? initial_search_threads : omp_get_num_procs()` — when 0, falls back to all available processors
            └─ Constructs diskann::Index<DT> with dynamic_index=true, enable_tags=true

User: index.insert(vector, vector_id=42)
  │
  ├─ _dynamic_memory_index.py: validate, check capacity (auto consolidate_delete if needed)
  └─ _index.insert_point(vector.data(), id) → status code 0 = success

User: index.batch_insert(vectors, vector_ids, num_threads=8)
  │
  └─ dynamic_memory_index.cpp: parallel OMP loop calling _index.insert_point per vector

User: index.mark_deleted(vector_id=42)     # soft delete (lazy_delete)
User: index.consolidate_delete()            # hard delete (consolidate_deletes with write_params)

User: index.save(save_path="/tmp/myindex")
  │
  ├─ Auto-consolidate if points_deleted (with warning)
  └─ _index.save(path, compact_before_save=True) + _write_index_metadata()

User: index = DynamicMemoryIndex.from_file(index_directory=..., max_vectors=...)
  │
  ├─ Verify .tags file exists
  ├─ _ensure_index_metadata() for dtype/metric
  ├─ Construct empty DynamicMemoryIndex
  └─ index._index.load(index_prefix_path)
```

---

## 5. Modification Scenarios

### Scenario 1: Add a New Distance Metric

**Goal:** Support a new metric (e.g., "hamming") in the Python bindings.

**Files to modify:**
1. `python/src/_common.py` — Add mapping in `_valid_metric()`: `elif metric.lower() == "hamming": return _native_dap.HAMMING`
2. `python/src/_common.py` — Add `HAMMING` to `_Metric` enum and `to_native()`/`from_native()`/`to_str()` methods
3. `python/src/__init__.py` — Update `DistanceMetric` type alias: `Literal["l2", "mips", "cosine", "hamming"]`
4. `python/src/module.cpp` — The C++ `_native_dap.HAMMING` enum value must be exported. The metric enum comes from `include/distance.h` (`diskann::Metric`), so ensure it's defined there first.
5. `python/src/_builder.py` — Update dtype/metric restriction tables in docstrings and validation logic (e.g., `_assert(dap_metric != _native_dap.HAMMING or ...)`)

### Scenario 2: Add a New Vector Data Type (e.g., float16)

**Goal:** Support `np.float16` vectors.

**Files to modify:**
1. `python/include/common.h` — `NeighborsAndDistances` is generic; no change needed there. But you need a new template instantiation.
2. `python/src/module.cpp` — Add a new `Variant` (`Float16Variant`) and call `add_variant<half>(m, Float16Variant)`. Also add `PYBIND11_MAKE_OPAQUE(std::vector<half>)` if needed.
3. `python/src/builder.cpp` — Add explicit template instantiations: `template void build_memory_index<half>(...)` and `template void build_disk_index<half>(...)`.
4. `python/src/static_memory_index.cpp`, `dynamic_memory_index.cpp`, `static_disk_index.cpp` — Add `template class StaticMemoryIndex<half>;` etc.
5. `python/src/_common.py` — Add `np.float16` to `_VALID_DTYPES`, update `valid_dtype()`, update `_DataType` enum.
6. `python/src/__init__.py` — Update `VectorDType` type alias.
7. **All Python wrapper classes** — Add `elif vector_dtype == np.float16: _index = _native_dap.StaticMemoryFloat16Index` routing.

### Scenario 3: Add a New Search Method (e.g., Range Search for DynamicMemoryIndex)

**Goal:** Expose range search (find all neighbors within distance threshold) on the dynamic index.

**Files to modify:**
1. `python/include/dynamic_memory_index.h` — Declare `NeighborsAndDistances<DynamicIdType> range_search(query, threshold, complexity)`.
2. `python/src/dynamic_memory_index.cpp` — Implement `range_search()` calling `_index.search_with_tags()` or a range-specific core API, returning variable-length results.
3. `python/src/module.cpp` — In `add_variant<T>`, add `.def("range_search", &diskannpy::DynamicMemoryIndex<T>::range_search, ...)` to the `DynamicMemoryIndex` class binding.
4. `python/src/_dynamic_memory_index.py` — Add `def range_search(self, query, threshold, complexity)` method with validation, dtype coercion, and return type.

### Scenario 4: Add Filter Support to DynamicMemoryIndex

**Goal:** Enable label-based filtered search on DynamicMemoryIndex (currently only `StaticMemoryIndex` has `search_with_filter`).

**Files to modify:**
1. `python/include/dynamic_memory_index.h` — Add `search_with_filter()` method declaration.
2. `python/src/dynamic_memory_index.cpp` — Implement by calling `_index.search_with_tags()` with filter parameter (the core `diskann::Index` supports filtered search when built with labels).
3. `python/src/module.cpp` — Expose `search_with_filter` in the `DynamicMemoryIndex` pybind11 class binding.
4. `python/src/_dynamic_memory_index.py` — Add Python `search_with_filter()` method with label map loading (similar to `_static_memory_index.py`'s `enable_filters` pattern).

### Scenario 5: Change Return Type to Include Query Statistics

**Goal:** Return per-query statistics (e.g., number of hops, comparisons) alongside search results.

**Files to modify:**
1. `python/include/common.h` — Define a new return type, e.g., `NeighborsDistancesAndStats`.
2. `python/src/static_disk_index.cpp` — The disk search already has `diskann::QueryStats stats` internally (line visible in `search()` method). Expose these stats in the return value.
3. `python/src/module.cpp` — Update the pybind11 `.def("search", ...)` to use the new return type.
4. `python/src/__init__.py` — Add a `QueryResponseWithStats` NamedTuple.
5. `python/src/_static_disk_index.py` — Update `search()` to return the new type.
