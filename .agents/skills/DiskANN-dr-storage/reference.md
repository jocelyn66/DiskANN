# DiskANN Storage Module — Reference

## File Index

| File | Role |
|------|------|
| `include/abstract_data_store.h` | Abstract base class template for all vector data stores |
| `src/abstract_data_store.cpp` | Non-virtual method implementations + explicit template instantiations |
| `include/in_mem_data_store.h` | In-memory full-precision data store declaration |
| `src/in_mem_data_store.cpp` | In-memory data store implementation (~400 lines) |
| `include/pq_data_store.h` | PQ-compressed data store declaration |
| `src/pq_data_store.cpp` | PQ data store implementation (~250 lines, many stubs) |
| `include/abstract_graph_store.h` | Abstract base class for graph adjacency stores (non-templated) |
| `include/in_mem_graph_store.h` | In-memory graph store declaration |
| `src/in_mem_graph_store.cpp` | In-memory graph store implementation (~250 lines) |

---

## Core Types

### `location_t`
- **Definition**: `typedef uint32_t location_t` in `include/types.h`
- **Purpose**: Index into the data/graph stores. Represents a point ID. All store APIs use this type.

### `Metric` (enum)
- **Definition**: `include/distance.h`
- **Values**: `L2 = 0`, `INNER_PRODUCT = 1`, `COSINE = 2`, `FAST_L2 = 3`

---

## Class: `AbstractDataStore<data_t>`

**File**: `include/abstract_data_store.h`
**Template parameter**: `data_t` — element type (`float`, `int8_t`, `uint8_t`)
**Explicit instantiations** (in `src/abstract_data_store.cpp`): `float`, `int8_t`, `uint8_t`

### Protected Fields

| Field | Type | Description |
|-------|------|-------------|
| `_capacity` | `location_t` | Maximum number of vectors the store can hold |
| `_dim` | `size_t` | Logical dimensionality of each vector |

### Constructor

```cpp
AbstractDataStore(const location_t capacity, const size_t dim);
```
Initializes `_capacity` and `_dim`.

### Non-Virtual Methods (implemented in `src/abstract_data_store.cpp`)

```cpp
location_t capacity() const;          // returns _capacity
size_t get_dims() const;              // returns _dim
location_t resize(const location_t new_num_points);
    // Delegates to expand() if new > current, shrink() if new < current
```

### Pure Virtual Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `load` | `location_t load(const string& filename)` | Load vectors from file; returns count loaded |
| `save` | `size_t save(const string& filename, location_t num_pts)` | Save `num_pts` vectors to file; returns bytes written |
| `get_aligned_dim` | `size_t get_aligned_dim() const` | Padded dimension (for SIMD alignment) |
| `populate_data` | `void populate_data(const data_t* vectors, location_t num_pts)` | Bulk load from pointer |
| `populate_data` | `void populate_data(const string& filename, size_t offset)` | Bulk load from file |
| `extract_data_to_bin` | `void extract_data_to_bin(const string& filename, location_t num_pts)` | Export to binary file |
| `get_vector` | `void get_vector(location_t i, data_t* dest) const` | Copy vector `i` to `dest` |
| `set_vector` | `void set_vector(location_t i, const data_t* vector)` | Write vector at location `i` |
| `prefetch_vector` | `void prefetch_vector(location_t loc)` | CPU cache prefetch for vector at `loc` |
| `move_vectors` | `void move_vectors(location_t old_start, location_t new_start, location_t count)` | Move range, zero old |
| `copy_vectors` | `void copy_vectors(location_t from, location_t to, location_t count)` | Copy range (no zeroing) |
| `preprocess_query` | `void preprocess_query(const data_t* query, AbstractScratch<data_t>* scratch) const` | Prepare query for distance computation |
| `get_distance` | `float get_distance(const data_t* query, location_t loc) const` | Distance: query vs stored vector |
| `get_distance` | `void get_distance(const data_t* query, const location_t* locs, uint32_t count, float* dists, AbstractScratch<data_t>* scratch) const` | Batch distance: query vs array of locations |
| `get_distance` | `void get_distance(const data_t* query, const vector<location_t>& ids, vector<float>& dists, AbstractScratch<data_t>* scratch) const` | Batch distance: query vs vector of IDs |
| `get_distance` | `float get_distance(location_t loc1, location_t loc2) const` | Distance between two stored vectors |
| `calculate_medoid` | `location_t calculate_medoid() const` | Find point closest to dataset centroid |
| `get_dist_fn` | `Distance<data_t>* get_dist_fn() const` | Return underlying distance function |
| `get_alignment_factor` | `size_t get_alignment_factor() const` | SIMD alignment requirement |
| `expand` | `location_t expand(location_t new_num_points)` | **(protected)** Grow capacity |
| `shrink` | `location_t shrink(location_t new_num_points)` | **(protected)** Reduce capacity |

---

## Class: `InMemDataStore<data_t>`

**File**: `include/in_mem_data_store.h` (declaration), `src/in_mem_data_store.cpp` (implementation)
**Inherits**: `AbstractDataStore<data_t>`
**Explicit instantiations**: `float`, `int8_t`, `uint8_t`

### Private Fields

| Field | Type | Description |
|-------|------|-------------|
| `_data` | `data_t*` | Aligned flat array; layout: `[capacity * aligned_dim]` elements |
| `_aligned_dim` | `size_t` | `ROUND_UP(dim, distance_fn->get_required_alignment())` |
| `_distance_fn` | `unique_ptr<Distance<data_t>>` | Owned distance function |
| `_pre_computed_norms` | `shared_ptr<float[]>` | Optional precomputed norms (currently unused) |

### Constructor

```cpp
InMemDataStore(location_t capacity, size_t dim, unique_ptr<Distance<data_t>> distance_fn);
```
- Computes `_aligned_dim = ROUND_UP(dim, distance_fn->get_required_alignment())`
- Allocates `capacity * _aligned_dim * sizeof(data_t)` bytes with `alloc_aligned(..., 8 * sizeof(data_t))` alignment
- Zero-initializes the buffer

### Destructor
Calls `aligned_free(_data)`.

### Key Implementation Details

**`populate_data(const data_t* vectors, location_t num_pts)`**:
- Zeros the target area, then `memmove`s each vector from `dim`-stride source to `aligned_dim`-stride destination
- If `_distance_fn->preprocessing_required()`, calls `preprocess_base_points()` (e.g., L2-normalizes for cosine)

**`set_vector(location_t loc, const data_t* vector)`**:
- Zeros the aligned slot, copies `dim` elements, then preprocesses if required
- Used during streaming inserts

**`preprocess_query(const data_t* query, AbstractScratch<data_t>* scratch)`**:
- Simply `memcpy`s the query into `scratch->aligned_query_T()`
- Throws if scratch is null

**`get_distance(const data_t* query, location_t loc)`**:
- `return _distance_fn->compare(query, _data + _aligned_dim * loc, _aligned_dim)`

**`expand(location_t new_size)`** (non-Windows):
- `alloc_aligned` new buffer → `memcpy` old → `aligned_free` old → update `_data` and `_capacity`

**`shrink(location_t new_size)`** (non-Windows):
- Same pattern but copies only `new_size * _aligned_dim` elements

**`calculate_medoid()`**:
- Computes centroid as mean of all vectors (cast to `float`)
- Finds the point with minimum L2 distance to the centroid
- Note: currently single-threaded (OpenMP pragma is commented out)

**`move_vectors(old_start, new_start, count)`**:
- Handles overlapping ranges correctly via `memmove` (through `copy_vectors`)
- Zeros the non-overlapping portion of the old range

**`save(filename, num_pts)`**:
- Delegates to `save_data_in_base_dimensions()` which strips alignment padding, writing only `dim` elements per vector

**`load_impl(filename)`**:
- Reads binary metadata (num_points, dim), validates dimension match
- Resizes if file has more points than current capacity
- Uses `copy_aligned_data_from_file()` to read with alignment

---

## Class: `PQDataStore<data_t>`

**File**: `include/pq_data_store.h` (declaration), `src/pq_data_store.cpp` (implementation)
**Inherits**: `AbstractDataStore<data_t>`
**Explicit instantiations**: `float`, `int8_t`, `uint8_t`

### Private Fields

| Field | Type | Description |
|-------|------|-------------|
| `_quantized_data` | `uint8_t*` | Aligned buffer of PQ codes; layout: `[capacity * num_chunks]` bytes |
| `_num_chunks` | `size_t` | Number of PQ sub-quantizers (must be ≤ dim) |
| `_use_opq` | `bool` | OPQ flag (default `false`, reserved for future) |
| `_distance_metric` | `Metric` | Cached metric enum |
| `_distance_fn` | `unique_ptr<Distance<data_t>>` | Full-precision distance (returned by `get_dist_fn()`) |
| `_pq_distance_fn` | `unique_ptr<QuantizedDistance<data_t>>` | PQ distance function used for actual search |

### Constructor

```cpp
PQDataStore(size_t dim, location_t num_points, size_t num_pq_chunks,
            unique_ptr<Distance<data_t>> distance_fn,
            unique_ptr<QuantizedDistance<data_t>> pq_distance_fn);
```
- Validates `num_pq_chunks <= dim`
- Takes ownership of both distance functions
- Does NOT allocate `_quantized_data` (deferred to `load()` or `populate_data()`)

### Implemented Methods

**`populate_data(const string& filename, size_t offset)`**:
- Full PQ pipeline: reads raw vectors → trains PQ codebook via `generate_quantized_data()` → loads compressed vectors and pivot tables
- Allocates `_quantized_data` with alignment 1
- Calls `_pq_distance_fn->load_pivot_data()` to initialize the distance lookup tables

**`preprocess_query(const data_t* query, AbstractScratch<data_t>* scratch)`**:
- Checks `scratch` for null (throws `ANNException("Scratch space is null")`)  
- Extracts `PQScratch<data_t>*` from the scratch object; checks for null (throws `ANNException("PQScratch space has not been set in the scratch object.")`)  
- Calls `_pq_distance_fn->preprocess_query()` which computes chunk-distance lookup tables

**`get_distance(query, locations, count, distances, scratch)` (batch, array)**:
- Calls `aggregate_coords()` to gather PQ codes for the batch of locations
- Calls `_pq_distance_fn->preprocessed_distance()` to compute all distances using lookup tables

**`get_distance(query, ids, distances, scratch)` (batch, vector)**:
- Same pattern with `vector<location_t>` input

**`load_impl(file_prefix)`**:
- Loads `_quantized_data` from `pq_distance_fn->get_quantized_vectors_filename(prefix)`
- Loads pivot data from `pq_distance_fn->get_pivot_data_filename(prefix)`

**`save(filename, num_pts)`**:
- Saves `_quantized_data` via `diskann::save_bin()`

**`get_aligned_dim()`**: Returns `get_dims()` (no alignment needed).
**`get_alignment_factor()`**: Returns `1`.
**`get_dist_fn()`**: Returns `_distance_fn.get()` (full-precision, NOT PQ).
**`calculate_medoid()`**: Returns random point (not truly computed).

### Unimplemented Methods (throw `std::logic_error`)

- `populate_data(const data_t*, location_t)` — from raw pointer
- `extract_data_to_bin()`
- `get_vector()` — has an inverted condition bug: `if (i < capacity())` throws `logic_error("Not implemented yet")` (valid indices), else throws `ANNException` (out-of-range). Valid calls always get "not implemented"
- `set_vector()` — would need PQ compression
- `move_vectors()`
- Single-point `get_distance()` overloads (both query-vs-loc and loc-vs-loc)
- `expand()`, `shrink()`

---

## Class: `AbstractGraphStore`

**File**: `include/abstract_graph_store.h`
**Not templated** — graph structure is type-independent.

### Private Fields

| Field | Type | Description |
|-------|------|-------------|
| `_capacity` | `size_t` | Total graph node count |
| `_reserve_graph_degree` | `size_t` | Pre-reserved neighbor list capacity per node |

### Constructor

```cpp
AbstractGraphStore(size_t total_pts, size_t reserve_graph_degree);
```
Inline implementation in header.

### Pure Virtual Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `load` | `tuple<uint32_t, uint32_t, size_t> load(const string& path, size_t num_points)` | Returns `(nodes_read, start_node, num_frozen_points)` |
| `store` | `int store(const string& path, size_t num_points, size_t num_fz_points, uint32_t start)` | Returns file size |
| `get_neighbours` | `const vector<location_t>& get_neighbours(location_t i) const` | Zero-copy neighbor access |
| `add_neighbour` | `void add_neighbour(location_t i, location_t id)` | Append a neighbor |
| `clear_neighbours` | `void clear_neighbours(location_t i)` | Empty neighbor list |
| `swap_neighbours` | `void swap_neighbours(location_t a, location_t b)` | Swap two nodes' neighbor lists |
| `set_neighbours` | `void set_neighbours(location_t i, vector<location_t>& neighbours)` | Replace neighbor list |
| `resize_graph` | `size_t resize_graph(size_t new_size)` | Resize graph node count |
| `clear_graph` | `void clear_graph()` | Remove all data |
| `get_max_observed_degree` | `uint32_t get_max_observed_degree()` | Largest neighbor list seen |
| `get_max_range_of_graph` | `size_t get_max_range_of_graph()` | Max degree seen during load |

### Non-Virtual Methods

```cpp
size_t get_total_points();              // returns _capacity
void set_total_points(size_t new_cap);  // protected, updates _capacity
size_t get_reserve_graph_degree();      // protected, returns _reserve_graph_degree
```

---

## Class: `InMemGraphStore`

**File**: `include/in_mem_graph_store.h` (declaration), `src/in_mem_graph_store.cpp` (implementation)
**Inherits**: `AbstractGraphStore`

### Private Fields

| Field | Type | Description |
|-------|------|-------------|
| `_max_range_of_graph` | `size_t` | Maximum neighbor list length observed during load |
| `_max_observed_degree` | `uint32_t` | Maximum neighbor list length observed during build/load |
| `_graph` | `vector<vector<uint32_t>>` | Adjacency lists; `_graph[i]` is the neighbor list for node `i` |

### Constructor

```cpp
InMemGraphStore(size_t total_pts, size_t reserve_graph_degree);
```
- Calls `resize_graph(total_pts)` to initialize `_graph`
- Reserves `reserve_graph_degree` capacity in each inner vector

### Key Implementation Details

**`load_impl(const string& filename, size_t expected_num_points)`**:
- Reads 24-byte Vamana header: `expected_file_size`, `_max_observed_degree`, `start`, `file_frozen_pts`
- Resizes `_graph` if `expected_num_points > get_total_points()`
- Reads each node's adjacency list: `[uint32_t k][k * uint32_t neighbor_ids]`
- Tracks `_max_range_of_graph` (max `k` seen)
- Returns `tuple(nodes_read, start, file_frozen_pts)`

**`save_graph(path, num_points, num_frozen_points, start)`**:
- Initializes `index_size = 24` (the header size), writes the 24-byte header, then each node's `[uint32_t size][data...]`
- Accumulates `index_size` during the node loop, then seeks back to rewrite the final `index_size` and recomputed `max_degree` in the header
- Returns total file size as `int`

**`add_neighbour(i, id)`**:
- `_graph[i].emplace_back(id)` then checks `_graph[i].size()` (the stored list size after appending) to update `_max_observed_degree`

**`set_neighbours(i, neighbours)`**:
- `_graph[i].assign(neighbours.begin(), neighbours.end())` then checks `neighbours.size()` (the input parameter size, not `_graph[i].size()`) to update `_max_observed_degree`

**`resize_graph(new_size)`**:
- `_graph.resize(new_size)` + `set_total_points(new_size)`
- Note: new entries get empty vectors (no reservation — unlike constructor)

**`get_neighbours(i)`**:
- Returns `_graph.at(i)` — uses `at()` for bounds checking (throws `std::out_of_range` on invalid index)

---

## Dependency Graph

```
AbstractDataStore<data_t>
 ├── InMemDataStore<data_t>   uses  Distance<data_t>
 └── PQDataStore<data_t>      uses  Distance<data_t> + QuantizedDistance<data_t>

AbstractGraphStore
 └── InMemGraphStore

AbstractScratch<data_t>  ← passed into preprocess_query() and batch get_distance()
 ├── _aligned_query_T    ← used by InMemDataStore::preprocess_query()
 └── _pq_scratch         ← used by PQDataStore::preprocess_query() and get_distance()
```

---

## Binary File Formats

### Vector Data File (`.bin`)
```
[4 bytes: uint32_t num_points] [4 bytes: uint32_t dim] [num_points * dim * sizeof(data_t) bytes: row-major vectors]
```
Alignment padding is stripped on save and re-added on load.

### PQ Compressed Data File
```
[4 bytes: uint32_t num_points] [4 bytes: uint32_t num_chunks] [num_points * num_chunks bytes: uint8_t PQ codes]
```

### Graph File (Vamana format)
```
Header (24 bytes):
  [8 bytes: size_t file_size] [4 bytes: uint32_t max_degree] [4 bytes: uint32_t start_node] [8 bytes: size_t num_frozen]
Body (variable):
  For each node: [4 bytes: uint32_t k] [k * 4 bytes: uint32_t neighbor_ids]
```

---

## Template Instantiations

All templated classes are explicitly instantiated for three types:
- `float` — standard floating-point vectors
- `int8_t` — signed byte vectors
- `uint8_t` — unsigned byte vectors

Instantiation lines appear at the bottom of each `.cpp` file:
```cpp
template DISKANN_DLLEXPORT class ClassName<float>;
template DISKANN_DLLEXPORT class ClassName<int8_t>;
template DISKANN_DLLEXPORT class ClassName<uint8_t>;
```
