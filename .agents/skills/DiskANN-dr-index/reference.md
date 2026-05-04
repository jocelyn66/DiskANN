# DiskANN Index Module — Reference

## Complete Data Structures

### `Index<T, TagT, LabelT>` (include/index.h)

The main templated class. `T` is the vector data type (float, int8_t, uint8_t), `TagT` is the user-facing point identifier type (uint32_t, int32_t, uint64_t, int64_t, tag_uint128), `LabelT` is the filter label type (uint16_t, uint32_t).

#### Template instantiations (src/index.cpp, lines ~3380+):
- `T`: float, int8_t, uint8_t
- `TagT`: int32_t, uint32_t, int64_t, uint64_t, tag_uint128
- `LabelT`: uint16_t, uint32_t

#### Private member fields:

| Field | Type | Purpose |
|-------|------|---------|
| `_dist_metric` | `Metric` | Distance metric enum (L2, COSINE, INNER_PRODUCT) |
| `_data_store` | `shared_ptr<AbstractDataStore<T>>` | Vector data storage |
| `_graph_store` | `unique_ptr<AbstractGraphStore>` | Graph adjacency list storage |
| `_opt_graph` | `char*` | Interleaved data+graph for FastL2 optimized search |
| `_dim` | `size_t` | Original vector dimensionality |
| `_nd` | `size_t` | Number of active (non-deleted) data points |
| `_max_points` | `size_t` | Maximum capacity for data points |
| `_num_frozen_pts` | `size_t` | Number of frozen entry points |
| `_frozen_pts_used` | `size_t` | Number of frozen point slots actually used |
| `_node_size` | `size_t` | Size of interleaved node in optimized layout |
| `_data_len` | `size_t` | Data portion size in optimized layout |
| `_neighbor_len` | `size_t` | Neighbor portion size in optimized layout |
| `_start` | `uint32_t` | Entry point location for search |
| `_has_built` | `bool` | Whether index has been built or loaded |
| `_saturate_graph` | `bool` | Whether to fill unused degree slots after pruning |
| `_dynamic_index` | `bool` | Whether dynamic operations (insert/delete) are enabled |
| `_enable_tags` | `bool` | Whether tag-based point identification is active |
| `_normalize_vecs` | `bool` | True when using normalized L2 for cosine metric |
| `_deletes_enabled` | `bool` | Whether delete operations have been enabled |
| `_filtered_index` | `bool` | Whether label-based filtering is active |
| `_location_to_labels` | `vector<vector<LabelT>>` | Labels assigned to each location |
| `_labels` | `tsl::robin_set<LabelT>` | Set of all distinct labels |
| `_labels_file` | `string` | Path to labels file |
| `_label_to_start_id` | `unordered_map<LabelT, uint32_t>` | Per-label medoid/entry point |
| `_medoid_counts` | `unordered_map<uint32_t, uint32_t>` | How many labels use each point as medoid |
| `_use_universal_label` | `bool` | Whether a universal (match-all) label exists |
| `_universal_label` | `LabelT` | The universal label value |
| `_filterIndexingQueueSize` | `uint32_t` | L value for filtered search during indexing |
| `_label_map` | `unordered_map<string, LabelT>` | String-to-integer label mapping |
| `_indexingQueueSize` | `uint32_t` | L (search list size) during index build |
| `_indexingRange` | `uint32_t` | R (maximum degree) during index build |
| `_indexingMaxC` | `uint32_t` | C (max candidate pool for pruning, default 750) |
| `_indexingAlpha` | `float` | Alpha for distance-ratio pruning |
| `_indexingThreads` | `uint32_t` | Number of threads for build |
| `_query_scratch` | `ConcurrentQueue<InMemQueryScratch<T>*>` | Pool of reusable scratch spaces |
| `_pq_dist` | `bool` | Whether PQ distances are used during build |
| `_use_opq` | `bool` | Whether optimized PQ (with rotation) is used |
| `_num_pq_chunks` | `size_t` | Number of PQ chunks |
| `_pq_distance_fn` | `shared_ptr<QuantizedDistance<T>>` | PQ distance function |
| `_pq_data_store` | `shared_ptr<AbstractDataStore<T>>` | PQ-compressed data store (or alias to _data_store) |
| `_pq_generated` | `bool` | Whether PQ codebook has been generated |
| `_pq_table` | `FixedChunkPQTable` | PQ codebook table |
| `_tag_to_location` | `tsl::sparse_map<TagT, uint32_t>` | Tag → internal location mapping |
| `_location_to_tag` | `natural_number_map<uint32_t, TagT>` | Internal location → tag mapping |
| `_empty_slots` | `natural_number_set<uint32_t>` | Available slot positions |
| `_delete_set` | `unique_ptr<tsl::robin_set<uint32_t>>` | Locations pending deletion |
| `_data_compacted` | `bool` | Whether data has been compacted (no gaps) |
| `_is_saved` | `bool` | Whether index has been saved to disk |
| `_conc_consolidate` | `bool` | Whether concurrent consolidation is enabled |
| `_update_lock` | `shared_timed_mutex` | Protects save/load vs. operations |
| `_consolidate_lock` | `shared_timed_mutex` | Protects consolidate/compact |
| `_tag_lock` | `shared_timed_mutex` | Protects tag maps and metadata: `_tag_to_location`, `_location_to_tag`, `_empty_slots`, `_nd`, `_max_points`, `_label_to_start_id` |
| `_delete_lock` | `shared_timed_mutex` | Protects delete set and compaction flag |
| `_locks` | `vector<non_recursive_mutex>` | Per-node locks for adjacency lists |
| `INDEX_GROWTH_FACTOR` | `static const float` | 1.5f — resize multiplier |

---

### `AbstractIndex` (include/abstract_index.h)

Type-erased base class that wraps `Index<T, TagT, LabelT>` method calls through `std::any`. All public template methods (e.g., `build<data_type, tag_type>()`) cast arguments to `std::any` and delegate to pure virtual `_build()`, `_search()`, etc.

#### Public template methods (non-virtual, defined in abstract_index.h):
- `build<data_type, tag_type>(const data_type*, size_t, const vector<tag_type>&)`
- `search<data_type, IDType>(const data_type*, size_t K, uint32_t L, IDType*, float*) → pair<uint32_t, uint32_t>`
- `search_with_tags<data_type, tag_type>(const data_type*, uint64_t K, uint32_t L, tag_type*, float*, vector<data_type*>&, bool, string) → size_t`
- `search_with_filters<IndexType>(const DataType&, const string&, size_t K, uint32_t L, IndexType*, float*) → pair<uint32_t, uint32_t>`
- `insert_point<data_type, tag_type>(const data_type*, tag_type) → int`
- `insert_point<data_type, tag_type, label_type>(const data_type*, tag_type, const vector<label_type>&) → int`
- `lazy_delete<tag_type>(const tag_type&) → int`
- `lazy_delete<tag_type>(const vector<tag_type>&, vector<tag_type>&)`
- `get_active_tags<tag_type>(robin_set<tag_type>&)`
- `set_start_points_at_random<data_type>(data_type radius, uint32_t seed)`
- `get_vector_by_tag<tag_type, data_type>(tag_type&, data_type*) → int`
- `set_universal_label<label_type>(label_type)`

#### Pure virtual methods:
- `build(const string&, size_t, IndexFilterParams&)`
- `save(const char*, bool)`
- `load(const char*, uint32_t, uint32_t)`
- `consolidate_deletes(const IndexWriteParameters&) → consolidation_report`
- `optimize_index_layout()`
- Private: `_build`, `_search`, `_search_with_filters`, `_insert_point` (2 overloads), `_lazy_delete` (2 overloads), `_get_active_tags`, `_set_start_points_at_random`, `_get_vector_by_tag`, `_search_with_tags`, `_search_with_optimized_layout`, `_set_universal_label`

---

### `IndexConfig` (include/index_config.h)

Configuration struct for index creation. Has a private constructor — must be built via `IndexConfigBuilder`.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `data_strategy` | `DataStoreStrategy` | — | MEMORY |
| `graph_strategy` | `GraphStoreStrategy` | — | MEMORY |
| `metric` | `Metric` | — | L2, COSINE, INNER_PRODUCT |
| `dimension` | `size_t` | — | Vector dimensionality |
| `max_points` | `size_t` | — | Maximum number of data points |
| `dynamic_index` | `bool` | false | Enable dynamic operations |
| `enable_tags` | `bool` | false | Enable tag-based identification |
| `pq_dist_build` | `bool` | false | Use PQ distances during build |
| `concurrent_consolidate` | `bool` | false | Allow concurrent consolidation |
| `use_opq` | `bool` | false | Use optimized PQ |
| `filtered_index` | `bool` | false | Enable label filtering |
| `num_pq_chunks` | `size_t` | 0 | Number of PQ chunks |
| `num_frozen_pts` | `size_t` | 0 (static), 1 (dynamic) | Number of frozen entry points |
| `label_type` | `string` | "uint32" | Label type name |
| `tag_type` | `string` | "uint32" | Tag type name |
| `data_type` | `string` | — | Data type name (required) |
| `index_write_params` | `shared_ptr<IndexWriteParameters>` | nullptr | Build parameters |
| `index_search_params` | `shared_ptr<IndexSearchParams>` | nullptr | Search parameters |

**Validation in `IndexConfigBuilder::build()`**: Forces `num_frozen_pts >= 1` for dynamic index. Requires `initial_search_list_size > 0` for dynamic index when search params provided.

---

### `IndexWriteParameters` (include/parameters.h)

Immutable build-time parameters. Constructed via `IndexWriteParametersBuilder`.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `search_list_size` | `uint32_t` | (required) | L — beam width during build |
| `max_degree` | `uint32_t` | (required) | R — max graph degree |
| `saturate_graph` | `bool` | false | Fill unused edge slots after pruning |
| `max_occlusion_size` | `uint32_t` | 750 | C — max candidate pool for occlusion |
| `alpha` | `float` | 1.2 | Distance ratio threshold for occlusion |
| `num_threads` | `uint32_t` | omp_get_num_procs() | Build parallelism |
| `filter_list_size` | `uint32_t` | search_list_size | Lf — beam width for filtered build |

### `IndexSearchParams` (include/parameters.h)

| Field | Type | Purpose |
|-------|------|---------|
| `initial_search_list_size` | `uint32_t` | Default L for search |
| `num_search_threads` | `uint32_t` | Number of search threads (determines scratch pool size) |

### `IndexFilterParams` (include/index_build_params.h)

| Field | Type | Purpose |
|-------|------|---------|
| `save_path_prefix` | `string` | Output prefix for label files |
| `label_file` | `string` | Path to raw label file |
| `tags_file` | `string` | Path to tags file |
| `universal_label` | `string` | String representation of universal label |
| `filter_threshold` | `uint32_t` | Filter threshold value |

### `consolidation_report` (include/abstract_index.h)

| Field | Type | Purpose |
|-------|------|---------|
| `_status` | `status_code` | SUCCESS=0, FAIL=1, LOCK_FAIL=2, INCONSISTENT_COUNT_ERROR=3 |
| `_active_points` | `size_t` | Live points after consolidation |
| `_max_points` | `size_t` | Capacity |
| `_empty_slots` | `size_t` | Available slot count |
| `_slots_released` | `size_t` | Slots freed by this consolidation |
| `_delete_set_size` | `size_t` | Points still pending deletion |
| `_num_calls_to_process_delete` | `size_t` | Number of graph repair operations |
| `_time` | `double` | Duration in seconds |

---

### `IndexFactory` (include/index_factory.h)

Factory for constructing `AbstractIndex` instances from `IndexConfig`.

#### Methods:

| Method | Signature | Purpose |
|--------|-----------|---------|
| `create_instance()` | `unique_ptr<AbstractIndex>` | Create index from stored config |
| `construct_graphstore()` | `static unique_ptr<AbstractGraphStore>(GraphStoreStrategy, size_t, size_t)` | Create graph store |
| `construct_datastore<T>()` | `static shared_ptr<AbstractDataStore<T>>(DataStoreStrategy, size_t, size_t, Metric)` | Create data store with appropriate distance function |
| `construct_pq_datastore<T>()` | `static shared_ptr<PQDataStore<T>>(DataStoreStrategy, size_t, size_t, Metric, size_t, bool)` | Create PQ-compressed data store |
| `construct_inmem_distance_fn<T>()` | `static Distance<T>*(Metric)` | Create distance function; special-cases COSINE+float → AVXNormalizedCosineDistanceFloat |

**Type dispatch chain** in `create_instance()`:
1. `create_instance(string data_type, string tag_type, string label_type)` — dispatches on data_type
2. `create_instance<data_type>(string tag_type, string label_type)` — dispatches on tag_type
3. `create_instance<data_type, tag_type>(string label_type)` — dispatches on label_type
4. `create_instance<data_type, tag_type, label_type>()` — constructs stores and returns `make_unique<Index<...>>()`

---

## Complete Function Signatures — Index<T, TagT, LabelT>

### Constructors & Destructor (src/index.cpp lines 30-180)

```cpp
// Config-based constructor (primary)
Index(const IndexConfig &index_config,
      shared_ptr<AbstractDataStore<T>> data_store,
      unique_ptr<AbstractGraphStore> graph_store,
      shared_ptr<AbstractDataStore<T>> pq_data_store = nullptr);

// Legacy constructor (delegates to config-based)
Index(Metric m, size_t dim, size_t max_points,
      shared_ptr<IndexWriteParameters> index_parameters,
      shared_ptr<IndexSearchParams> index_search_params,
      size_t num_frozen_pts = 0, bool dynamic_index = false,
      bool enable_tags = false, bool concurrent_consolidate = false,
      bool pq_dist_build = false, size_t num_pq_chunks = 0,
      bool use_opq = false, bool filtered_index = false);

// Destructor:
// Acquires all four mutexes as unique_lock in order: _update_lock, _consolidate_lock, _tag_lock, _delete_lock
// Acquires all per-node _locks[i] via LockGuard in a loop
// delete[] _opt_graph if non-null
// _query_scratch destroyed via ScratchStoreManager::destroy()
~Index();
```

### Build Methods (src/index.cpp lines 1570-1900)

```cpp
void build(const char *filename, size_t num_points_to_load,
           const vector<TagT> &tags = {});
void build(const char *filename, size_t num_points_to_load,
           const char *tag_filename);
void build(const T *data, size_t num_points_to_load,
           const vector<TagT> &tags);
void build(const string &data_file, size_t num_points_to_load,
           IndexFilterParams &filter_params);
void build_filtered_index(const char *filename, const string &label_file,
                          size_t num_points_to_load,
                          const vector<TagT> &tags = {});
```

### Search Methods (src/index.cpp lines 1960-2280)

```cpp
template <typename IDType>
pair<uint32_t, uint32_t> search(const T *query, size_t K, uint32_t L,
                                 IDType *indices, float *distances = nullptr);
// Returns: (hops, comparisons)
// Frozen points excluded by: best_L_nodes[i].id < _max_points
// Loop collects results until pos == K, then breaks
// If pos < K after loop: prints warning to cerr
// INNER_PRODUCT distances are negated (-1 * distance) before returning

template <typename IndexType>
pair<uint32_t, uint32_t> search_with_filters(const T *query,
                                              const LabelT &filter_label,
                                              size_t K, uint32_t L,
                                              IndexType *indices,
                                              float *distances);

size_t search_with_tags(const T *query, uint64_t K, uint32_t L,
                        TagT *tags, float *distances,
                        vector<T*> &res_vectors,
                        bool use_filters = false,
                        const string filter_label = "");
// Returns: number of results found

void search_with_optimized_layout(const T *query, size_t K, size_t L,
                                   uint32_t *indices);
```

### Dynamic Operations (src/index.cpp lines 2830-3120)

```cpp
int insert_point(const T *point, const TagT tag);
int insert_point(const T *point, const TagT tag,
                 const vector<LabelT> &labels);
// Returns: 0 on success, -1 on failure

int enable_delete();
// Returns: 0 on success, -2 if tags not enabled

int lazy_delete(const TagT &tag);
// Returns: 0 on success, -1 if tag not found

void lazy_delete(const vector<TagT> &tags, vector<TagT> &failed_tags);

consolidation_report consolidate_deletes(const IndexWriteParameters &params);
```

### Save/Load (src/index.cpp lines 270-700)

```cpp
void save(const char *filename, bool compact_before_save = false);
void load(const char *index_file, uint32_t num_threads, uint32_t search_l);
static size_t get_graph_num_frozen_points(const string &graph_file);
```

### Utility Methods

```cpp
size_t get_num_points();
size_t get_max_points();
void set_universal_label(const LabelT &label);
LabelT get_converted_label(const string &raw_label);
void set_start_points(const T *data, size_t data_count);
void set_start_points_at_random(T radius, uint32_t random_seed = 0);
void optimize_index_layout();
// Throws if _dynamic_index is true
// _data_len = (aligned_dim + 1) * sizeof(float)   — +1 for precomputed norm
// _neighbor_len = (max_observed_degree + 1) * sizeof(uint32_t) — +1 for count
// _node_size = _data_len + _neighbor_len
// Per-node layout: [norm(4B)][vector data][neighbor_count(4B)][neighbor_ids...]
// Casts distance function to DistanceFastL2<T>* to compute norms
// Clears _graph_store (resize to 0) after building _opt_graph
void get_active_tags(tsl::robin_set<TagT> &active_tags);
int get_vector_by_tag(TagT &tag, T *vec);
void print_status();
void count_nodes_at_bfs_levels();
bool is_index_saved();
void reposition_frozen_point_to_end();
void prune_all_neighbors(uint32_t max_degree, uint32_t max_occlusion,
                         float alpha);
```

### Core Internal Methods (protected, src/index.cpp)

```cpp
// Main graph construction loop (lines 1280-1380)
// Both passes use: #pragma omp parallel for schedule(dynamic, 2048)
void link();

// Greedy beam search (lines 810-990)
pair<uint32_t, uint32_t> iterate_to_fixed_point(
    InMemQueryScratch<T> *scratch, uint32_t Lindex,
    const vector<uint32_t> &init_ids, bool use_filter,
    const vector<LabelT> &filters, bool search_invocation);

// Search for a point's neighbors and prune them (lines 1000-1065)
// Unfiltered path: single iterate_to_fixed_point() call
// Filtered path:
//   1. First iterate_to_fixed_point() with filter_specific_start_nodes and filteredLindex
//   2. Results collected into std::set<Neighbor> for dedup/merging
//   3. scratch->clear() between the two searches
//   4. Second iterate_to_fixed_point() unfiltered with init_ids and Lindex
//   5. Merge unfiltered results into the set
//   6. Copy merged set back to scratch->pool()
// After both paths: self-loop (node referencing itself) removed before pruning
void search_for_point_and_prune(int location, uint32_t Lindex,
    vector<uint32_t> &pruned_list, InMemQueryScratch<T> *scratch,
    bool use_filter = false, uint32_t filteredLindex = 0);

// Distance-ratio based candidate pruning (lines 1070-1170)
// Algorithm: cur_alpha starts at 1.0, multiplied by 1.2f each round.
// For L2/COSINE: occlude_factor[t] = max(occlude_factor[t], iter2->distance / djk)
//   (djk == 0 yields float::max)
// For INNER_PRODUCT: x = -iter2->distance, y = -djk;
//   if (y > cur_alpha * x) then occlude_factor[t] = max(occlude_factor[t], eps)
//   where eps = cur_alpha + 0.01f
void occlude_list(uint32_t location, vector<Neighbor> &pool,
    float alpha, uint32_t degree, uint32_t maxc,
    vector<uint32_t> &result, InMemQueryScratch<T> *scratch,
    const tsl::robin_set<uint32_t> *delete_set_ptr = nullptr);

// Prune neighbor candidates (lines 1175-1215)
void prune_neighbors(uint32_t location, vector<Neighbor> &pool,
    vector<uint32_t> &pruned_list, InMemQueryScratch<T> *scratch);
void prune_neighbors(uint32_t location, vector<Neighbor> &pool,
    uint32_t range, uint32_t max_candidate_size, float alpha,
    vector<uint32_t> &pruned_list, InMemQueryScratch<T> *scratch);

// Add reverse edges from neighbors back to the inserted node (lines 1220-1285)
// Pruning threshold: GRAPH_SLACK_FACTOR * range (not just range)
// Under threshold: simply append via add_neighbour, no pruning
// Over threshold: copy_of_neighbors = existing neighbors + n
//   Build dummy_pool of Neighbor objects with actual distances from des to each candidate
//   Reserve size: ceil(1.05 * GRAPH_SLACK_FACTOR * range)
//   Then prune via prune_neighbors()
void inter_insert(uint32_t n, vector<uint32_t> &pruned_list,
    uint32_t range, InMemQueryScratch<T> *scratch);
void inter_insert(uint32_t n, vector<uint32_t> &pruned_list,
    InMemQueryScratch<T> *scratch);

// Graph repair after deletion (lines 2320-2390)
// consolidate_deletes throws exception if _start is in delete_set
// Main loop uses: #pragma omp parallel for schedule(dynamic, 8192)
void process_delete(const tsl::robin_set<uint32_t> &old_delete_set,
    size_t loc, uint32_t range, uint32_t maxc, float alpha,
    InMemQueryScratch<T> *scratch);

// Data compaction after consolidation (lines 2540-2650)
// Algorithm:
// - new_location[] computed by sequential new_counter++ for occupied slots (in _location_to_tag)
// - Frozen points keep identity mapping: new_location[old] = old
// - Adjacency lists rewritten with remapped neighbor IDs; dangling refs counted and logged to cerr
// - _tag_to_location rebuilt from _location_to_tag using new_location mapping
// - _location_to_tag rebuilt from _tag_to_location
// - _empty_slots rebuilt to cover [_nd .. _max_points)
void compact_data();
void compact_frozen_point();

// Slot management (lines 2650-2720)
int reserve_location();
size_t release_location(int location);
size_t release_locations(const tsl::robin_set<uint32_t> &locations);

// Resize capacity (lines 2830-2870)
void resize(size_t new_max_points);
```

---

## Memory Layout

### Internal Location Numbering

```
[0 ... _nd-1]                     Active data points
[_nd ... _max_points-1]           Empty slots (available for insertion)
[_max_points ... _max_points+_num_frozen_pts-1]   Frozen points (entry points)
```

`_start` is set to `_max_points` when frozen points exist (i.e., points to the first frozen slot). For static indexes without frozen points, `_start` is the medoid.

### File Layout (saved files from prefix `P`)

| File | Format | Content |
|------|--------|---------|
| `P` | Binary graph: [file_size(8B), max_degree(4B), start(4B), num_frozen(8B), then per-node: [degree(4B), neighbor_ids(4B each)]] | Graph adjacency lists |
| `P.data` | Standard .bin format: [num_points(4B), dim(4B), then vectors] | Vector data |
| `P.tags` | Binary: [num_points(4B), 1(4B), then TagT values] | Point tags |
| `P.del` | Binary: [num_deleted(4B), 1(4B), then uint32_t locations] | Pending deletions |
| `P_labels.txt` | Text: one line per point, comma-separated integer labels | Label assignments |
| `P_labels_to_medoids.txt` | Text: `label, medoid_id` per line | Per-label entry points |
| `P_labels_map.txt` | Text: `string_label\tinteger_label` per line | Label string→int mapping |
| `P_universal_label.txt` | Text: single integer | Universal label value |
| `P_raw_labels.txt` | Text: compacted raw label strings | Written on compact_before_save with dynamic filtered index |

---

## Constants & Defaults

| Constant | Value | Location | Purpose |
|----------|-------|----------|---------|
| `OVERHEAD_FACTOR` | 1.1 | index.h | RAM estimate multiplier |
| `EXPAND_IF_FULL` | 0 | index.h | Whether to auto-resize on insert (disabled). When enabled: releases _delete_lock, _tag_lock, _update_lock (shared); acquires _update_lock exclusive + _tag_lock + _delete_lock; calls resize(max_points * INDEX_GROWTH_FACTOR); releases all; re-acquires shared _update_lock + _tag_lock + _delete_lock; retries reserve_location; throws if retry still fails |
| `DEFAULT_MAXC` | 750 | index.h | Default max occlusion candidate pool |
| `MAX_POINTS_FOR_USING_BITSET` | 10000000 | index.cpp | Threshold for bitset vs robin_set for visited tracking; the `fast_iterate` flag is set to `true` when `total_num_points <= MAX_POINTS_FOR_USING_BITSET` |
| `INDEX_GROWTH_FACTOR` | 1.5f | index.cpp | Resize multiplier |
| `GRAPH_SLACK_FACTOR` | ~1.3 | defaults.h | Over-allocation factor for adjacency lists |
| `METADATA_ROWS` | 5 | index.h | Number of metadata entries in graph file |

---

## Lock Acquisition Order

When acquiring multiple locks, always follow this order to prevent deadlocks:

1. `_update_lock`
2. `_consolidate_lock`
3. `_tag_lock`
4. `_delete_lock`
5. `_locks[i]` (per-node locks)

### Lock usage by operation:

| Operation | `_update_lock` | `_consolidate_lock` | `_tag_lock` | `_delete_lock` | `_locks[i]` |
|-----------|:-:|:-:|:-:|:-:|:-:|
| `search()` | shared | — | — | — | shared (brief) |
| `insert_point()` | shared | — | unique | unique→release | unique per node |
| `lazy_delete()` | shared | — | unique | unique | — |
| `consolidate_deletes()` | unique (or shared if conc) | unique | unique | shared | unique per node |
| `save()` | unique | unique | unique | unique | — |
| `load()` | unique | unique | unique | unique | — |
| `compact_data()` | (caller holds) | — | — | — | — |

---

## Enums

### `DataStoreStrategy` (include/index_config.h)
- `MEMORY` — In-memory vector storage

### `GraphStoreStrategy` (include/index_config.h)
- `MEMORY` — In-memory adjacency list storage

### `Metric` (defined in distance module, used throughout)
- `L2` — Euclidean distance
- `COSINE` — Cosine similarity (internally normalized to L2)
- `INNER_PRODUCT` — Maximum inner product search (MIPS)
