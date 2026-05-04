---
name: DiskANN-dr-index
description: Use when working with the index module of DiskANN — core in-memory Vamana graph index with build, search, insert, delete, consolidate, and filtered search
---

# DiskANN Index Module — Core In-Memory Vamana Graph Index

## 1. Module Purpose & Capabilities

The index module implements DiskANN's core in-memory approximate nearest neighbor (ANN) index based on the **Vamana graph algorithm**. It provides a navigable small-world graph where each data point is a node connected to its approximate nearest neighbors. The module supports both **static batch builds** and **dynamic operations** (insert, delete, consolidate), with optional **label-based filtered search**, **PQ-compressed distance computation during build**, and **tag-based point identification**.

### Capabilities exposed externally:
- **Batch index construction** from files or in-memory arrays
- **Greedy graph search** (iterate-to-fixed-point) with configurable search list size L
- **Filtered search** with label-based constraints
- **Tag-based search** returning user-defined tags instead of internal IDs
- **Dynamic point insertion** with automatic graph edge updates
- **Lazy deletion** with deferred graph cleanup via consolidation
- **Save/Load** to/from disk (graph, data, tags, delete lists, label metadata)
- **FastL2 optimized layout search** for static float indexes
- **Index resizing** for growing datasets
- **Type-erased API** via `AbstractIndex` enabling runtime polymorphism over template parameters

### Key files:
- [include/index.h](include/index.h) — `Index<T, TagT, LabelT>` class declaration
- [src/index.cpp](src/index.cpp) — Full implementation (~3500 lines)
- [include/abstract_index.h](include/abstract_index.h) — `AbstractIndex` type-erased base class
- [src/abstract_index.cpp](src/abstract_index.cpp) — Type-erased method implementations + explicit template instantiations
- [include/index_factory.h](include/index_factory.h) — `IndexFactory` for string-based type dispatch
- [src/index_factory.cpp](src/index_factory.cpp) — Factory implementation with data/graph/PQ store construction
- [include/index_config.h](include/index_config.h) — `IndexConfig` + `IndexConfigBuilder`
- [include/index_build_params.h](include/index_build_params.h) — `IndexFilterParams` + `IndexFilterParamsBuilder`

---

## 2. Core Design Logic

### 2.1 Why Vamana Graph?

DiskANN uses the Vamana algorithm (not HNSW) which builds a single-layer navigable graph. The key design decisions:

1. **Single graph layer, not hierarchical**: Unlike HNSW, Vamana builds one graph with controlled degree. This simplifies the data structure and makes it amenable to disk-based layouts. The graph is stored as adjacency lists with a configurable maximum degree R (`_indexingRange`).

2. **Greedy search with occlusion-based pruning**: The `iterate_to_fixed_point()` function performs a greedy beam search from entry point(s) to find nearest neighbors. A `fast_iterate` flag (set when `total_num_points <= MAX_POINTS_FOR_USING_BITSET`) controls whether visited-node tracking uses a `boost::dynamic_bitset` (fast path) or a `tsl::robin_set` (large index path). The `occlude_list()` function prunes candidate neighbors using a distance-ratio criterion controlled by `alpha` — if a candidate is "occluded" by a closer already-selected neighbor (i.e., the ratio `candidate_distance / distance_to_closer_neighbor > alpha`), it is excluded. This controls graph sparsity vs. recall.

   **`occlude_list` algorithm details**: `cur_alpha` starts at 1.0 and is multiplied by 1.2f each round until it exceeds `alpha`. For **L2/COSINE**: `occlude_factor[t] = max(occlude_factor[t], iter2->distance / djk)` (with djk=0 yielding float::max). For **INNER_PRODUCT**: distances are negated (`x = -iter2->distance, y = -djk`), comparison is `y > cur_alpha * x`, and if true, `occlude_factor[t] = max(occlude_factor[t], eps)` where `eps = cur_alpha + 0.01f`.

3. **Frozen points as entry points**: For dynamic indexes, one or more "frozen points" are placed at position `_max_points` onward. These are never deleted and serve as stable graph entry points. `_start` is set to `_max_points` when frozen points exist (i.e., `_start` points to the first frozen slot). For static indexes, the medoid of the dataset is used as the entry point.

4. **Type erasure via AbstractIndex**: Since `Index<T, TagT, LabelT>` is a triple-template class, the `AbstractIndex` base uses `std::any` wrappers (`DataType`, `TagType`, `LabelType`, etc.) to provide a non-templated interface. This allows `IndexFactory` to construct indexes from string type names at runtime.

5. **Separation of data and graph stores**: The index delegates vector storage to `AbstractDataStore<T>` and graph storage to `AbstractGraphStore`. This enables pluggable backends (currently only in-memory implementations exist).

### 2.2 Build Strategy

The build process (`link()` in [src/index.cpp](src/index.cpp)) uses a two-pass approach:

- **Pass 1**: For each point (parallelized with `#pragma omp parallel for schedule(dynamic, 2048)`), run `search_for_point_and_prune()` which: (a) performs greedy search to find candidate neighbors, (b) prunes them via `occlude_list()`, (c) sets the pruned list as the node's neighbors, and (d) calls `inter_insert()` to add reverse edges to the neighbors.

- **Pass 2**: A cleanup pass (also `#pragma omp parallel for schedule(dynamic, 2048)`) that re-prunes any node whose degree exceeds `_indexingRange` (due to reverse edge insertion from Pass 1).

For **filtered indexes**, `search_for_point_and_prune()` makes **two separate `iterate_to_fixed_point()` calls**: first a filtered search (using label-specific start nodes and `filteredLindex`), then an unfiltered search (using global init IDs and `Lindex`). A `std::set<Neighbor>` is used to deduplicate/merge the two candidate pools. `scratch->clear()` is called between the two searches to reset state. After merging, self-loop references (node referencing itself) are removed from the pool before pruning.

### 2.3 Dynamic Index Design

Dynamic operations require:
- **Tags enabled** (`_enable_tags = true`): Every point has a user-facing `TagT` identifier. Internal location IDs are hidden.
- **At least 1 frozen point**: Ensures a stable search entry point even as points are inserted/deleted.
- **Two-phase deletion**: `lazy_delete()` marks a point for deletion (adds to `_delete_set`, removes from tag maps). `consolidate_deletes()` later repairs the graph by replacing deleted neighbors with second-order neighbors via `process_delete()`. If `_start` is in the delete set, an exception is thrown. The main processing loop uses `#pragma omp parallel for schedule(dynamic, 8192)` for parallelism.
- **Slot management**: `_empty_slots` tracks available positions. `reserve_location()` pops from this set; `release_locations()` returns slots after consolidation.
- **EXPAND_IF_FULL lock dance** (compile-time flag, default disabled): When `reserve_location()` returns -1 (no space) and `EXPAND_IF_FULL` is enabled: (1) releases `_delete_lock`, `_tag_lock`, and `_update_lock` (shared), (2) acquires `_update_lock` as exclusive, re-acquires `_tag_lock` and `_delete_lock`, (3) calls `resize(_max_points * INDEX_GROWTH_FACTOR)`, (4) releases all three locks, (5) re-acquires `_update_lock` (shared), `_tag_lock`, `_delete_lock`, (6) retries `reserve_location()` — throws exception if retry still fails.

### 2.4 Concurrency Model

The index uses four `std::shared_timed_mutex` locks with a strict acquisition order to prevent deadlocks:

1. `_update_lock` — RW lock between save/load (exclusive) and search/insert/delete/consolidate (shared)
2. `_consolidate_lock` — Ensures only one consolidate or compact operation at a time
3. `_tag_lock` — Protects `_tag_to_location`, `_location_to_tag`, `_empty_slots`, `_nd`, `_max_points`, `_label_to_start_id`
4. `_delete_lock` — Protects `_delete_set` and `_data_compacted`

Additionally, per-node `non_recursive_mutex` locks (`_locks[i]`) protect individual adjacency lists during concurrent modification.

When `_conc_consolidate` is true, consolidation can run concurrently with searches and inserts (using shared `_update_lock` instead of exclusive).

### 2.5 Key Trade-offs

| Decision | Trade-off |
|----------|-----------|
| `alpha` parameter | Higher alpha → more edges → better recall but slower build and larger graph |
| `saturate_graph` | When true, fills unused edge slots with any available candidates after occlusion pruning. Improves recall at cost of diversity |
| PQ distance during build | Uses compressed distances for candidate search (faster build) but re-computes exact distances during pruning. Not compatible with dynamic index |
| `GRAPH_SLACK_FACTOR` | Adjacency lists are over-allocated by this factor (default ~1.3x) to accommodate reverse edges without immediate pruning. In `inter_insert()`, pruning threshold is `GRAPH_SLACK_FACTOR * range` (not just `range`): under threshold, simply append via `add_neighbour` with no pruning; over threshold, build `copy_of_neighbors` (existing + n), compute actual distances from `des` to each candidate as `Neighbor` objects, and prune. Reserve size for the candidate pool is `ceil(1.05 * GRAPH_SLACK_FACTOR * range)` |
| Concurrent consolidate | Allows searches during consolidation at cost of per-node locking overhead |

---

## 3. State Flow

### 3.1 Index Construction (Static)

```
User calls: index.build(filename, num_points)
  → _update_lock (exclusive)
  → load data from file into _data_store
  → set _nd = num_points
  → build_with_data_populated(tags)
      → setup tag maps if enabled
      → initialize_query_scratch()
      → generate_frozen_point()     // copy medoid to frozen slot
      → link()                       // main graph construction
          → calculate_entry_point()  // if no frozen pts, find medoid
          → parallel for each point:
              search_for_point_and_prune(node, L)
                → iterate_to_fixed_point()  // greedy search
                → prune_neighbors()         // occlusion-based pruning
              inter_insert()                // add reverse edges
          → cleanup pass: re-prune over-degree nodes
```

### 3.2 Search Flow

```
User calls: index.search(query, K, L, indices, distances)
  → ScratchStoreManager gets a scratch space from _query_scratch pool
  → _update_lock (shared)
  → _data_store->preprocess_query()     // e.g., normalize for COSINE
  → iterate_to_fixed_point(scratch, L, init_ids, ...)
      → initialize candidates from _start + frozen points
      → while has_unexpanded_node():
          → pop closest unexpanded from best_L_nodes
          → get its neighbors from _graph_store
          → filter visited, compute distances via _pq_data_store
          → insert new candidates into best_L_nodes
  → extract top-K from best_L_nodes:
      → frozen points excluded by condition: best_L_nodes[i].id < _max_points
      → loop collects results until pos == K, then breaks
      → if pos < K after loop, prints warning to cerr
      → INNER_PRODUCT distances are negated before returning (-1 * distance)
  → return (hops, comparisons)
```

### 3.3 Insert Flow (Dynamic)

```
User calls: index.insert_point(point, tag, labels)
  → _update_lock (shared), _tag_lock (unique), _delete_lock (unique)
  → reserve_location() → get slot from _empty_slots or consecutive
  → setup _location_to_labels
  → for each new label not yet in _labels:
      if _frozen_pts_used >= _num_frozen_pts → throw exception
      allocate frozen point at location (_max_points + _frozen_pts_used)
      COPY inserting point's vector data to the new frozen point location
      register label→frozen_point in _label_to_start_id
      increment _frozen_pts_used
  → unlock _delete_lock
  → set tag maps, unlock _tag_lock
  → _data_store->set_vector(location, point)
  → search_for_point_and_prune(location, L)
  → set neighbors in _graph_store
  → inter_insert(location, pruned_list)  // add reverse edges
```

### 3.4 Delete + Consolidate Flow

```
lazy_delete(tag):
  → _update_lock (shared), _tag_lock (unique), _delete_lock (unique)
  → add location to _delete_set
  → remove from _location_to_tag and _tag_to_location
  → set _data_compacted = false

consolidate_deletes(params):
  → _update_lock (exclusive or shared if _conc_consolidate)
  → _consolidate_lock (unique)
  → swap _delete_set with empty set (old_delete_set)
  → parallel for each live point:
      process_delete(old_delete_set, loc, ...)
        → get adjacency list of loc
        → for each deleted neighbor: add its non-deleted neighbors to expanded set
        → if modified: re-prune via occlude_list()
  → release_locations(old_delete_set) → move deleted slots to _empty_slots
```

### 3.5 optimize_index_layout() (FastL2)

Builds the interleaved `_opt_graph` memory layout for `search_with_optimized_layout()`. **Only for static (non-dynamic) indexes** — throws exception if `_dynamic_index` is true.

```
_data_len  = (aligned_dim + 1) * sizeof(float)    // +1 for precomputed norm
_neighbor_len = (max_observed_degree + 1) * sizeof(uint32_t)  // +1 for the count
_node_size = _data_len + _neighbor_len
```

Per-node layout in `_opt_graph`: `[norm(4B)][vector data][neighbor_count(4B)][neighbor_ids...]`

Key steps:
- Casts distance function to `DistanceFastL2<T>*` to compute norms
- For each node: copies norm, then vector data, then degree count + neighbor IDs
- Calls `_graph_store->clear_graph()` and `_graph_store->resize_graph(0)` after building `_opt_graph`, freeing the graph store memory

### 3.6 ~Index() Destructor

- Acquires all four mutex locks as `unique_lock` in order: `_update_lock`, `_consolidate_lock`, `_tag_lock`, `_delete_lock`
- Acquires all per-node `_locks[i]` via `LockGuard` in a loop
- `delete[] _opt_graph` if non-null
- `_query_scratch` destroyed via `ScratchStoreManager::destroy()`

### 3.7 compact_data() Algorithm

- `new_location[]` computed by sequential `new_counter++` for occupied slots (those in `_location_to_tag`)
- Frozen points keep identity mapping: `new_location[old] = old`
- Adjacency lists are rewritten: neighbor IDs remapped through `new_location[]`; dangling references (to empty locations) are counted and logged to `cerr` ("#dangling references after data compaction: N")
- Data vectors and adjacency lists are physically moved via `_data_store->copy_vectors()` and `_graph_store->swap_neighbours()`
- `_tag_to_location` is rebuilt from `_location_to_tag` using the `new_location` mapping
- `_location_to_tag` is then rebuilt from the new `_tag_to_location`
- `_empty_slots` is rebuilt to cover `[_nd .. _max_points)`
- Sets `_data_compacted = true`

### 3.8 Save/Load Flow

**Save** writes 4+ files from a prefix:
- `{prefix}` — graph file (via `_graph_store->store()`)
- `{prefix}.data` — vector data (via `_data_store->save()`)
- `{prefix}.tags` — tag array
- `{prefix}.del` — pending delete set
- `{prefix}_labels.txt`, `{prefix}_labels_to_medoids.txt`, `{prefix}_universal_label.txt`, `{prefix}_labels_map.txt` — filter metadata

**Load** reads these files and reconstructs internal state, including `_delete_set`, tag maps, label maps, and `_empty_slots`.

### 3.9 Error Handling

- Construction validates: dynamic requires tags, PQ build incompatible with dynamic and INNER_PRODUCT
- Search throws if K > L
- Insert rejects tag 0 (reserved for frozen points), rejects duplicate tags
- Consolidate returns `consolidation_report` with status codes: SUCCESS, FAIL, LOCK_FAIL, INCONSISTENT_COUNT_ERROR
- Load validates dimension consistency and point count agreement between data/graph/tags files

---

## 4. Common Modification Scenarios

### Scenario 1: Add a new distance metric

1. Implement the metric in [include/distance.h](include/distance.h) / [src/distance.cpp](src/distance.cpp)
2. Update `IndexFactory::construct_inmem_distance_fn<T>()` in [src/index_factory.cpp](src/index_factory.cpp#L63) to handle the new `Metric` enum value
3. If the metric has special preprocessing needs (like COSINE normalization), update `_data_store->preprocess_query()` in the data store
4. Check `occlude_list()` in [src/index.cpp](src/index.cpp) around line 1100 — the pruning logic has metric-specific branches for L2/COSINE vs INNER_PRODUCT. Add a branch for your metric.
5. Check the distance sign convention in `search()` around line 2020 — INNER_PRODUCT distances are negated before returning.

### Scenario 2: Add a new data type (e.g., float16)

1. Add template instantiations at the bottom of [src/index.cpp](src/index.cpp) (lines ~3380-3500) for `Index<float16, TagT, LabelT>` for all tag/label type combos
2. Add template instantiations in [src/abstract_index.cpp](src/abstract_index.cpp) for the new data type
3. Update `IndexFactory::check_config()` in [src/index_factory.cpp](src/index_factory.cpp#L40) to accept the new type string
4. Update the `create_instance()` string-dispatch chain in [src/index_factory.cpp](src/index_factory.cpp#L160) to handle the new type string
5. Implement a distance function for the type in the distance module

### Scenario 3: Change the graph pruning strategy

The pruning logic is in two functions in [src/index.cpp](src/index.cpp):
- `occlude_list()` (line ~1070): The core occlusion criterion. Modify the `occlude_factor` computation and the `cur_alpha` loop to change how candidates are scored and selected.
- `prune_neighbors()` (line ~1175): Calls `occlude_list()` and optionally saturates the graph. The `_saturate_graph` logic (line ~1207) fills remaining slots.

For filter-aware pruning, the inner loop in `occlude_list()` at line ~1130 checks `_location_to_labels` — pruning is disallowed if the occluder doesn't share all labels with the occluded candidate.

### Scenario 4: Support a new storage backend (e.g., disk-backed data store)

1. Create a new class inheriting `AbstractDataStore<T>` (like `InMemDataStore<T>` in [include/in_mem_data_store.h](include/in_mem_data_store.h))
2. Add a new enum value to `DataStoreStrategy` in [include/index_config.h](include/index_config.h#L10)
3. Update `IndexFactory::construct_datastore<T>()` in [src/index_factory.cpp](src/index_factory.cpp#L74) to handle the new strategy
4. The `Index` class accesses data only through `_data_store` and `_pq_data_store` — no direct data access.

### Scenario 5: Add concurrent search thread count limiting

Modify `search()` in [src/index.cpp](src/index.cpp) (line ~1990). The scratch space pool (`_query_scratch`) already limits concurrency — it's a `ConcurrentQueue` initialized with a fixed number of scratch objects in `initialize_query_scratch()`. The `ScratchStoreManager` blocks when no scratch is available. To change the limit, modify the `num_threads` calculation during construction or in `initialize_query_scratch()` (line ~192).

### Scenario 6: Change the entry point selection strategy

- For static indexes: modify `calculate_entry_point()` (line ~745) which delegates to `_data_store->calculate_medoid()`
- For dynamic indexes: modify `generate_frozen_point()` (line ~2262) which copies the medoid vector to the frozen slot
- For filtered indexes: the medoid selection per label is in `build_filtered_index()` (line ~1900) — it randomly samples `num_cands=25` points per label and picks the one with the lowest medoid count to balance load
