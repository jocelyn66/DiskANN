# DiskANN Apps Module — Reference

## File: apps/CMakeLists.txt

Defines 8 executables. All link `${PROJECT_NAME}` (the diskann library) and `Boost::program_options`. Disk-related apps also link `${DISKANN_ASYNC_LIB}` (for `LinuxAlignedFileReader`). All link `${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS}` for memory allocation optimization. Installed on non-MSVC platforms.

---

## File: apps/build_memory_index.cpp

**Purpose**: Build a static in-memory Vamana graph index from a binary vector file.

**Entry**: `main()` — single function, no templates in main.

**Required Args**:
| Arg | Type | Description |
|-----|------|-------------|
| `--data_type` | string | `float`, `int8`, `uint8` |
| `--dist_fn` | string | `l2`, `mips`, `cosine` |
| `--data_path` | string | Input vectors in `.bin` format |
| `--index_path_prefix` | string | Output path prefix for index files |

**Optional Args**:
| Arg | Short | Default | Description |
|-----|-------|---------|-------------|
| `--num_threads` | `-T` | `omp_get_num_procs()` | Build threads |
| `--max_degree` | `-R` | 64 | Max graph degree |
| `--Lbuild` | `-L` | 100 | Search list size during build |
| `--alpha` | | 1.2 | Pruning parameter (1=sparse, 1.2-1.4=dense) |
| `--build_PQ_bytes` | | 0 | PQ bytes for build (0=full precision) |
| `--use_opq` | | false | Use Optimized PQ |
| `--label_file` | | "" | Label file for filtered index |
| `--universal_label` | | "" | Universal label string |
| `--FilteredLbuild` | | 0 | Build complexity for filtered points |
| `--label_type` | | "uint" | Label storage type |

**Flow**: Parse args → `get_bin_metadata()` → `IndexWriteParametersBuilder` → `IndexFilterParamsBuilder` → `IndexConfigBuilder` → `IndexFactory::create_instance()` → `index->build(data_path, data_num, filter_params)` → `index->save(prefix)`

**Key design choice**: Uses IndexFactory (type-erased) so the `main()` function itself is not templated — the factory handles type dispatch internally based on the `data_type` string in the config.

---

## File: apps/build_disk_index.cpp

**Purpose**: Build an SSD-optimized index with PQ compression for large-scale datasets.

**Entry**: `main()` — dispatches to `diskann::build_disk_index<T>()`.

**Additional Required Args** (beyond standard):
| Arg | Short | Description |
|-----|-------|-------------|
| `--search_DRAM_budget` | `-B` | DRAM budget in GB for search-time PQ compression level |
| `--build_DRAM_budget` | `-M` | DRAM budget in GB for index construction |

**Additional Optional Args**:
| Arg | Default | Description |
|-----|---------|-------------|
| `--QD` | 0 | Quantized dimension for compression |
| `--codebook_prefix` | "" | Pre-trained codebook path prefix |
| `--PQ_disk_bytes` | 0 | Bytes per vector on SSD (0=no compression) |
| `--append_reorder_data` | false | Store full-precision data alongside compressed |
| `--filter_threshold` | 0 | Max labels per node before graph splitting |

**Flow**: Parse args → validate constraints (reorder requires disk_PQ>0 and float) → build params string `"R L B M threads disk_PQ reorder build_PQ QD"` → dispatch `diskann::build_disk_index<T, LabelT>(data, prefix, params, metric, ...)`.

**Key detail**: The parameter string is passed as a space-separated C-string to the library function which parses it internally. The label type dispatch uses `uint16_t` when `label_type == "ushort"`.

---

## File: apps/build_stitched_index.cpp

**Purpose**: Build a filtered index by: (1) creating per-label sub-indices, (2) stitching them into a unified graph, (3) pruning.

**Entry**: `main()` → `handle_args()` for parsing.

**Key functions defined in this file**:
- `handle_args()` — CLI parsing extracted to separate function
- `save_full_index()` — Writes stitched graph binary + auxiliary files (labels, medoids, universal_label, data)
- `stitch_label_indices<T>()` — Unions per-label graph edges (no duplicate edges) into one graph; returns graph + expected file size
- `prune_and_save<T>()` — Loads stitched graph into `Index<T>`, calls `index.prune_all_neighbors(stitched_R, 750, 1.2)`, saves
- `clean_up_artifacts()` — Removes temporary per-label bin files and indices
- `print_progress()`, `random()` — inline helpers

**Additional Args**:
| Arg | Default | Description |
|-----|---------|-------------|
| `--stitched_R` | 100 | Final graph degree after pruning |

**Pipeline**: `convert_labels_string_to_int()` → `parse_label_file()` → `generate_label_specific_vector_files<T>()` → `generate_label_indices<T>()` → `stitch_label_indices<T>()` → `save_full_index()` → `prune_and_save<T>()` → `clean_up_artifacts()`

**Auxiliary files written**:
- `{prefix}_labels.txt` — copy of label file
- `{prefix}.data` — copy of input data
- `{prefix}_labels_to_medoids.txt` — label→medoid mapping
- `{prefix}_universal_label.txt` — if universal label set
- `{prefix}` — the graph file (binary: size, max_degree, entry_point, frozen_pts, adjacency lists)
- `{prefix}_full` — un-pruned stitched graph

---

## File: apps/search_disk_index.cpp

**Purpose**: k-NN search on a disk-based PQFlashIndex with beam search, caching, and optional filtering.

**Entry**: `main()` → dispatches to `search_disk_index<T, LabelT>()`.

**Template function**: `search_disk_index<T, LabelT>()` — loads index, caches nodes, runs search loop.

**Required Args**:
| Arg | Short | Description |
|-----|-------|-------------|
| `--query_file` | | Query vectors in bin format |
| `--recall_at` | `-K` | Number of nearest neighbors to return |
| `--search_list` | `-L` | Space-separated list of L values to test |
| `--result_path` | | Output path prefix |

**Optional Args**:
| Arg | Default | Description |
|-----|---------|-------------|
| `--gt_file` | "null" | Ground truth file for recall computation |
| `--beamwidth` | 2 | Beam width (0=auto-optimize) |
| `--num_nodes_to_cache` | 0 | Nodes to cache via BFS around medoid |
| `--search_io_limit` | uint32::max | Max I/O operations per query |
| `--use_reorder_data` | false | Use full-precision reorder data |
| `--filter_label` | "" | Single filter for all queries |
| `--query_filters_file` | "" | Per-query filter file |
| `--fail_if_recall_below` | 0.0 | Exit with error if recall below threshold |

**Search loop**: For each L in Lvec → optionally optimize beamwidth → parallel `cached_beam_search()` (or filtered variant with `get_converted_label()`) → collect `QueryStats` → compute QPS, latency (mean, p99.9), mean IOs, recall → print table → save results as `{prefix}_{L}_idx_uint32.bin` and `{prefix}_{L}_dists_float.bin`.

**Key objects**: `PQFlashIndex<T, LabelT>`, `LinuxAlignedFileReader` (or Windows variant), `QueryStats`.

**Helper**: `print_stats()` — formats percentile statistics table.

---

## File: apps/search_memory_index.cpp

**Purpose**: k-NN search on an in-memory index, supporting static/dynamic, tagged, and filtered search.

**Entry**: `main()` → dispatches to `search_memory_index<T, LabelT>()`.

**Additional Optional Args** (beyond standard search args):
| Arg | Default | Description |
|-----|---------|-------------|
| `--dynamic` | false | Whether index was built dynamically |
| `--tags` | false | Search using external tags |
| `--print_all_recalls` | false | Print Recall@1 through Recall@K |
| `--print_qps_per_thread` | false | Show QPS divided by thread count |

**Search dispatch** (4 paths per query):
1. `filtered_search && !tags` → `index->search_with_filters()`
2. `metric == FAST_L2` → `index->search_with_optimized_layout()`
3. `tags && !filtered` → `index->search_with_tags()`
4. `tags && filtered` → `index->search_with_tags(..., raw_filter)`
5. default → `index->search()`

**Key detail**: Uses `IndexFactory` + `IndexConfigBuilder`, determines `num_frozen_pts` from graph metadata via `diskann::get_graph_num_frozen_points()`. Supports `FAST_L2` metric (float only) which uses optimized memory layout.

---

## File: apps/range_search_disk_index.cpp

**Purpose**: Range search on disk index — find all points within a distance threshold.

**Entry**: `main()` → dispatches to `search_disk_index<T, LabelT>()` (note: same function name but different signature from the k-NN search app).

**Key difference from search_disk_index**: Uses `_pFlashIndex->range_search()` returning variable-sized result sets (`vector<vector<uint32_t>>` per query). Uses `calculate_range_search_recall()` for evaluation. Has `max_list_size = 10000` cap.

**Unique Required Args**:
| Arg | Short | Description |
|-----|-------|-------------|
| `--range_threshold` | `-K` | Distance threshold for range search |

**No** `--result_path`, `--filter_label`, or `--fail_if_recall_below` args.

---

## File: apps/test_insert_deletes_consolidate.cpp

**Purpose**: Test dynamic index operations — batch build, incremental inserts, lazy deletes, and consolidation.

**Entry**: `main()` → dispatches to `build_incremental_index<T>()`.

**Key functions**:
- `load_aligned_bin_part<T>()` — Read a slice of a bin file into aligned memory
- `get_save_filename()` — Generate checkpoint filename with skip/delete/threshold info
- `insert_till_next_checkpoint<T, TagT, LabelT>()` — OMP-parallel insert batch via `index.insert_point()`
- `delete_from_beginning<T, TagT>()` — Lazy-delete a range then `consolidate_deletes()`
- `build_incremental_index<T>()` — Main orchestration function

**Required Args**:
| Arg | Description |
|-----|-------------|
| `--points_to_skip` | Skip first N points from file |
| `--beginning_index_size` | Initial batch build size |
| `--points_per_checkpoint` | Insert batch size |
| `--checkpoints_per_snapshot` | Save to disk every N checkpoints |
| `--points_to_delete_from_beginning` | How many old points to delete |

**Optional Args**:
| Arg | Default | Description |
|-----|---------|-------------|
| `--max_points_to_insert` | 0 (=all) | Total points to process |
| `--do_concurrent` | false | Run inserts and deletes concurrently |
| `--start_deletes_after` | 0 | Start deletes after this many points |
| `--start_point_norm` | 0 | Radius for random start point |
| `--num_start_points` | `defaults::NUM_FROZEN_POINTS_DYNAMIC` | Frozen start points |

**Two modes**:
1. **Sequential** (`concurrent=false`): Insert all → delete from beginning → save
2. **Concurrent** (`concurrent=true`): Insert batches via `std::async`, trigger delete after `start_deletes_after` threshold

**Tags**: Uses `TagT = uint32_t`, tags are `1 + location_index` (1-based).

---

## File: apps/test_streaming_scenario.cpp

**Purpose**: Simulate a sliding-window streaming workload with concurrent insert and delete.

**Entry**: `main()` → dispatches to `build_incremental_index<T, TagT, LabelT>()`.

**Key functions**:
- `load_aligned_bin_part<T>()` — Same pattern as test_insert_deletes
- `get_save_filename()` — Uses active_window/consolidate_interval/max_points
- `insert_next_batch<T, TagT, LabelT>()` — OMP-parallel insert with error counting
- `delete_and_consolidate<T, TagT, LabelT>()` — Lazy delete + consolidate with **retry on LOCK_FAIL** and INCONSISTENT_COUNT_ERROR (waits 5 seconds and retries)
- `build_incremental_index<T, TagT, LabelT>()` — Sliding window orchestration

**Required Args** (unique to this app):
| Arg | Description |
|-----|-------------|
| `--active_window` | Number of live points in the sliding window |
| `--consolidate_interval` | Points added/deleted per step |
| `--start_point_norm` | Required: radius for random start point (no batch build) |

**Optional Args**:
| Arg | Default | Description |
|-----|---------|-------------|
| `--insert_threads` | `omp_get_num_procs()/2` | Threads for insertion |
| `--consolidate_threads` | `omp_get_num_procs()/2` | Threads for consolidation |

**Sliding window logic**:
1. Insert first `active_window` points
2. Loop: insert next `consolidate_interval` → wait for previous delete → delete oldest `consolidate_interval` (async)
3. Max index capacity = `active_window + 4 * consolidate_interval`

**Key difference from test_insert_deletes**: Always starts from random start points (no batch build), always concurrent, uses sliding window instead of append-then-delete.

---

## File: include/program_options_utils.hpp

**Purpose**: Shared string constants for CLI argument descriptions used across all apps.

**Namespace**: `program_options_utils`

**Key constants**:
- `DATA_TYPE_DESCRIPTION` — "data type, one of {int8, uint8, float}"
- `DISTANCE_FUNCTION_DESCRIPTION` — "distance function {l2, mips, fast_l2, cosine}"
- `INDEX_PATH_PREFIX_DESCRIPTION`, `RESULT_PATH_DESCRIPTION`, `QUERY_FILE_DESCRIPTION`
- `NUMBER_OF_RESULTS_DESCRIPTION`, `SEARCH_LIST_DESCRIPTION`, `INPUT_DATA_PATH`
- `FILTER_LABEL_DESCRIPTION`, `FILTERS_FILE_DESCRIPTION`, `LABEL_TYPE_DESCRIPTION`
- `GROUND_TRUTH_FILE_DESCRIPTION`, `NUMBER_THREADS_DESCRIPTION`, `FAIL_IF_RECALL_BELOW`
- `NUMBER_OF_NODES_TO_CACHE`, `BEAMWIDTH`
- `MAX_BUILD_DEGREE`, `GRAPH_BUILD_COMPLEXITY`, `GRAPH_BUILD_ALPHA`
- `BUIlD_GRAPH_PQ_BYTES` (note: typo "BUIlD" with lowercase L is in the codebase)
- `USE_OPQ`, `LABEL_FILE`, `UNIVERSAL_LABEL`, `FILTERED_LBUILD`

**Function**: `make_program_description(executable_name, description)` → formatted help string

---

## Utility Applications Catalog (apps/utils/)

### Format Converters

| File | Usage | Description |
|------|-------|-------------|
| `fvecs_to_bin.cpp` | `fvecs_to_bin <float/int8/uint8> input.fvecs output.bin` | Convert .fvecs to DiskANN .bin format |
| `fvecs_to_bvecs.cpp` | `fvecs_to_bvecs input.fvecs output.bvecs` | Convert float vecs to byte vecs |
| `ivecs_to_bin.cpp` | `ivecs_to_bin input.ivecs output.bin` | Convert .ivecs to .bin |
| `tsv_to_bin.cpp` | `tsv_to_bin <float/int8/uint8> input.tsv output.bin [npts] [dims]` | TSV text to binary |
| `bin_to_tsv.cpp` | `bin_to_tsv <float/int8/uint8> input.bin output.tsv` | Binary to TSV text |
| `float_bin_to_int8.cpp` | `float_bin_to_int8 input.bin output.bin bias` | Quantize float to int8 |
| `int8_to_float.cpp` | `int8_to_float input.bin output.bin` | Upcast int8 to float |
| `int8_to_float_scale.cpp` | `int8_to_float_scale input.bin output.bin bias scale` | Upcast with scaling |
| `uint8_to_float.cpp` | `uint8_to_float input.bin output.bin` | Upcast uint8 to float |
| `uint32_to_uint8.cpp` | `uint32_to_uint8 input.bin output.bin` | Downcast uint32 to uint8 |

### Index Pipeline Tools

| File | Usage | Description |
|------|-------|-------------|
| `generate_pq.cpp` | `generate_pq <type> data.bin prefix bytes rate PQ/OPQ` | Generate PQ codebooks and compress vectors |
| `partition_data.cpp` | `partition_data <type> data.bin prefix rate nparts k_index` | k-means partition data (DEPRECATED) |
| `partition_with_ram_budget.cpp` | `partition_with_ram_budget <type> data.bin prefix rate budget k_index` | Partition with RAM constraint |
| `merge_shards.cpp` | `merge_shards prefix suffix idmaps_pfx idmaps_sfx nshards max_deg out_index out_medoids` | Merge per-shard indices |
| `create_disk_layout.cpp` | `create_disk_layout <type> data.bin vamana.index output.diskindex` | Create disk-optimized layout from in-memory index |

### Evaluation Tools

| File | Args Style | Description |
|------|-----------|-------------|
| `compute_groundtruth.cpp` | Boost.PO: `--data_type --dist_fn --base_file --query_file --gt_file --K [--tags_file]` | Brute-force exact k-NN using MKL BLAS |
| `compute_groundtruth_for_filters.cpp` | Boost.PO | Ground truth for filtered search (requires label file) |
| `calculate_recall.cpp` | Positional: `<gt.bin> <results.bin> <recall_at>` | Compute recall from saved results |
| `simulate_aggregate_recall.cpp` | Positional args | Simulate aggregate recall from multiple shards |

### Analysis and Data Generation

| File | Args Style | Description |
|------|-----------|-------------|
| `vector_analysis.cpp` | Positional: `<type> <action> <input> [output]` | Analyze norms, normalize, augment for MIPS |
| `count_bfs_levels.cpp` | Boost.PO: `--data_type --index_path_prefix --data_dims` | Count BFS depth levels in graph |
| `stats_label_data.cpp` | Boost.PO: `--labels_file --universal_label [--density]` | Label distribution statistics |
| `rand_data_gen.cpp` | Boost.PO | Generate random test vectors (float/int8/uint8) |
| `gen_random_slice.cpp` | Positional args | Extract random slice from dataset |
| `generate_synthetic_labels.cpp` | Boost.PO | Generate Zipf-distributed synthetic labels |

---

## Binary File Format (.bin)

Used across all apps for vectors and results:
```
[4 bytes: int32 num_points][4 bytes: int32 num_dims]
[num_points × num_dims × sizeof(T) bytes: row-major data]
```

Ground truth format (from compute_groundtruth):
```
[4 bytes: npts][4 bytes: K]
[npts × K × 4 bytes: uint32 neighbor IDs]
[npts × K × 4 bytes: float distances]  (optional)
```

---

## CMake Link Dependencies Summary

| App | diskann | Boost::PO | ASYNC_LIB | TCMALLOC | MKL |
|-----|---------|-----------|-----------|----------|-----|
| build_memory_index | ✓ | ✓ | | ✓ | |
| build_disk_index | ✓ | ✓ | ✓ | ✓ | |
| build_stitched_index | ✓ | ✓ | | ✓ | |
| search_memory_index | ✓ | ✓ | ✓ | ✓ | |
| search_disk_index | ✓ | ✓ | ✓ | ✓ | |
| range_search_disk_index | ✓ | ✓ | ✓ | ✓ | |
| test_insert_deletes_consolidate | ✓ | ✓ | | ✓ | |
| test_streaming_scenario | ✓ | ✓ | | ✓ | |
| compute_groundtruth | ✓ | ✓ | ✓ | | ✓ |
| compute_groundtruth_for_filters | ✓ | ✓ | ✓ | | ✓ |

`DISKANN_ASYNC_LIB` = `aio` on Linux (for aligned file reader). Required by any app that does disk I/O.
`DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS` = optional tcmalloc linkage from gperftools.
