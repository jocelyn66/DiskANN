---
name: DiskANN-dr-apps
description: Use when working with the apps module of DiskANN — CLI tools for building memory/disk indices, searching, and testing dynamic operations
---

# DiskANN Apps Module

The `apps/` directory contains all CLI entry points for DiskANN operations: building indices, searching, testing dynamic insert/delete, and data utility tools. Every app links against the core `diskann` library and `Boost::program_options`.

## Module Purpose & Capabilities

### Core Applications (apps/)

| App | Purpose |
|-----|---------|
| `build_memory_index` | Build an in-memory Vamana graph index |
| `build_disk_index` | Build an SSD-optimized PQ-compressed disk index |
| `build_stitched_index` | Build a filtered index by stitching per-label sub-indices |
| `search_memory_index` | Search an in-memory index (k-NN) |
| `search_disk_index` | Search a disk-based PQ flash index (k-NN) |
| `range_search_disk_index` | Range search on a disk index (all points within distance threshold) |
| `test_insert_deletes_consolidate` | Test dynamic insert/delete/consolidate on an in-memory index |
| `test_streaming_scenario` | Test sliding-window streaming: concurrent insert + delete with consolidation |

### Utility Applications (apps/utils/)

**Format converters**: `fvecs_to_bin`, `fvecs_to_bvecs`, `ivecs_to_bin`, `tsv_to_bin`, `bin_to_tsv`, `float_bin_to_int8`, `int8_to_float`, `int8_to_float_scale`, `uint8_to_float`, `uint32_to_uint8`

**Index pipeline tools**: `generate_pq`, `partition_data`, `partition_with_ram_budget`, `merge_shards`, `create_disk_layout`

**Evaluation**: `compute_groundtruth` (MKL-based brute-force k-NN), `compute_groundtruth_for_filters`, `calculate_recall`, `simulate_aggregate_recall`

**Analysis**: `vector_analysis` (norm stats, normalization, augmentation), `count_bfs_levels`, `stats_label_data`

**Data generation**: `rand_data_gen`, `gen_random_slice`, `generate_synthetic_labels`

---

## Core Design Logic

### Pattern 1: Boost.Program_Options CLI Parsing

Every main app uses `boost::program_options` with a consistent structure:

```cpp
namespace po = boost::program_options;

po::options_description desc{
    program_options_utils::make_program_description("app_name", "Description")};

// Required parameters group
po::options_description required_configs("Required");
required_configs.add_options()("data_type", po::value<std::string>(&data_type)->required(),
                               program_options_utils::DATA_TYPE_DESCRIPTION);
// ... more required params

// Optional parameters group
po::options_description optional_configs("Optional");
optional_configs.add_options()("num_threads,T",
                               po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                               program_options_utils::NUMBER_THREADS_DESCRIPTION);
// ... more optional params

desc.add(required_configs).add(optional_configs);
po::variables_map vm;
po::store(po::parse_command_line(argc, argv, desc), vm);
po::notify(vm);
```

Shared description strings live in `include/program_options_utils.hpp` in namespace `program_options_utils`. The function `make_program_description(name, desc)` creates formatted help text.

### Pattern 2: Distance Metric String-to-Enum Mapping

Every app that accepts `--dist_fn` maps the string to `diskann::Metric`:

```cpp
if (dist_fn == "l2")        metric = diskann::Metric::L2;
else if (dist_fn == "mips") metric = diskann::Metric::INNER_PRODUCT;
else if (dist_fn == "cosine") metric = diskann::Metric::COSINE;
// search_memory_index additionally supports "fast_l2" → diskann::Metric::FAST_L2
```

### Pattern 3: Type Dispatching (data_type × label_type)

All apps use runtime string-to-template dispatching. The `--data_type` parameter selects the template instantiation:

```cpp
if (data_type == "float")
    return do_work<float>(...);
else if (data_type == "int8")
    return do_work<int8_t>(...);
else if (data_type == "uint8")
    return do_work<uint8_t>(...);
```

When filtered search is involved, there's a second dispatch on `--label_type`:
- `"uint"` / `"uint32"` → `LabelT = uint32_t` (default)
- `"ushort"` / `"uint16"` → `LabelT = uint16_t`

This creates a 3×2 dispatch matrix (or 3×1 when labels aren't used). Apps that support filtering (search_disk_index, search_memory_index, test_streaming_scenario) dispatch on both. The tag type `TagT` is always `uint32_t`.

### Pattern 4: IndexFactory for Index Creation

The newer apps (build_memory_index, search_memory_index, test_insert_deletes_consolidate, test_streaming_scenario) use the builder pattern:

```cpp
auto config = diskann::IndexConfigBuilder()
    .with_metric(metric)
    .with_dimension(dim)
    .with_max_points(max_pts)
    .with_data_type(data_type_string)
    .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
    .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
    .is_dynamic_index(false)
    // ... more config
    .build();
auto index_factory = diskann::IndexFactory(config);
auto index = index_factory.create_instance();
```

Older apps (build_disk_index, build_stitched_index) call library functions directly (e.g., `diskann::build_disk_index<T>(...)`) with a space-separated parameter string.

### Pattern 5: Utility Apps Are Simpler

Utils use two patterns:
1. **Boost.Program_Options** (compute_groundtruth, rand_data_gen, stats_label_data, count_bfs_levels, generate_synthetic_labels)
2. **Raw argc/argv** (fvecs_to_bin, generate_pq, create_disk_layout, merge_shards, partition_data) — simpler tools with positional arguments

---

## Core Data Structures

### IndexWriteParameters / IndexWriteParametersBuilder
Controls graph construction: `L` (search list size during build), `R` (max degree), `alpha` (pruning parameter), `num_threads`, `filter_list_size`, `max_occlusion_size`, `saturate_graph`.

### IndexConfig / IndexConfigBuilder
Full index configuration: metric, dimension, max_points, data_type, label_type, tag_type, dynamic flag, PQ settings, frozen points count, store strategies.

### IndexSearchParams
Wraps search-time parameters: `search_list_size` and `num_threads`.

### IndexFilterParams / IndexFilterParamsBuilder
Filter-specific params for build: `universal_label`, `label_file`, `save_path_prefix`.

### QueryStats
Per-query statistics populated during disk search: `total_us`, `io_us`, `cpu_us`, `n_ios`. Used to compute mean latency, p99.9 latency, mean IOs.

### consolidation_report
Returned by `consolidate_deletes()`: `_active_points`, `_max_points`, `_empty_slots`, `_slots_released`, `_delete_set_size`, `_time`, `_status`.

---

## State Flow

### Build Memory Index Flow
```
CLI args → po::parse → metric mapping → get_bin_metadata(data_path)
→ IndexWriteParametersBuilder(L, R).with_filter_list_size(Lf).with_alpha(alpha)...build()
→ IndexFilterParamsBuilder → IndexConfigBuilder
→ IndexFactory(config).create_instance() → index->build(data_path, data_num, filter_params)
→ index->save(prefix)
```

### Build Disk Index Flow
```
CLI args → po::parse → metric mapping → validate constraints
→ build parameter string "R L B M threads disk_PQ reorder build_PQ QD"
→ diskann::build_disk_index<T>(data, prefix, params, metric, opq, ...)
```
The `build_disk_index` library function internally handles PQ generation, partitioning, shard building, merging, and disk layout creation.

### Build Stitched Index Flow
```
CLI args → handle_args() → convert_labels_string_to_int()
→ parse_label_file() → generate_label_specific_vector_files<T>()
→ generate_label_indices<T>() (build per-label Vamana index)
→ stitch_label_indices<T>() (union edges across per-label graphs)
→ save_full_index() → prune_and_save<T>() (prune to stitched_R)
→ clean_up_artifacts()
```

`save_full_index` writes the graph binary with hardcoded `index_entry_point = 0` and `index_num_frozen_points = 0`. The header metadata size is `METADATA = 2 * sizeof(uint64_t) + 2 * sizeof(uint32_t)` (index_size, num_frozen_points as uint64 + max_observed_degree, entry_point as uint32).

### Search Memory Index Flow
```
CLI args → po::parse → metric mapping → IndexConfigBuilder
→ IndexFactory::create_instance() → index->load(path)
→ [optimize_index_layout() if FAST_L2]
→ for each L in Lvec:
    → parallel: index->search() / search_with_filters() / search_with_tags()
    → compute recall vs ground truth → print stats table
→ save result bins
```

### Search Disk Index Flow
```
CLI args → po::parse → metric mapping → PQFlashIndex<T>(reader, metric)
→ _pFlashIndex->load() → cache_bfs_levels() → load_cache_list()
→ [optional warmup]
→ for each L in Lvec:
    → [optimize beamwidth if W=0]
    → parallel: cached_beam_search() (with optional filter label)
    → compute recall vs ground truth → print stats table
→ save result bins
```

### Range Search Disk Index Flow
Same as search_disk_index but calls `_pFlashIndex->range_search()` instead of `cached_beam_search()`, returning variable-size result sets. Uses `calculate_range_search_recall()` for evaluation.

### Test Insert/Deletes/Consolidate Flow
```
CLI args → IndexWriteParametersBuilder → IndexConfigBuilder
→ IndexFactory::create_instance()
→ if beginning_index_size == 0 && start_point_norm == 0: prints error and returns -1
→ if beginning_index_size > 0: index->build(data, beginning_index_size, tags)
→ if beginning_index_size == 0: index->set_start_points_at_random(start_point_norm)
→ loop: load_aligned_bin_part → insert_till_next_checkpoint (OMP parallel)
    → [optional: concurrent delete_from_beginning + consolidate_deletes]
→ delete_from_beginning if non-concurrent
→ index->save()
```
The `concurrent` flag controls whether inserts and deletes run simultaneously via `std::async`.

### Test Streaming Scenario Flow
```
CLI args → IndexConfigBuilder → IndexFactory::create_instance()
→ validate: max_points_to_insert >= active_window + consolidate_interval (else ANNException)
→ validate: consolidate_interval >= max_points_to_insert / 1000 (else ANNException "consolidate_interval is too small")
→ set_start_points_at_random(norm)
→ insert initial active_window points
→ sliding window loop:
    insert next consolidate_interval points (async)
    delete oldest consolidate_interval points (async)
    consolidate_deletes with retry on LOCK_FAIL
→ index->save()
```
The sliding window maintains `active_window` live points. Every step adds `consolidate_interval` new points to the right while deleting the same count from the left.

---

## Common Modification Scenarios

### 1. Adding a New Data Type (e.g., float16)

**Files to modify:**
- Every app's `main()` that does type dispatch — add an `else if (data_type == "float16")` branch
- `include/program_options_utils.hpp` — update `DATA_TYPE_DESCRIPTION`
- Every utils app with type dispatch

**Pattern to follow:**
```cpp
else if (data_type == std::string("float16"))
    return do_work<float16_t>(...);
```

This is a cross-cutting concern because every app dispatches independently.

### 2. Adding a New CLI Parameter to an Existing App

**Steps:**
1. Add variable declaration in `main()` alongside other params
2. Add `po::value<T>(&var)->default_value(...)` to the appropriate options group (required_configs or optional_configs)
3. Use the description constants from `program_options_utils` or add a new one in `include/program_options_utils.hpp`
4. Pass the new parameter through to the templated work function
5. Use it in the logic

**Example — adding a `--verbose` flag to search_memory_index:**
```cpp
// In main():
bool verbose;
optional_configs.add_options()("verbose", po::bool_switch(&verbose)->default_value(false),
                               "Enable verbose output");
// Pass to search_memory_index<T>() template function
```

### 3. Adding a New App/Executable

**Steps:**
1. Create `apps/new_app.cpp` with the standard structure:
   - Include `program_options_utils.hpp`, relevant DiskANN headers
   - `namespace po = boost::program_options;`
   - Template work function with data type parameter
   - `main()` with po parsing, metric mapping, type dispatch
2. Add to `apps/CMakeLists.txt`:
   ```cmake
   add_executable(new_app new_app.cpp)
   target_link_libraries(new_app ${PROJECT_NAME} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options)
   ```
3. Add to the install targets list in the same CMakeLists.txt

### 4. Adding a New Distance Metric

**Files to modify:**
- Every app's metric mapping block: add `else if (dist_fn == "new_metric") metric = diskann::Metric::NEW_METRIC;`
- `include/program_options_utils.hpp` — update `DISTANCE_FUNCTION_DESCRIPTION`
- Validation checks (some apps restrict metrics to float data type)

### 5. Converting an Older App to Use IndexFactory

**Pattern:** Replace direct `diskann::Index<T>` construction with:
1. Build `IndexConfig` via `IndexConfigBuilder`
2. Create `IndexFactory` from config
3. Call `index_factory.create_instance()` returning `std::unique_ptr<AbstractIndex>`
4. Use abstract interface methods: `build()`, `load()`, `search()`, `save()`

The build_stitched_index app still uses direct `Index<T>` construction for its pruning step and could be modernized this way.

---

## Key Relationships to Other Modules

- **index module**: All build/search apps ultimately call `Index<T>` or `AbstractIndex` methods
- **disk_utils module**: `build_disk_index` calls `diskann::build_disk_index<T>()` which orchestrates PQ, partition, shard build, merge, and disk layout
- **pq_flash_index module**: `search_disk_index` and `range_search_disk_index` instantiate `PQFlashIndex<T>` for SSD-based search
- **filter_utils module**: Stitched index and filtered search apps use label parsing and per-label vector generation
- **index_factory module**: Newer apps use `IndexFactory` + `IndexConfigBuilder` for type-erased index creation
- **program_options_utils.hpp**: Shared CLI description strings used across all apps

## See Also

- [reference.md](reference.md) — Per-file function index, argument tables, and utils catalog
