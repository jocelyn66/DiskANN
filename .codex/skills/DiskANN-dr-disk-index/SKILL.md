---
name: DiskANN-dr-disk-index
description: Use when working with the disk-index module of DiskANN — PQFlashIndex for SSD-based approximate nearest neighbor search with beam search, caching, and disk layout utilities
---

# DiskANN Disk-Index Module

## Module Purpose & Capabilities

The disk-index module implements **SSD-resident approximate nearest neighbor (ANN) search** via `PQFlashIndex`. It stores a Vamana graph index on disk (in sector-aligned layout), keeps PQ-compressed vectors in RAM for fast approximate distance calculations, and uses a **beam search** algorithm that fetches graph neighborhoods from SSD in batches. The module also provides the full build pipeline: partitioning data into shards, building per-shard Vamana indices in RAM, merging them, and writing the final sector-aligned disk layout. A two-level caching system (BFS-based and sample-query-based) keeps frequently-accessed nodes in memory to minimize SSD I/O.

### Key Capabilities Exposed Externally

1. **Disk index build** — `build_disk_index()` orchestrates end-to-end construction from raw vectors to a ready-to-search disk index.
2. **Disk index search** — `PQFlashIndex::cached_beam_search()` performs ANN search with configurable beam width, IO limits, filter support, and optional full-precision reordering.
3. **Range search** — `PQFlashIndex::range_search()` finds all neighbors within a distance threshold.
4. **Cache generation** — `cache_bfs_levels()` and `generate_cache_list_from_sample_queries()` produce node lists for in-memory caching.
5. **Disk layout creation** — `create_disk_layout()` transforms a Vamana in-memory index + base vectors into sector-aligned format.
6. **Shard merging** — `merge_shards()` merges multiple per-shard Vamana graphs into a single graph.
7. **Beam width optimization** — `optimize_beamwidth()` auto-tunes beam width for QPS/latency tradeoff.

---

## Core Design Logic

### Why This Architecture?

DiskANN solves the problem of **billion-scale ANN search** where the full graph index cannot fit in RAM. The design makes three key trade-offs:

1. **PQ in RAM + full vectors on disk**: Keeps PQ-compressed representations (~4-64 bytes/point) in RAM for cheap approximate distance computation, while the full-precision vectors and graph edges live on SSD. This is the key insight that makes billion-scale search feasible with limited RAM.

2. **Sector-aligned disk layout**: The disk file is organized into 4096-byte sectors. Each sector holds either multiple nodes (if `max_node_len < SECTOR_LEN`) or a single node spans multiple sectors (if `max_node_len > SECTOR_LEN`). This alignment enables O_DIRECT reads, bypassing the OS page cache for predictable I/O performance.

3. **Beam search instead of sequential traversal**: Instead of fetching one node at a time (which would serialize I/O), the algorithm expands `beam_width` nodes simultaneously, issuing all their disk reads as a batch via Linux AIO (`io_submit`/`io_getevents`). This amortizes I/O latency across multiple reads.

### Beam Search Mental Model

The search maintains a priority queue (`retset`) of candidate nodes sorted by PQ distance to the query. Each iteration:

1. **Select beam**: Pick up to `beam_width` unexpanded nodes from `retset`. Check cache first — cached nodes are processed immediately without disk I/O.
2. **Issue batch reads**: For non-cached nodes, compute sector offsets and issue all reads as one batch.
3. **Process neighborhoods**: For each fetched node, compute its true distance to the query (using full-precision coords from disk or disk-PQ), then compute PQ distances to all its neighbors. Insert promising neighbors into `retset`.
4. **Terminate**: When `retset` has no more unexpanded nodes or `io_limit` is reached.

After search completes, optionally **reorder** the top results using full-precision vectors stored in a separate section of the disk file (the "reorder data" region).

### Caching Strategy

Two-level caching to minimize SSD reads for hot nodes:

- **Coordinate cache** (`_coord_cache`): Maps node_id → full-precision vector data. Used to compute exact distances without disk reads.
- **Neighborhood cache** (`_nhood_cache`): Maps node_id → (num_neighbors, neighbor_ids). Allows exploring a cached node's edges without fetching its sector.

Cache population methods:
- **BFS from medoids** (`cache_bfs_levels`): Does BFS from the medoid(s) to collect the first N nodes. Good for general workloads since nearby-medoid nodes are accessed on virtually every query.
  - **Maximum cache cap**: 10% of `_num_points` (`tenp_nodes = round(_num_points * 0.1)`). If `tenp_nodes == 0`, it is set to 1.
  - **BFS expansion**: Reads neighbor lists in blocks of **1024** (`BLOCK_SIZE`). Each block issues batched disk reads via `read_nodes()`, processes returned neighborhoods, and inserts discovered neighbors into the next BFS level.
- **Sample query profiling** (`generate_cache_list_from_sample_queries`): Runs sample queries and tracks which nodes are most visited. Good for skewed workloads.

### Filter Support

Filtered search uses per-label medoids. When a filter label is specified, search starts from the label-specific medoid. During neighbor expansion, nodes not matching the filter are skipped. A "universal label" can be assigned to nodes that match all filters.

Dense points (with many labels) are split into "dummy" copies, each carrying a subset of labels. A dummy→real mapping resolves final IDs.

### Distance Metric Handling

- **L2**: Used directly.
- **Inner Product / Cosine**: Data is pre-processed at build time (normalization + extra dimension for MIPS-to-L2 reduction). At search time, queries are normalized and the L2 comparator is used.
- **Integral types (int8_t, uint8_t) with COSINE/MIPS**: The constructor prints a WARNING ("Cannot normalize integral data types… Consider using L2") but proceeds with the **original metric unchanged** (no remap to L2). This can produce erroneous results or poor recall.

---

## State Flow

### Build Pipeline

```
raw vectors + params
    │
    ├── [Optional] preprocess_base_file() — normalize for cosine/MIPS
    │
    ├── generate_quantized_data() — create PQ pivots + compressed vectors
    │
    ├── [Optional] generate_disk_quantized_data() — if using disk PQ
    │
    ├── build_merged_vamana_index()
    │       ├── partition_with_ram_budget() — split data into shards
    │       ├── For each shard: build Index<T> in memory → save
    │       └── merge_shards() → merged vamana graph file
    │
    ├── create_disk_layout() — sector-aligned file from merged graph + base vectors
    │       Writes: [sector 0: metadata] [sectors 1..N: graph+coords] [optional: reorder data sectors]
    │
    └── gen_random_slice() — sample file for warmup/cache generation
```

### Load Pipeline (`PQFlashIndex::load_from_separate_paths`)

```
1. Load PQ table from `_pq_pivots.bin`
2. Load compressed vectors from `_pq_compressed.bin` into RAM (this->data)
3. [Optional] Load labels, label map, filter medoids, dummy map
4. [Optional] Load disk PQ table if disk uses PQ compression
5. Read disk index metadata (sector 0): num_points, dims, medoid, max_node_len, nnodes_per_sector, frozen point info, reorder info
6. Open disk file via AlignedFileReader
7. Setup thread-specific IO contexts (register_thread + io_setup)
8. Load medoids + centroids
9. [Optional] Load max_base_norm for inner product rescaling
```

### Search Flow (`cached_beam_search`)

```
1. Acquire thread scratch space (ScratchStoreManager)
2. Normalize query if MIPS/cosine
3. Preprocess PQ query: center, rotate, compute per-chunk distance tables
4. Find best medoid (closest centroid to query)   - **Unfiltered**: uses `_dist_cmp_float->compare()` — full-precision float distance against stored centroid data
   - **Filtered**: uses PQ distance via the `compute_dists` lambda (no centroid data stored for per-label medoids)
   - Missing filter label throws `ANNException("Cannot find medoid for specified filter.")`5. Insert medoid into retset with PQ distance
6. WHILE retset has unexpanded nodes AND io_limit not reached:
   a. Pop up to beam_width unexpanded nodes
   b. Partition into cached_nhoods vs frontier (need disk read)
   c. For frontier: build AlignedRead requests, issue batch read via reader->read()
   d. For cached_nhoods: compute exact distance, expand neighbors using PQ distances
   e. For frontier_nhoods (from disk): parse node buf → compute exact distance → expand neighbors via PQ
   f. Filter: skip dummy/non-matching-label nodes
7. Sort full_retset by distance
8. [Optional reorder]: read full-precision vectors for top-k*3 results, recompute exact distances, re-sort
9. Copy top-k results, map dummy→real IDs, rescale distances for MIPS
```

### I/O Flow

```
AlignedFileReader (abstract)
    ├── LinuxAlignedFileReader (O_DIRECT + Linux AIO: io_setup/io_submit/io_getevents)
    └── WindowsAlignedFileReader (FILE_FLAG_NO_BUFFERING + IOCP: ReadFile + GetQueuedCompletionStatus)

Each thread has its own IOContext (io_context_t on Linux, HANDLE+IOCP on Windows).
Thread registration: register_thread() → io_setup(MAX_EVENTS=1024, &ctx)
Reads: AlignedRead{offset, len, buf} all 512-aligned → io_submit batch → io_getevents wait
```

**`execute_io` retries:** The `n_retries` parameter defaults to **0** (no retry by default). On `io_submit` or `io_getevents` failure the process calls `exit(-1)` immediately, so the retry loop only activates if callers explicitly pass a non-zero value.

**`cached_ifstream::read()` cache behavior (include/cached_io.h):**
- When a read request exceeds the remaining cache (`n_bytes > cache_size - cur_off`), the cached portion is memcpy'd, the remainder is read directly from the file, and then `cur_off` is set to `cache_size`.
- The cache is refilled ONLY if the remaining file size (`fsize - reader.tellg()`) >= `cache_size`.
- When `size_left < cache_size`, `cur_off` stays at `cache_size`, which makes the condition `n_bytes <= (cache_size - cur_off)` always false (since `cache_size - cur_off == 0`). This effectively **disables the cache for all subsequent reads**, causing them to fall through to the direct-from-file path every time.

### Error Handling

- Exceptions: `ANNException`, `FileException` thrown on I/O failures, metadata mismatches, invalid parameters
- On Linux AIO: if `io_submit` or `io_getevents` return error, the process exits (`exit(-1)`)
- On Windows/Bing infra: per-request status tracked; failed reads return `retval[i] = false` in `read_nodes`
- PQ chunk count validated against `MAX_PQ_CHUNKS` (512)
- Graph degree validated against `MAX_GRAPH_DEGREE` (512)

---

## Common Modification Scenarios

### 1. Adding a New Distance Metric

**Key files:**
- `src/pq_flash_index.cpp` — constructor (line ~33): the `metric_to_invoke` logic that maps metrics to L2 for float types
- `src/pq_flash_index.cpp` — `cached_beam_search` (line ~1290): query normalization block
- `src/pq_flash_index.cpp` — `cached_beam_search` (line ~1650): distance rescaling in final results
- `src/disk_utils.cpp` — `build_disk_index` (line ~1200): preprocessing block for inner product/cosine

**What to do:** Add a new case in the constructor's metric mapping, add preprocessing in `build_disk_index`, and handle the new metric in the distance re-scaling section of `cached_beam_search`.

### 2. Changing the Disk Sector Size

**Key files:**
- `include/defaults.h` — `SECTOR_LEN = 4096`
- `src/disk_utils.cpp` — `create_disk_layout()`: all sector buffer allocations and layout math
- `src/pq_flash_index.cpp` — `get_node_sector()`, `offset_to_node()`: sector addressing
- `include/aligned_file_reader.h` — `AlignedRead` struct: assertions require 512-alignment (not 4096)

**What to do:** Change `defaults::SECTOR_LEN`. Ensure all `alloc_aligned` calls and AlignedRead offsets/lengths remain multiples of 512 (the kernel's O_DIRECT minimum). The layout will automatically adapt since it uses `SECTOR_LEN` throughout.

### 3. Adding Async I/O on Linux (Currently Synchronous Wait)

**Key file:** `src/linux_aligned_file_reader.cpp` — `execute_io()` function

**Current behavior:** `io_submit` followed by blocking `io_getevents` waiting for all ops. The `async` parameter in `LinuxAlignedFileReader::read()` is ignored with a warning.

**What to do:** Implement a non-blocking path: after `io_submit`, return immediately; provide a `wait()` method similar to the Windows/Bing infra path. This would require modifying the `cached_beam_search` loop to process completed reads individually (similar to the `#ifdef USE_BING_INFRA` block that already exists).

### 4. Modifying the Caching Strategy

**Key files:**
- `src/pq_flash_index.cpp` — `cache_bfs_levels()` (line ~360): BFS-based cache selection
- `src/pq_flash_index.cpp` — `generate_cache_list_from_sample_queries()` (line ~268): profiling-based cache selection
- `src/pq_flash_index.cpp` — `load_cache_list()` (line ~208): actually loads nodes into `_nhood_cache` and `_coord_cache`
- `src/pq_flash_index.cpp` — `cached_beam_search()` (line ~1434): cache lookup during search (`_nhood_cache.find(nbr.id)`)

**What to do:** To add an adaptive/LRU cache, replace the static `tsl::robin_map` caches with a concurrent LRU structure. Modify the cache lookup in `cached_beam_search` and add eviction logic. The `load_cache_list()` function demonstrates the data format expected by the cache maps.

### 5. Supporting a New Data Type (e.g., float16)

**Key files:**
- `src/pq_flash_index.cpp` — template instantiations at bottom (currently: `float`, `int8_t`, `uint8_t`)
- `src/disk_utils.cpp` — template instantiations at bottom
- `include/pq_flash_index.h` — template declaration `template <typename T, typename LabelT>`

**What to do:** Add explicit template instantiations for the new type. Ensure `sizeof(T)` works correctly in `_disk_bytes_per_point` calculations. You may need to add a distance function specialization.

### 6. Adding a New Index File Format / Metadata Field

**Key files:**
- `src/disk_utils.cpp` — `create_disk_layout()` (line ~870): writes sector 0 metadata via `output_file_meta` vector
- `src/pq_flash_index.cpp` — `load_from_separate_paths()` (line ~1025): reads metadata with `READ_U64` macros

**What to do:** Append the new field to `output_file_meta` in `create_disk_layout`. Read it with `READ_U64` in the load function at the corresponding position. The metadata is stored as a binary vector of uint64_t values in sector 0.
