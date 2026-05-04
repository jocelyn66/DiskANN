---
name: DiskANN-dr-filtering-partition
description: Use when working with the filtering-partition module of DiskANN — label-based filtering, PQ/OPQ encoding, and k-means data partitioning for disk index construction
---

# DiskANN Filtering, Partitioning & PQ Encoding Module

## Module Purpose & Capabilities

This module provides the offline data-preparation pipeline that transforms raw vector data into the artifacts needed to build DiskANN disk-based indices. It covers three interconnected subsystems: (1) **label-based filtering** — parsing label metadata, splitting vector datasets by label, and building per-label sub-indices; (2) **k-means partitioning** — sampling training data, running k-means to find cluster pivots, and sharding the full dataset into overlapping partitions so each shard fits in RAM; (3) **Product Quantization (PQ) and Optimized PQ (OPQ)** — training codebooks by per-chunk k-means, encoding vectors into compact byte codes, and providing a runtime lookup table for fast approximate distance computation during search. Together these components enable DiskANN to build billion-scale disk indices that would not fit in memory as a single graph.

## Core Design Logic

### Why Three Subsystems in One Pipeline

DiskANN's disk index construction (`build_disk_index`) requires:
1. PQ-compressed representations for fast approximate distance during beam search (the "PQ distance" used to prune disk reads).
2. Partitioned shards so each sub-graph can be built in RAM independently, then stitched.
3. Optional label-aware splitting so filtered queries only search relevant sub-graphs.

These three concerns share common data access patterns (streaming large bin files in blocks, random sampling for training) and are invoked sequentially by `disk_utils.cpp`, so they live in adjacent source files.

### Key Architectural Decisions

**PQ Chunk Assignment Strategy** — Dimensions are distributed across PQ chunks using a greedy load-balancing heuristic (not simply contiguous slicing). Each dimension goes to the least-loaded chunk that hasn't hit its capacity. This ensures balanced chunk sizes when `dim` is not evenly divisible by `num_pq_chunks`. See `generate_pq_pivots()` in `src/pq.cpp`.

**OPQ via Iterative SVD Rotation** — OPQ alternates between: (a) rotating training data with the current rotation matrix, (b) re-computing PQ pivots via per-chunk k-means, (c) computing a correlation matrix between original and quantized data, (d) SVD to extract a new rotation matrix as $R^T = U V^T$. This runs for `MAX_OPQ_ITERS` (20) iterations. The rotation matrix is persisted alongside the pivot file. See `generate_opq_pivots()` in `src/pq.cpp`.

**Partition with RAM Budget** — `partition_with_ram_budget()` iteratively increases the number of partitions (starting at 3, incrementing by 2) until the estimated RAM for the largest shard fits within the user's budget. The estimation uses `diskann::estimate_ram_usage()`. This makes the system self-tuning for the available hardware.

**Overlapping Shards (k_base)** — Each point is assigned to its `k_base` closest cluster centers, not just one. This creates overlap between shards so that after building sub-graphs independently, the stitched index has better connectivity across partition boundaries.

**Label Filtering with Universal Labels** — Points with a "universal label" are expanded to belong to *every* label. This is a design choice that keeps the query interface simple: universal-label points appear in every per-label sub-index, ensuring they can always bridge connectivity.

**Two I/O Backends for Label File Generation** — On POSIX systems, `generate_label_specific_vector_files()` uses `mmap` + `writev` for zero-copy scatter-gather I/O. A portable fallback `generate_label_specific_vector_files_compat()` uses `std::ifstream`/`std::ofstream` for Windows. Both return `label_id_to_orig_id` mappings.

### Trade-offs

- **Memory vs. Accuracy in PQ Training**: Training set is capped at `MAX_PQ_TRAINING_SET_SIZE` (256K) via random sampling (`gen_random_slice`). Larger samples improve codebook quality but increase training time quadratically.
- **Fixed 8-bit PQ**: `NUM_PQ_BITS=8` → 256 centroids per chunk. This is hardcoded for cache-line-friendly lookup tables (256 × `n_chunks` floats). Supporting fewer bits would require significant refactoring of the lookup table layout.
- **Block Processing (BLOCK_SIZE=5M)**: All streaming operations process 5M points at a time to bound memory. This constant is not configurable at runtime.

## Core Data Structures

### PQ Subsystem

**`FixedChunkPQTable`** — Runtime PQ lookup table used during disk-based search. Defined in `include/pq.h`, implemented in `src/pq.cpp`.
- `tables`: float array `[256 × ndims]` — PQ centroid values in row-major order (centroid_id × dim).
- `tables_tr`: same data transposed to column-major `[ndims × 256]` — optimized for per-dimension iteration in distance computation.
- `chunk_offsets`: uint32 array `[n_chunks + 1]` — maps chunk index to starting dimension.
- `centroid`: float array `[ndims]` — dataset centroid subtracted before PQ encoding.
- `rotmat_tr`: float array `[ndims × ndims]` — OPQ rotation matrix (nullptr if not using OPQ).
- `use_rotation`: bool — set to true if rotation matrix file exists.
- `ndims`, `n_chunks`: dimensions and number of PQ chunks.

**`PQScratch<T>`** — Per-thread scratch space for PQ distance computation during search. Defined in `include/pq_scratch.h`, implemented in `src/scratch.cpp`.
- `aligned_pqtable_dist_scratch`: `[256 × MAX_PQ_CHUNKS]` floats — pre-computed chunk distance tables for current query.
- `aligned_pq_coord_scratch`: `[MAX_PQ_CHUNKS × MAX_DEGREE]` uint8 — PQ codes of neighbor candidates.
- `aligned_dist_scratch`: `[MAX_DEGREE]` floats — accumulated distances.
- `aligned_query_float`, `rotated_query`: `[aligned_dim]` floats — query in float and rotated form.

**PQ Constants** — Defined in `include/pq_common.h`:
- `NUM_PQ_BITS = 8`, `NUM_PQ_CENTROIDS = 256`
- `MAX_OPQ_ITERS = 20`, `NUM_KMEANS_REPS_PQ = 12`
- `MAX_PQ_TRAINING_SET_SIZE = 256000`, `MAX_PQ_CHUNKS = 512`

**File Naming Helpers** in `include/pq_common.h`:
- `get_quantized_vectors_filename(prefix, use_opq, num_chunks)` → e.g. `prefix_pq75_compressed.bin` or `prefix_opq75_compressed.bin`
- `get_pivot_data_filename(prefix, use_opq, num_chunks)` → e.g. `prefix_pq75_pivots.bin`
- `get_rotation_matrix_suffix(pivot_filename)` → appends `_rotation_matrix.bin`

### Filtering Subsystem

**Type Aliases** — Defined in `include/filter_utils.h`:
- `label_set` = `tsl::robin_set<std::string>` — set of string labels for one point.
- `path` = `std::string` — file path alias.
- `parse_label_file_return_values` = `std::tuple<std::vector<label_set>, tsl::robin_map<std::string, uint32_t>, tsl::robin_set<std::string>>` — (point→labels, label→count, all_labels).
- `load_label_index_return_values` = `std::tuple<std::vector<std::vector<uint32_t>>, uint64_t>` — (adjacency lists, file_size).

### Partition Subsystem

No dedicated data structures; operates with float arrays and file I/O. Key intermediate files:
- `{prefix}_centroids.bin`: k-means pivot vectors.
- `{prefix}_subshard-{i}.bin`: data vectors for shard i.
- `{prefix}_subshard-{i}_ids_uint32.bin`: original point IDs for shard i.

### Scratch Space Integration

**`InMemQueryScratch<T>`** and **`SSDQueryScratch<T>`** (defined in `include/scratch.h`, implemented in `src/scratch.cpp`) each embed a `PQScratch<T>*`. `SSDQueryScratch` always allocates it; `InMemQueryScratch` does so only when `init_pq_scratch=true`. This connects PQ distance computation to the search hot path.

## Public API Surface

### PQ Training & Encoding (`include/pq.h`, `src/pq.cpp`)

| Function | Purpose |
|----------|---------|
| `generate_pq_pivots(train_data, num_train, dim, num_centers, num_pq_chunks, max_k_means_reps, pq_pivots_path, make_zero_mean)` | Train PQ codebook from float data, save pivots+centroid+chunk_offsets to a single bin file. |
| `generate_opq_pivots(train_data, num_train, dim, num_centers, num_pq_chunks, opq_pivots_path, make_zero_mean)` | Train OPQ codebook with iterative rotation, saves pivots + rotation matrix. |
| `generate_pq_pivots_simplified(train_data, num_train, dim, num_pq_chunks, pivot_data_vector)` | In-memory simplified PQ training (requires `dim % num_pq_chunks == 0`). No file I/O. |
| `generate_pq_data_from_pivots<T>(data_file, num_centers, num_pq_chunks, pq_pivots_path, pq_compressed_path, use_opq)` | Stream-encode an entire dataset to PQ codes using pre-trained pivots. |
| `generate_pq_data_from_pivots_simplified(data, num, pivot_data, pivots_num, dim, num_pq_chunks, pq)` | In-memory simplified PQ encoding. |
| `generate_disk_quantized_data<T>(data_file, disk_pq_pivots_path, disk_pq_compressed_path, metric, p_val, disk_pq_dims)` | End-to-end: sample → train → encode for disk PQ (used during disk index build). |
| `generate_quantized_data<T>(data_file, pq_pivots_path, pq_compressed_path, metric, p_val, num_pq_chunks, use_opq, codebook_prefix)` | End-to-end: sample → train (PQ or OPQ) → encode. Supports pre-trained codebook via `codebook_prefix`. |

### PQ Runtime (`include/pq.h`, `src/pq.cpp`)

| Function / Method | Purpose |
|-------------------|---------|
| `FixedChunkPQTable::load_pq_centroid_bin(pq_table_file, num_chunks)` | Load PQ tables + centroid + chunk offsets + optional rotation from the pivot file. |
| `FixedChunkPQTable::preprocess_query(query_vec)` | Subtract centroid, apply OPQ rotation if present. |
| `FixedChunkPQTable::populate_chunk_distances(query_vec, dist_vec)` | Fill `[256 × n_chunks]` L2 distance lookup table for a preprocessed query. |
| `FixedChunkPQTable::populate_chunk_inner_products(query_vec, dist_vec)` | Same but for MIPS (returns negative values for min-search compatibility). |
| `FixedChunkPQTable::l2_distance(query_vec, base_vec)` | Compute exact PQ-approximated L2 distance for a single base vector. |
| `FixedChunkPQTable::inner_product(query_vec, base_vec)` | Compute PQ-approximated inner product (negated). |
| `FixedChunkPQTable::inflate_vector(base_vec, out_vec)` | Reconstruct approximate float vector from PQ codes. |
| `aggregate_coords(ids, all_coords, ndims, out)` | Gather PQ codes for a batch of neighbor IDs. |
| `pq_dist_lookup(pq_ids, n_pts, pq_nchunks, pq_dists, dists_out)` | Batch PQ distance accumulation using pre-populated chunk distance tables. |

### Partition (`include/partition.h`, `src/partition.cpp`)

| Function | Purpose |
|----------|---------|
| `gen_random_slice<T>(base_file, output_prefix, sampling_rate)` | Sample vectors from a bin file, write to `{output_prefix}_data.bin` and `{output_prefix}_ids.bin`. |
| `gen_random_slice<T>(data_file, p_val, sampled_data, slice_size, ndims)` | Sample vectors into an in-memory float array. |
| `gen_random_slice<T>(inputdata, npts, ndims, p_val, sampled_data, slice_size)` | Same but from in-memory T* data. |
| `estimate_cluster_sizes(test_data, num_test, pivots, num_centers, dim, k_base, cluster_sizes)` | Estimate shard sizes by assigning test data to k_base closest centers. |
| `shard_data_into_clusters<T>(data_file, pivots, num_centers, dim, k_base, prefix_path)` | Stream full dataset, write shard data + ID map files. Each point goes to k_base shards. |
| `shard_data_into_clusters_only_ids<T>(data_file, pivots, num_centers, dim, k_base, prefix_path)` | Same but only writes ID maps (no data files). Used with `retrieve_shard_data_from_ids`. |
| `retrieve_shard_data_from_ids<T>(data_file, idmap_filename, data_filename)` | Reconstruct shard data file from the original data + ID map. |
| `partition<T>(data_file, sampling_rate, num_centers, max_k_means_reps, prefix_path, k_base)` | End-to-end: sample → k-means → shard with a fixed partition count. |
| `partition_with_ram_budget<T>(data_file, sampling_rate, ram_budget, graph_degree, prefix_path, k_base)` | End-to-end: auto-selects partition count to fit RAM budget, returns `num_parts`. |

### Label Filtering (`include/filter_utils.h`, `src/filter_utils.cpp`)

| Function | Purpose |
|----------|---------|
| `parse_label_file(label_data_path, universal_label)` | Parse comma-separated label file. Returns (point→labels map, label→count, all_labels). Universal-label points get all labels. |
| `parse_formatted_label_file<LabelT>(label_file)` | Parse numeric (uint16/uint32) label file. Returns (point→sorted_labels, all_labels). |
| `generate_label_specific_vector_files<T>(...)` | POSIX only: split data file into per-label files using mmap+writev. Returns label→original_id mapping. |
| `generate_label_specific_vector_files_compat<T>(...)` | Portable version using std streams. Same output. |
| `generate_label_indices<T>(input_data_path, final_index_path_prefix, all_labels, R, L, alpha, num_threads)` | Build a separate Vamana index for each label's data file. |
| `load_label_index(label_index_path, label_number_of_points)` | Load a per-label graph index into adjacency list form. |
| `loadTags(tags_file, base_file)` | Load uint32 tags from a bin file, validate point count matches base file. |

## State Flow

### PQ Training → Encoding → Disk Layout

```
1. gen_random_slice<T>(data_file, p_val) → sampled float training data
       ↓
2a. generate_pq_pivots(train_data, ..., pq_pivots_path)
    - Compute dataset centroid (if make_zero_mean)
    - Distribute dims to chunks via greedy load-balancing
    - For each chunk: k-means++ init → Lloyd's iterations → store pivots
    - Save single file: [offsets | pivots(256×dim) | centroid(dim) | chunk_offsets]
       ↓
2b. OR generate_opq_pivots(train_data, ..., opq_pivots_path)
    - Same as 2a but wraps pivots computation in OPQ loop:
      for MAX_OPQ_ITERS:
        rotate data, recompute PQ, compute correlation, SVD → new rotation
    - Saves pivot file + separate rotation_matrix.bin
       ↓
3. generate_pq_data_from_pivots<T>(data_file, pq_pivots_path, pq_compressed_path)
    - Stream data in BLOCK_SIZE chunks
    - Subtract centroid, optionally apply OPQ rotation
    - For each PQ chunk: find closest centroid → store centroid ID
    - Write [npts | num_chunks | byte_codes...] to compressed file
       ↓
4. FixedChunkPQTable::load_pq_centroid_bin() at search time
    - Loads pivots, centroid, chunk_offsets, optional rotation
    - Builds transposed table (tables_tr) for fast per-dimension access
```

### Partition Workflow

```
1. gen_random_slice<T>(data_file, sampling_rate) → float training sample
       ↓
2. kmeans::kmeanspp_selecting_pivots → initial centroids
   kmeans::run_lloyds → refined centroids
   save_bin(prefix_centroids.bin)
       ↓
3a. shard_data_into_clusters<T> (when data fits reasonably)
    - Stream data, convert to float, compute k_base closest centers
    - Write per-shard data files + ID map files
       ↓
3b. OR shard_data_into_clusters_only_ids<T> (large datasets)
    + retrieve_shard_data_from_ids<T>
    - First pass: only write ID maps
    - Second pass: reconstruct shard data from IDs
       ↓
4. Per-shard: build in-memory Vamana index
5. Stitch shards into final disk layout (done by disk_utils.cpp)
```

### RAM-Budget-Aware Partitioning

```
1. Sample train + test data
2. num_parts = 3
3. Loop:
   a. k-means on train data with num_parts
   b. Estimate cluster sizes from test data
   c. Extrapolate to full dataset (÷ sampling_rate)
   d. max_shard_ram = estimate_ram_usage(largest_shard)
   e. If max_shard_ram > budget: num_parts += 2, repeat
4. Write centroids + shard ID maps
```

### Label Filtering Flow

```
1. parse_label_file(label_path, universal_label)
   → points_to_labels, labels_to_count, all_labels
   (universal-label points expanded to all labels)
       ↓
2. generate_label_specific_vector_files[_compat]<T>(data_path, labels_to_count, points_to_labels, all_labels)
   → per-label data files: {data_path}_{label}
   → label_id_to_orig_id mapping (for ID remapping after index build)
       ↓
3. generate_label_indices<T>(data_path, index_prefix, all_labels, R, L, alpha, threads)
   → per-label Vamana indices: {index_prefix}_{label}
       ↓
4. load_label_index(label_index_path, npts) at merge time
   → adjacency lists for stitching
```

## PQ Pivot File Format

The pivot file is a single binary file with an offset table:

| Section | Offset stored at | Content |
|---------|------------------|---------|
| Offset table | byte 0 | `size_t[4]` — byte offsets to each section |
| Pivots | `offsets[0]` | `float[num_centers × dim]` — PQ centroid vectors |
| Centroid | `offsets[1]` | `float[dim × 1]` — mean vector |
| Chunk offsets | `offsets[2]` | `uint32_t[num_chunks + 1]` — dimension ranges per chunk |

For OPQ, an additional file `{pivot_path}_rotation_matrix.bin` stores `float[dim × dim]`.

The legacy (old) file format has 5 offsets instead of 4, detected by checking `nr == 5` in `load_pq_centroid_bin`.

## Common Modification Scenarios

### Scenario 1: Adding a New Distance Metric to PQ

**Goal**: Support cosine similarity in PQ distance tables.

**Files to modify**:
- `src/pq.cpp` — Add a `populate_chunk_cosine_distances()` method to `FixedChunkPQTable` similar to `populate_chunk_inner_products()`. Cosine requires normalizing the query and centroid vectors per-chunk.
- `include/pq.h` — Declare the new method in the `FixedChunkPQTable` class.
- `src/pq.cpp` in `generate_pq_pivots()` — Consider whether `make_zero_mean` should be false for cosine (it should, same as MIPS).
- `src/pq.cpp` in `generate_quantized_data()` — Add a branch for `Metric::COSINE` alongside the existing `INNER_PRODUCT` branch.

### Scenario 2: Supporting Variable-Bit PQ (e.g., 4-bit)

**Goal**: Use fewer bits per PQ code for higher compression.

**Files to modify**:
- `include/pq_common.h` — Change `NUM_PQ_BITS` from 8 to parameterized, update `NUM_PQ_CENTROIDS`. This is the single source of truth.
- `src/pq.cpp` — `generate_pq_pivots()` and `generate_pq_data_from_pivots()` currently use `num_centers=256` as an argument. Must propagate the new centroid count. The greedy dimension allocation and k-means logic are centroid-count-agnostic, so they work unchanged.
- `src/pq.cpp` — `generate_pq_data_from_pivots()` already branches on `num_centers > 256` to decide uint8 vs uint32 encoding. For 4-bit you'd need nibble packing.
- `include/pq.h` — `FixedChunkPQTable` has hardcoded `256` in its table layout (`tables_tr[j * 256 + i]`). Replace with `NUM_PQ_CENTROIDS`.
- `include/pq_scratch.h` — `aligned_pqtable_dist_scratch` size `[256 × MAX_PQ_CHUNKS]` must use `NUM_PQ_CENTROIDS`.

### Scenario 3: Adding Multi-Label Partitioning

**Goal**: Partition data by label sets rather than just geometric proximity, so per-label queries only read relevant shards.

**Files to modify**:
- `src/filter_utils.cpp` — Extend `parse_label_file()` to return label co-occurrence statistics.
- `src/partition.cpp` — Add a `partition_by_labels<T>()` function that first groups points by label, then runs k-means within each label group. Or: modify `shard_data_into_clusters()` to accept a label-aware weighting function for center assignment.
- `include/partition.h` — Declare the new partition function.
- `src/filter_utils.cpp` — Modify `generate_label_specific_vector_files` to optionally write sub-partition files per label instead of one monolithic file per label.

### Scenario 4: Changing Block Processing Size

**Goal**: Tune memory usage during PQ encoding or partitioning.

**Files to modify**:
- `src/partition.cpp` line 29 and `src/pq.cpp` line 14 — Both define `#define BLOCK_SIZE 5000000`. Change this value or make it a function parameter. The constant controls how many points are held in memory simultaneously during streaming operations. Reducing it lowers peak memory but increases I/O round-trips.

### Scenario 5: Using a Pre-Trained Codebook

**Goal**: Skip PQ training and use an externally provided codebook.

**How it already works**: `generate_quantized_data()` in `src/pq.cpp` checks `file_exists(codebook_prefix)`. If the codebook prefix path exists, it skips training entirely and jumps straight to `generate_pq_data_from_pivots()`. The codebook must follow the standard pivot file format. This is the intended extension point for custom codebooks.

## Template Instantiations

All templated functions are explicitly instantiated for `float`, `uint8_t`, and `int8_t` at the bottom of their respective `.cpp` files. Adding support for a new data type (e.g., `float16`) requires adding explicit instantiations in:
- `src/pq.cpp` — for `generate_pq_data_from_pivots`, `generate_disk_quantized_data`, `generate_quantized_data`
- `src/partition.cpp` — for `gen_random_slice`, `partition`, `partition_with_ram_budget`, `shard_data_into_clusters`, `retrieve_shard_data_from_ids`
- `src/filter_utils.cpp` — for `generate_label_indices`, `generate_label_specific_vector_files_compat`
- `src/scratch.cpp` — for `PQScratch`, `InMemQueryScratch`, `SSDQueryScratch`, `SSDThreadData`
