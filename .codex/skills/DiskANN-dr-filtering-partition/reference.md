# Reference — DiskANN Filtering, Partitioning & PQ Module

## File Index

| File | Location | Role |
|------|----------|------|
| `filter_utils.h` | `include/filter_utils.h` | Label filtering API: type aliases, parse/split/build declarations |
| `filter_utils.cpp` | `src/filter_utils.cpp` | Label file parsing, per-label vector file generation, per-label index building |
| `partition.h` | `include/partition.h` | Partitioning API: random sampling, sharding, partition declarations |
| `partition.cpp` | `src/partition.cpp` | k-means partitioning, shard I/O, RAM-budget-aware auto-partitioning |
| `pq.h` | `include/pq.h` | PQ API: FixedChunkPQTable class, PQ pivot/data generation declarations |
| `pq.cpp` | `src/pq.cpp` | PQ/OPQ training, encoding, runtime distance table, simplified in-memory variants |
| `pq_common.h` | `include/pq_common.h` | PQ constants and file naming helpers |

#### `include/pq_common.h` Constants

| Constant | Value | Purpose |
|----------|-------|--------|
| `NUM_PQ_BITS` | 8 | Bits per PQ code → 256 centroids per chunk |
| `NUM_PQ_CENTROIDS` | 256 (`1 << NUM_PQ_BITS`) | Number of k-means centroids per PQ chunk |
| `MAX_OPQ_ITERS` | 20 | Maximum SVD rotation iterations in OPQ training |
| `NUM_KMEANS_REPS_PQ` | 12 | Lloyd's iterations for PQ/disk-PQ codebook training |
| `MAX_PQ_TRAINING_SET_SIZE` | 256000 | Cap on sampled training vectors for PQ pivot generation |
| `MAX_PQ_CHUNKS` | 512 | Maximum number of PQ chunks allowed |
| `pq_scratch.h` | `include/pq_scratch.h` | PQScratch template class declaration |
| `scratch.cpp` | `src/scratch.cpp` | PQScratch, InMemQueryScratch, SSDQueryScratch implementations |

## Complete Function Reference

### `include/filter_utils.h` / `src/filter_utils.cpp`

#### `diskann::parse_label_file(path label_data_path, std::string universal_label) → parse_label_file_return_values`
Parses a text file where line `i` has comma-separated string labels for point `i`. Points with `universal_label` are expanded to all labels. Returns:
- `std::vector<label_set>`: point_id → set of labels
- `tsl::robin_map<std::string, uint32_t>`: label → count of points
- `tsl::robin_set<std::string>`: all distinct labels

**Internal detail — `points_with_universal_label` vector**: During the per-line parse loop, a local `std::vector<uint32_t> points_with_universal_label` collects the point ID of every point whose labels include the universal label. These points are *not* added to `all_labels` or `labels_to_number_of_points` during the loop. After the loop, a post-loop expansion iterates over `points_with_universal_label` and sets each point's label set to the full `all_labels` set, incrementing `labels_to_number_of_points` for every label. This two-phase design ensures `all_labels` is fully populated before expanding universal-label points.

**Edge cases**: Points with no labels and no universal label cause `exit(-1)`. The universal label itself is excluded from `all_labels`.

#### `diskann::parse_formatted_label_file<LabelT>(path label_file) → tuple<vector<vector<LabelT>>, robin_set<LabelT>>`
Parses a file with numeric labels (uint16 or uint32). Labels on each line are comma-separated, optionally followed by a tab and additional data (only the pre-tab portion is parsed). Labels per point are sorted. Returns (points→sorted_labels, all_labels).

#### `diskann::generate_label_specific_vector_files<T>(...) → robin_map<string, vector<uint32_t>>`
**POSIX only** (guarded by `#ifndef _WINDOWS`). Uses `mmap` to read input data, creates `iovec` arrays per label, writes per-label files via `writev`. Each output file: `{input_data_path}_{label}` in standard bin format (npts, dim, vectors...). Returns map from label to vector of original point IDs.

**Parameters**: `input_data_path`, `labels_to_number_of_points`, `point_ids_to_labels`, `all_labels`.

Handles `IOV_MAX` limit by chunking `writev` calls.

#### `diskann::generate_label_specific_vector_files_compat<T>(...) → robin_map<string, vector<uint32_t>>`
Portable (Windows-compatible) version. Allocates per-label buffers, copies vectors via `memcpy`, writes via `std::ofstream`. Same interface and return type.

#### `diskann::generate_label_indices<T>(path input_data_path, path final_index_path_prefix, label_set all_labels, uint32_t R, uint32_t L, float alpha, uint32_t num_threads)`
For each label, builds a Vamana `Index<T>` with L2 metric on the label-specific data file `{input_data_path}_{label}`, saves to `{final_index_path_prefix}_{label}`. Suppresses stdout during building, prints total time at end.

#### `diskann::load_label_index(path label_index_path, uint32_t label_number_of_points) → load_label_index_return_values`
Reads a DiskANN graph index file: header (file_size, max_degree, entry_point, num_frozen), then adjacency lists. Returns `(vector<vector<uint32_t>>, uint64_t file_size)`.

#### `diskann::loadTags(const string &tags_file, const string &base_file) → vector<uint32_t>`
Loads uint32 tag data from a bin file. Validates dimensionality==1 and that point count matches the base file. Returns vector of tags indexed by location.

---

### `include/partition.h` / `src/partition.cpp`

#### `gen_random_slice<T>(const string base_file, const string output_prefix, double sampling_rate)`
Streams `base_file`, samples each vector with probability `sampling_rate`, writes sampled vectors to `{output_prefix}_data.bin` and their original IDs to `{output_prefix}_ids.bin`. Both files use standard bin format (npts, ndims/1, data...). The point count header is written last (seek back to offset 0).

#### `gen_random_slice<T>(const string data_file, double p_val, float *&sampled_data, size_t &slice_size, size_t &ndims)`
Streams `data_file`, samples vectors, converts T→float, returns heap-allocated `sampled_data` array. Caller owns the memory. Sets `slice_size` and `ndims`.

#### `gen_random_slice<T>(const T *inputdata, size_t npts, size_t ndims, double p_val, float *&sampled_data, size_t &slice_size)`
Same as above but from an in-memory array instead of file.

#### `estimate_cluster_sizes(float *test_data, size_t num_test, float *pivots, size_t num_centers, size_t dim, size_t k_base, vector<size_t> &cluster_sizes)`
Assigns each test point to its `k_base` closest pivot centers (via `math_utils::compute_closest_centers`), counts assignments per cluster. Processes in blocks of `BLOCK_SIZE`. Populates `cluster_sizes` output vector.

#### `shard_data_into_clusters<T>(const string data_file, float *pivots, size_t num_centers, size_t dim, size_t k_base, string prefix_path) → int`
Streams the full dataset, assigns each point to `k_base` closest centers. Writes:
- `{prefix_path}_subshard-{i}.bin` — shard data (original T type)
- `{prefix_path}_subshard-{i}_ids_uint32.bin` — original point IDs

Each shard file has header (npts, dim) written last (seeks back). Returns 0 on success.

#### `shard_data_into_clusters_only_ids<T>(...) → int`
Same as above but only writes the ID map files, not the data files. Used for very large datasets where we want to avoid writing redundant data. The data is later retrieved by `retrieve_shard_data_from_ids`.

#### `retrieve_shard_data_from_ids<T>(const string data_file, string idmap_filename, string data_filename) → int`
Reads ID map, streams original data file, extracts matching points, writes shard data file. Assumes IDs in the idmap are sorted (sequential scan with `cur_pos` pointer).

#### `partition<T>(const string data_file, float sampling_rate, size_t num_parts, size_t max_k_means_reps, const string prefix_path, size_t k_base) → int`
Full pipeline:
1. `gen_random_slice<T>()` → training data
2. `kmeanspp_selecting_pivots()` + `run_lloyds()` → pivots
3. `save_bin(prefix_centroids.bin)` → persist pivots
4. `shard_data_into_clusters<T>()` → shard files

#### `partition_with_ram_budget<T>(const string data_file, double sampling_rate, double ram_budget, size_t graph_degree, const string prefix_path, size_t k_base) → int`
Auto-tuning version. Starts with `num_parts=3`, loops:
1. k-means on training sample
2. Estimate cluster sizes on separate test sample
3. Scale by `1/sampling_rate` for full dataset estimate
4. `estimate_ram_usage(largest_shard_size, dim, sizeof(T), graph_degree)` → max RAM
5. If exceeds budget: `num_parts += 2`, retry

Uses `shard_data_into_clusters_only_ids` (not full data) to minimize I/O. Returns final `num_parts`.

---

### `include/pq.h` / `src/pq.cpp`

#### `FixedChunkPQTable` Class

**Constructor/Destructor**: Default constructor zeroes pointers. Destructor `delete[]`s all allocated arrays (except in `EXEC_ENV_OLS` mode).

**`load_pq_centroid_bin(const char *pq_table_file, size_t num_chunks)`**
Loads the pivot file:
1. Read offset table (`size_t[4]` or `size_t[5]` for legacy format)
2. Load pivots `float[256 × ndims]` → `tables`
3. Load centroid `float[ndims]` → `centroid`
4. Load chunk offsets `uint32_t[n_chunks+1]` → `chunk_offsets`
5. If rotation matrix file exists, load `float[ndims × ndims]` → `rotmat_tr`, set `use_rotation=true`
6. Build transposed table `tables_tr[j * 256 + i] = tables[i * ndims + j]`

**`preprocess_query(float *query_vec)`**
Subtracts centroid from query. If `use_rotation`, applies rotation: `query_out = query × rotmat_tr`.

**`populate_chunk_distances(const float *query_vec, float *dist_vec)`**
For each chunk, for each of 256 centroids, accumulates squared differences across the chunk's dimensions. Output: `dist_vec[chunk * 256 + centroid_id]` = L2 distance for that chunk.

**`populate_chunk_inner_products(const float *query_vec, float *dist_vec)`**
Same structure but accumulates negative dot products (to convert max-IP to min-distance). For each chunk, for each dimension `j`, for each centroid `idx` in 0..255: `chunk_dists[idx] -= (float)(centers_dim_vec[idx] * query_vec[j])`. The negation converts inner product maximization into distance minimization so the search code's min-heap logic works unchanged.

**`l2_distance(const float *query_vec, uint8_t *base_vec) → float`**
Point-to-point PQ-approximated L2: sums squared differences for the centroid assigned to each chunk.

**`inner_product(const float *query_vec, uint8_t *base_vec) → float`**
Returns negative dot product (for compatibility with min-distance search). Accumulation: for each chunk, for each dimension `j` in the chunk, `res += tables_tr[256*j + base_vec[chunk]] * query_vec[j]` (i.e., `centers_dim_vec[base_vec[chunk]] * query_vec[j]`). Returns `-res`.

**`inflate_vector(uint8_t *base_vec, float *out_vec)`**
Reconstructs approximate float vector by looking up centroid values per chunk and adding back the centroid offset.

**`get_num_chunks() → uint32_t`**
Returns `n_chunks`.

#### Free Functions

**`aggregate_coords(ids, all_coords, ndims, out)`** — Two overloads (vector and raw pointer). Copies PQ codes for given IDs from a flat PQ code array into a contiguous output buffer.

**`pq_dist_lookup(pq_ids, n_pts, pq_nchunks, pq_dists, dists_out)`** — Two overloads (vector and raw pointer). Accumulates pre-computed chunk distances for a batch of points using their PQ codes. Uses `_mm_prefetch` for cache optimization.

#### PQ Training Functions

**`generate_pq_pivots(const float *train_data, size_t num_train, uint32_t dim, uint32_t num_centers, uint32_t num_pq_chunks, uint32_t max_k_means_reps, string pq_pivots_path, bool make_zero_mean) → int`**

Dimension allocation heuristic:
1. Compute `low_val = floor(dim / num_pq_chunks)`, `high_val = ceil(dim / num_pq_chunks)`
2. `max_num_high = dim - low_val * num_pq_chunks` chunks get `high_val` dims, rest get `low_val`
3. Greedy assignment: each dim goes to the least-loaded chunk under capacity

Per-chunk k-means:
1. Extract sub-dimensions from training data
2. `kmeanspp_selecting_pivots` → initial centers
3. `run_lloyds` → refined centers
4. Copy back to full pivot array at correct offset

File format: `[size_t[4] offsets | float[256×dim] pivots | float[dim] centroid | uint32_t[chunks+1] chunk_offsets]`

Returns -1 if pivot file already exists with matching dimensions, 0 on success.

**`generate_opq_pivots(const float *train_data, size_t num_train, uint32_t dim, uint32_t num_centers, uint32_t num_pq_chunks, string opq_pivots_path, bool make_zero_mean) → int`**

Same dimension allocation as PQ. Adds:
1. Initialize rotation matrix to identity
2. For `MAX_OPQ_ITERS=20` rounds:
   a. `cblas_sgemm`: rotate training data
   b. Per-chunk k-means (using previous pivots as warm start after round 0)
   c. Reconstruct quantized data by replacing each point's chunk with its assigned centroid
   d. `cblas_sgemm`: correlation matrix = `train_data^T × quantized_data`
   e. `LAPACKE_sgesdd`: SVD of correlation matrix → U, S, V^T
   f. New rotation = `U × V^T`
3. Save pivot file + rotation matrix file

**`generate_pq_pivots_simplified(const float *train_data, size_t num_train, size_t dim, size_t num_pq_chunks, vector<float> &pivot_data_vector) → int`**

Simplified in-memory version. Requirements: `dim % num_pq_chunks == 0`. Fixed `num_centers=256`, `KMEANS_ITERS=15`, no zero-mean. No file I/O. No OMP pragmas (for controlled resource allocation).

**`generate_pq_data_from_pivots<T>(data_file, num_centers, num_pq_chunks, pq_pivots_path, pq_compressed_path, use_opq) → int`**

Streams data in blocks:
1. Load pivot file (pivots, centroid, chunk_offsets, optional rotation)
2. For each block: read T data → convert to float → subtract centroid → optionally rotate
3. For each chunk: find closest centroid → store ID
4. If `num_centers ≤ 256`: compress IDs to uint8. Otherwise: write as uint32.
5. Output format: `[uint32 npts | uint32 num_chunks | byte_codes...]`

**`generate_pq_data_from_pivots_simplified(data, num, pivot_data, pivots_num, dim, num_pq_chunks, pq) → int`**

In-memory simplified version. Requirements: `dim % num_pq_chunks == 0`, `pivots_num == 256 * dim`. Output stored in `vector<uint8_t> pq`.

**`generate_disk_quantized_data<T>(data_file, disk_pq_pivots_path, disk_pq_compressed_path, metric, p_val, disk_pq_dims)`**

Convenience: sample → `generate_pq_pivots` (always PQ, not OPQ) → `generate_pq_data_from_pivots`. Uses `NUM_KMEANS_REPS_PQ=12`. For MIPS, always uses float data type for encoding.

**`generate_quantized_data<T>(data_file, pq_pivots_path, pq_compressed_path, metric, p_val, num_pq_chunks, use_opq, codebook_prefix)`**

Full generality wrapper:
- If `codebook_prefix` path doesn't exist: sample, train (PQ or OPQ), then encode.
- If `codebook_prefix` exists: skip training, just encode.
- `make_zero_mean = true` for L2 distance, false for MIPS and OPQ.

---

### `include/pq_common.h`

#### Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `NUM_PQ_BITS` | 8 | Bits per PQ code |
| `NUM_PQ_CENTROIDS` | 256 (= 1 << 8) | Centroids per chunk |
| `MAX_OPQ_ITERS` | 20 | OPQ rotation refinement iterations |
| `NUM_KMEANS_REPS_PQ` | 12 | Lloyd's iterations for PQ training |
| `MAX_PQ_TRAINING_SET_SIZE` | 256000 | Max training set size |
| `MAX_PQ_CHUNKS` | 512 | Max number of PQ chunks (scratch size bound) |

#### Helpers

- `get_quantized_vectors_filename(prefix, use_opq, num_chunks)` — Returns `{prefix}pq{N}_compressed.bin` or `{prefix}_opq{N}_compressed.bin`.
- `get_pivot_data_filename(prefix, use_opq, num_chunks)` — Returns `{prefix}pq{N}_pivots.bin` or `{prefix}_opq{N}_pivots.bin`.
- `get_rotation_matrix_suffix(pivot_filename)` — Returns `{pivot_filename}_rotation_matrix.bin`.

---

### `include/pq_scratch.h` / `src/scratch.cpp`

#### `PQScratch<T>`

**Constructor** `PQScratch(size_t graph_degree, size_t aligned_dim)`:
- Allocates aligned buffers: pq_coord_scratch `[degree × MAX_PQ_CHUNKS]`, pqtable_dist_scratch `[256 × MAX_PQ_CHUNKS]`, dist_scratch `[degree]`, query_float `[aligned_dim]`, rotated_query `[aligned_dim]`.
- All allocations use `diskann::alloc_aligned` with 256-byte alignment.

**`initialize(size_t dim, const T *query, float norm)`**:
Converts query to float (optionally dividing by norm), stores in both `aligned_query_float` and `rotated_query`.

**Destructor**: Frees all aligned buffers via `diskann::aligned_free`.

#### `InMemQueryScratch<T>`

Embeds `PQScratch<T>*` (allocated when `init_pq_scratch=true`). Also owns aligned query buffer, candidate pool, priority queue, visited sets. See `include/scratch.h` for field details.

#### `SSDQueryScratch<T>`

Always allocates a `PQScratch<T>`. Also owns `coord_scratch` (for re-ranking with full coordinates), `sector_scratch` (for raw disk sector reads), visited set, and result sets.

#### `SSDThreadData<T>`

Wrapper around `SSDQueryScratch<T>` + `IOContext`. One instance per search thread.

## Dependency Graph

```
filter_utils.h → cached_io.h, common_includes.h, memory_mapper.h, utils.h, tsl/robin_map.h, tsl/robin_set.h
filter_utils.cpp → filter_utils.h, index.h, parameters.h
partition.h → neighbor.h, parameters.h, tsl/robin_set.h, utils.h
partition.cpp → utils.h, math_utils.h, index.h, parameters.h, memory_mapper.h, partition.h
pq.h → utils.h, pq_common.h
pq.cpp → mkl.h, pq.h, partition.h, math_utils.h, tsl/robin_map.h
pq_common.h → (standalone, no deps)
pq_scratch.h → pq_common.h, utils.h
scratch.cpp → scratch.h, pq_scratch.h
```

Key external dependencies:
- **Intel MKL** — `cblas_sgemm` for matrix multiply (OPQ rotation), `LAPACKE_sgesdd` for SVD.
- **math_utils.h** — `compute_closest_centers()` used in both partitioning and PQ encoding.
- **kmeans (math_utils)** — `kmeanspp_selecting_pivots()`, `run_lloyds()` — the actual k-means implementation lives in `math_utils.cpp`.
- **tsl::robin_map/set** — Hash maps used throughout for label→data mappings.
- **POSIX I/O** — `mmap`, `writev`, `iovec` used in label file generation (non-Windows).
