# DiskANN Disk-Index Module — Reference

## File Index

| File | Role |
|------|------|
| `include/pq_flash_index.h` | PQFlashIndex class declaration — the main search-time index class |
| `src/pq_flash_index.cpp` | PQFlashIndex implementation — load, search, cache, range_search |
| `include/disk_utils.h` | Build-time utilities — function declarations for index construction |
| `src/disk_utils.cpp` | Build-time utilities — implementations for disk layout, merging, building |
| `include/cached_io.h` | Sequential cached read/write streams (cached_ifstream, cached_ofstream) |
| `include/aligned_file_reader.h` | Abstract aligned file reader interface + AlignedRead/IOContext structs |
| `include/linux_aligned_file_reader.h` | Linux AIO-based reader declaration |
| `src/linux_aligned_file_reader.cpp` | Linux AIO-based reader implementation |
| `include/windows_aligned_file_reader.h` | Windows IOCP-based reader declaration |
| `src/windows_aligned_file_reader.cpp` | Windows IOCP-based reader implementation |
| `include/memory_mapper.h` | Memory-mapped file wrapper declaration |
| `src/memory_mapper.cpp` | Memory-mapped file wrapper implementation (mmap/MapViewOfFile) |

---

## Core Data Structures

### `PQFlashIndex<T, LabelT>` — `include/pq_flash_index.h`

The main disk-based index class. Template parameters:
- `T`: vector element type (`float`, `int8_t`, `uint8_t`)
- `LabelT`: label/filter type (`uint32_t`, `uint16_t`)

#### Key Fields

| Field | Type | Purpose |
|-------|------|---------|
| `reader` | `shared_ptr<AlignedFileReader>&` | Handle to the disk file reader |
| `metric` | `diskann::Metric` | Distance metric (L2, IP, Cosine) |
| `data` | `uint8_t*` | PQ-compressed vectors in RAM (num_points × n_chunks bytes) |
| `_n_chunks` | `uint64_t` | Number of PQ chunks per vector |
| `_pq_table` | `FixedChunkPQTable` | PQ centroid lookup table for distance computation |
| `_num_points` | `uint64_t` | Total number of points in the index |
| `_data_dim` | `uint64_t` | Original data dimensionality |
| `_aligned_dim` | `uint64_t` | Dimension rounded up to multiple of 8 |
| `_disk_bytes_per_point` | `uint64_t` | Bytes per point on disk (sizeof(T)*dim or PQ bytes) |
| `_max_node_len` | `uint64_t` | Max bytes per node (coords + degree + neighbor IDs) |
| `_nnodes_per_sector` | `uint64_t` | Nodes per 4096-byte sector (0 if node > sector) |
| `_max_degree` | `uint64_t` | Maximum out-degree of graph |
| `_medoids` | `uint32_t*` | Entry point node ID(s) |
| `_num_medoids` | `size_t` | Number of medoids (default 1) |
| `_centroid_data` | `float*` | Full-precision vectors of medoid nodes (for closest-medoid selection) |
| `_nhood_cache_buf` | `unsigned*` | Backing buffer for neighborhood cache |
| `_nhood_cache` | `tsl::robin_map<uint32_t, pair<uint32_t, uint32_t*>>` | node_id → (nnbrs, nbr_ids) |
| `_coord_cache_buf` | `T*` | Backing buffer for coordinate cache |
| `_coord_cache` | `tsl::robin_map<uint32_t, T*>` | node_id → coordinate pointer |
| `_thread_data` | `ConcurrentQueue<SSDThreadData<T>*>` | Per-thread scratch + IOContext pool |
| `_use_disk_index_pq` | `bool` | Whether disk stores PQ-compressed data instead of full vectors |
| `_disk_pq_table` | `FixedChunkPQTable` | PQ table for disk-compressed vectors |
| `_disk_pq_n_chunks` | `uint64_t` | Number of PQ chunks if disk uses PQ |
| `_reorder_data_exists` | `bool` | Whether full-precision reorder data exists at end of file |
| `_reorder_data_start_sector` | `uint64_t` | First sector of the reorder data region |
| `_nvecs_per_sector` | `uint64_t` | Vectors per sector in the reorder region |
| `_max_base_norm` | `float` | Max norm for MIPS rescaling |
| `_num_frozen_points` | `uint64_t` | Number of frozen points (for streaming DiskANN) |
| `_frozen_location` | `uint64_t` | ID of the frozen point |
| `_dist_cmp` | `shared_ptr<Distance<T>>` | Distance function for type T |
| `_dist_cmp_float` | `shared_ptr<Distance<float>>` | Distance function for float (medoid selection) |

#### Filter-Related Fields

| Field | Type | Purpose |
|-------|------|---------|
| `_pts_to_label_offsets` | `uint32_t*` | Per-point start offset into `_pts_to_labels` |
| `_pts_to_label_counts` | `uint32_t*` | Per-point label count |
| `_pts_to_labels` | `LabelT*` | Flat array of all point-label assignments |
| `_filter_to_medoid_ids` | `unordered_map<LabelT, vector<uint32_t>>` | Label → medoid IDs |
| `_use_universal_label` | `bool` | Whether a universal label exists |
| `_universal_filter_label` | `LabelT` | The universal label value |
| `_dummy_pts` | `tsl::robin_set<uint32_t>` | Set of dummy point IDs |
| `_dummy_to_real_map` | `tsl::robin_map<uint32_t, uint32_t>` | dummy→real ID mapping |
| `_real_to_dummy_map` | `tsl::robin_map<uint32_t, vector<uint32_t>>` | real→dummy IDs |
| `_label_map` | `unordered_map<string, LabelT>` | String label → numeric label |

---

### `SSDQueryScratch<T>` — `include/scratch.h`

Per-thread scratch space for SSD search operations.

| Field | Type | Purpose |
|-------|------|---------|
| `coord_scratch` | `T*` | Buffer for copying a node's coordinates (aligned_dim sized) |
| `sector_scratch` | `char*` | Buffer for disk sector reads (MAX_N_SECTOR_READS × SECTOR_LEN) |
| `sector_idx` | `size_t` | Next available sector scratch slot |
| `visited` | `tsl::robin_set<size_t>` | Set of visited node IDs |
| `retset` | `NeighborPriorityQueue` | Priority queue of candidates (size l_search) |
| `full_retset` | `vector<Neighbor>` | All expanded nodes with exact distances |

Inherited from `AbstractScratch<T>`:
- `_aligned_query_T` — aligned query in type T
- `_pq_scratch` — PQScratch instance (contains `aligned_query_float`, `rotated_query`, `aligned_pqtable_dist_scratch`, `aligned_dist_scratch`, `aligned_pq_coord_scratch`)

---

### `SSDThreadData<T>` — `include/scratch.h`

Bundles scratch space + IO context for a thread.

| Field | Type | Purpose |
|-------|------|---------|
| `scratch` | `SSDQueryScratch<T>` | Query scratch buffers |
| `ctx` | `IOContext` | Thread-specific async IO context |

---

### `AlignedRead` — `include/aligned_file_reader.h`

Represents a single aligned disk read request.

| Field | Type | Constraint |
|-------|------|------------|
| `offset` | `uint64_t` | Must be 512-aligned |
| `len` | `uint64_t` | Must be 512-aligned |
| `buf` | `void*` | Must be 512-aligned |

---

### `AlignedFileReader` (abstract) — `include/aligned_file_reader.h`

Abstract interface for sector-aligned disk I/O.

| Method | Description |
|--------|-------------|
| `open(fname)` | Open the disk index file |
| `close()` | Close the file |
| `register_thread()` | Create an IO context for the calling thread |
| `deregister_thread()` | Destroy the calling thread's IO context |
| `deregister_all_threads()` | Destroy all IO contexts |
| `get_ctx()` | Get the calling thread's IOContext |
| `read(read_reqs, ctx, async)` | Execute a batch of aligned reads |

Protected state:
- `ctx_map`: `tsl::robin_map<thread::id, IOContext>` — maps threads to their IO contexts
- `ctx_mut`: `std::mutex` — protects ctx_map

---

### `IOContext` — `include/aligned_file_reader.h`

Platform-specific I/O context:
- **Linux**: `io_context_t` (kernel AIO context handle)
- **Windows (non-Bing)**: `HANDLE fhandle`, `HANDLE iocp`, `vector<OVERLAPPED> reqs`
- **Windows (Bing)**: `shared_ptr<IDiskPriorityIO>`, request/status vectors, `volatile long m_completeCount`

---

### `LinuxAlignedFileReader` — `include/linux_aligned_file_reader.h`

Linux implementation using kernel AIO (`libaio`).

| Field | Type | Purpose |
|-------|------|---------|
| `file_sz` | `uint64_t` | File size |
| `file_desc` | `int` (FileHandle) | File descriptor opened with O_DIRECT|O_RDONLY|O_LARGEFILE |
| `bad_ctx` | `io_context_t` | Sentinel value `(io_context_t)-1` |

Key implementation detail: `execute_io()` in `src/linux_aligned_file_reader.cpp` breaks requests into batches of MAX_EVENTS (1024), issues `io_prep_pread` + `io_submit`, then blocks on `io_getevents`.

---

### `WindowsAlignedFileReader` — `include/windows_aligned_file_reader.h`

Windows implementation using I/O Completion Ports.

Each registered thread gets: a file handle opened with `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS`, an IOCP, and MAX_IO_DEPTH (128) OVERLAPPED structures. Reads are issued via `ReadFile` (async) and completed via `GetQueuedCompletionStatus`.

---

### `MemoryMapper` — `include/memory_mapper.h`

Simple cross-platform memory-mapped file wrapper.

| Field | Type (Linux) | Type (Windows) | Purpose |
|-------|------|------|---------|
| `_fd` | `int` | `HANDLE` (file mapping) | File handle |
| `_bareFile` | — | `HANDLE` | Raw file handle (Windows) |
| `_buf` | `char*` | `char*` | Mapped buffer pointer |
| `_fileSize` | `size_t` | `size_t` | File size |

Uses `mmap(PROT_READ, MAP_PRIVATE)` on Linux, `CreateFileMapping + MapViewOfFile` on Windows.

---

### `cached_ifstream` — `include/cached_io.h`

Sequential cached input stream for large file reads during index build.

| Field | Type | Purpose |
|-------|------|---------|
| `reader` | `std::ifstream` | Underlying file stream |
| `cache_size` | `uint64_t` | Size of the read-ahead cache buffer |
| `cache_buf` | `char*` | The cache buffer |
| `cur_off` | `uint64_t` | Current read offset within cache_buf |
| `fsize` | `uint64_t` | Total file size |

Reads satisfy from cache first; if request exceeds cache, reads directly from file and refills cache.

---

### `cached_ofstream` — `include/cached_io.h`

Sequential cached output stream for large file writes during index build.

| Field | Type | Purpose |
|-------|------|---------|
| `writer` | `std::ofstream` | Underlying file stream |
| `cache_size` | `uint64_t` | Write buffer size |
| `cache_buf` | `char*` | Write buffer |
| `cur_off` | `uint64_t` | Current position in write buffer |
| `fsize` | `uint64_t` | Total bytes written |

Accumulates writes in buffer; flushes to disk when buffer is full or on explicit `flush_cache()`/`close()`.

---

## Complete Public API

### PQFlashIndex<T, LabelT> Public Methods

```cpp
// Construction
PQFlashIndex(shared_ptr<AlignedFileReader> &fileReader, Metric metric = Metric::L2);
~PQFlashIndex();

// Loading
int load(uint32_t num_threads, const char *index_prefix);
int load_from_separate_paths(uint32_t num_threads, const char *index_filepath,
                             const char *pivots_filepath, const char *compressed_filepath);

// Cache management
void load_cache_list(vector<uint32_t> &node_list);
void generate_cache_list_from_sample_queries(string sample_bin, uint64_t l_search,
    uint64_t beamwidth, uint64_t num_nodes_to_cache, uint32_t num_threads,
    vector<uint32_t> &node_list);
void cache_bfs_levels(uint64_t num_nodes_to_cache, vector<uint32_t> &node_list,
                      bool shuffle = false);

// Search (4 overloads)
void cached_beam_search(const T *query, uint64_t k_search, uint64_t l_search,
    uint64_t *res_ids, float *res_dists, uint64_t beam_width,
    bool use_reorder_data = false, QueryStats *stats = nullptr);

void cached_beam_search(const T *query, uint64_t k_search, uint64_t l_search,
    uint64_t *res_ids, float *res_dists, uint64_t beam_width,
    bool use_filter, const LabelT &filter_label,
    bool use_reorder_data = false, QueryStats *stats = nullptr);

void cached_beam_search(const T *query, uint64_t k_search, uint64_t l_search,
    uint64_t *res_ids, float *res_dists, uint64_t beam_width,
    uint32_t io_limit, bool use_reorder_data = false, QueryStats *stats = nullptr);

void cached_beam_search(const T *query, uint64_t k_search, uint64_t l_search,
    uint64_t *res_ids, float *res_dists, uint64_t beam_width,
    bool use_filter, const LabelT &filter_label,
    uint32_t io_limit, bool use_reorder_data = false, QueryStats *stats = nullptr);

// Range search
uint32_t range_search(const T *query, double range, uint64_t min_l_search,
    uint64_t max_l_search, vector<uint64_t> &indices, vector<float> &distances,
    uint64_t min_beam_width, QueryStats *stats = nullptr);

// Node reading (for external tools)
vector<bool> read_nodes(const vector<uint32_t> &node_ids,
    vector<T*> &coord_buffers,
    vector<pair<uint32_t, uint32_t*>> &nbr_buffers);

// Accessors
LabelT get_converted_label(const string &filter_label);
uint64_t get_data_dim();
Metric get_metric();
vector<uint8_t> get_pq_vector(uint64_t vid);
uint64_t get_num_points();
uint64_t get_max_degree();
```

### Disk Utilities — Free Functions in `diskann` namespace

```cpp
// Full disk index build orchestrator
template <typename T, typename LabelT>
int build_disk_index(const char *dataFilePath, const char *indexFilePath,
    const char *indexBuildParameters, Metric compareMetric,
    bool use_opq = false, const string &codebook_prefix = "",
    bool use_filters = false, const string &label_file = "",
    const string &universal_label = "", uint32_t filter_threshold = 0,
    uint32_t Lf = 0);

// Build merged Vamana index (partition + per-shard build + merge)
template <typename T, typename LabelT>
int build_merged_vamana_index(string base_file, Metric compareMetric,
    uint32_t L, uint32_t R, double sampling_rate, double ram_budget,
    string mem_index_path, string medoids_file, string centroids_file,
    size_t build_pq_bytes, bool use_opq, uint32_t num_threads,
    bool use_filters = false, const string &label_file = "",
    const string &labels_to_medoids_file = "",
    const string &universal_label = "", uint32_t Lf = 0);

// Create sector-aligned disk layout from in-memory index
template <typename T>
void create_disk_layout(const string base_file, const string mem_index_file,
    const string output_file, const string reorder_data_file = "");

// Merge per-shard Vamana graphs
int merge_shards(const string &vamana_prefix, const string &vamana_suffix,
    const string &idmaps_prefix, const string &idmaps_suffix,
    uint64_t nshards, uint32_t max_degree, const string &output_vamana,
    const string &medoids_file, bool use_filters = false,
    const string &labels_to_medoids_file = "");

// Optimize beam width for QPS/latency
template <typename T, typename LabelT>
uint32_t optimize_beamwidth(unique_ptr<PQFlashIndex<T,LabelT>> &pFlashIndex,
    T *tuning_sample, uint64_t tuning_sample_num,
    uint64_t tuning_sample_aligned_dim, uint32_t L,
    uint32_t nthreads, uint32_t start_bw = 2);

// Load warmup vectors for cache warming
template <typename T>
T *load_warmup(const string &cache_warmup_file, uint64_t &warmup_num,
    uint64_t warmup_dim, uint64_t warmup_aligned_dim);

// Utility
double get_memory_budget(const string &mem_budget_str);
double get_memory_budget(double search_ram_budget_in_gb);
void add_new_file_to_single_index(string index_file, string new_file);
size_t calculate_num_pq_chunks(double final_index_ram_limit, size_t points_num, uint32_t dim);
void read_idmap(const string &fname, vector<uint32_t> &ivecs);
void extract_shard_labels(const string &in_label_file, const string &shard_ids_bin,
                          const string &shard_label_file);
template <typename T>
string preprocess_base_file(const string &infile, const string &indexPrefix, Metric &distMetric);
```

---

## Disk Index File Format

### Sector 0 — Metadata

Stored as a binary vector of `uint64_t` values (with a 2×uint32_t prefix for bin format compatibility):

| Offset (after 8-byte header) | Field | Description |
|------|-------|-------------|
| 0 | `npts` | Number of points |
| 1 | `ndims` | Number of dimensions (may be PQ dims if disk PQ) |
| 2 | `medoid` | Entry point node ID |
| 3 | `max_node_len` | Max bytes per node: (R+1)*4 + ndims*sizeof(T) |
| 4 | `nnodes_per_sector` | Nodes per sector (0 if node spans multiple sectors) |
| 5 | `vamana_frozen_num` | Number of frozen points (0 or 1) |
| 6 | `vamana_frozen_loc` | Frozen point location |
| 7 | `append_reorder_data` | 1 if reorder data appended, 0 otherwise |
| 8* | `reorder_data_start_sector` | First sector of reorder data (if applicable) |
| 9* | `ndims_reorder_vecs` | Dimensions of reorder vectors |
| 10* | `n_data_nodes_per_sector` | Reorder vectors per sector |
| last | `disk_index_file_size` | Total file size |

### Sectors 1..N — Graph Data

Each node is stored as: `[coords: T × ndims][nnbrs: uint32_t][nbr_ids: uint32_t × nnbrs]`

- If `nnodes_per_sector > 0`: multiple nodes packed sequentially within each sector
- If `nnodes_per_sector == 0`: each node occupies `ceil(max_node_len / SECTOR_LEN)` sectors

### Reorder Data Region (Optional)

After graph sectors, full-precision float vectors are packed into sectors for re-ranking top results.

---

## Associated File Naming Convention

Given index prefix `P`:
- `P_pq_pivots.bin` — PQ centroid table
- `P_pq_compressed.bin` — PQ-compressed vectors (loaded into RAM)
- `P_disk.index` — Sector-aligned disk index file
- `P_disk.index_medoids.bin` — Medoid IDs
- `P_disk.index_centroids.bin` — Medoid centroid vectors
- `P_disk.index_pq_pivots.bin` — Disk PQ table (if disk PQ enabled)
- `P_disk.index_labels.txt` — Per-point labels
- `P_disk.index_labels_map.txt` — String→numeric label mapping
- `P_disk.index_labels_to_medoids.txt` — Label→medoid mapping
- `P_disk.index_universal_label.txt` — Universal label value
- `P_disk.index_dummy_map.txt` — Dummy→real point ID mapping
- `P_disk.index_max_base_norm.bin` — Max base norm (for MIPS)
- `P_sample_data.bin` / `P_sample_ids.bin` — Sample data for warmup

---

## Key Constants

| Constant | Value | Location | Purpose |
|----------|-------|----------|---------|
| `SECTOR_LEN` | 4096 | `include/defaults.h` | Disk sector size |
| `MAX_N_SECTOR_READS` | 128 | `include/defaults.h` | Max sectors readable per beam search iteration |
| `MAX_GRAPH_DEGREE` | 512 | `include/defaults.h` | Maximum allowed graph degree |
| `MAX_PQ_CHUNKS` | 512 | `include/pq_common.h` | Maximum PQ chunks |
| `MAX_IO_DEPTH` | 128 | `include/aligned_file_reader.h` | Max concurrent IO operations per thread |
| `MAX_EVENTS` | 1024 | `src/linux_aligned_file_reader.cpp` | Max AIO events per io_submit batch |
| `FULL_PRECISION_REORDER_MULTIPLIER` | 3 | `include/pq_flash_index.h` | Fetch 3×k candidates for reorder |
| `MAX_SAMPLE_POINTS_FOR_WARMUP` | 100000 | `include/disk_utils.h` | Max warmup sample size |
| `PQ_TRAINING_SET_FRACTION` | 0.1 | `include/disk_utils.h` | Fraction of data for PQ training |
| `SPACE_FOR_CACHED_NODES_IN_GB` | 0.25 | `include/disk_utils.h` | RAM reserved for node cache |
| `THRESHOLD_FOR_CACHING_IN_GB` | 1.0 | `include/disk_utils.h` | Minimum budget to enable caching |
| `NUM_NODES_TO_CACHE` | 250000 | `include/disk_utils.h` | Default cache size |
| `WARMUP_L` | 20 | `include/disk_utils.h` | L_search for warmup queries |
| `NUM_KMEANS_REPS` | 12 | `include/disk_utils.h` | K-means repetitions |

---

## Template Instantiations

Both `PQFlashIndex` and `build_disk_index` are instantiated for:
- `<float, uint32_t>`, `<int8_t, uint32_t>`, `<uint8_t, uint32_t>`
- `<float, uint16_t>`, `<int8_t, uint16_t>`, `<uint8_t, uint16_t>`

---

## Internal Helper Functions (PQFlashIndex)

| Function | File | Purpose |
|----------|------|---------|
| `get_node_sector(node_id)` | `src/pq_flash_index.cpp:102` | Computes which sector contains a given node |
| `offset_to_node(sector_buf, node_id)` | `src/pq_flash_index.cpp:108` | Returns pointer to node within sector buffer |
| `offset_to_node_nhood(node_buf)` | `src/pq_flash_index.cpp:114` | Returns pointer to neighborhood data within node |
| `offset_to_node_coords(node_buf)` | `src/pq_flash_index.cpp:119` | Returns pointer to coordinate data within node |
| `setup_thread_data(nthreads, visited_reserve)` | `src/pq_flash_index.cpp:124` | Creates per-thread scratch + registers IO contexts |
| `use_medoids_data_as_centroids()` | `src/pq_flash_index.cpp:509` | Reads medoid vectors from disk to use as centroid data |
| `point_has_label(point_id, label_id)` | `src/pq_flash_index.cpp:651` | Checks if a point has a specific label |
| `parse_label_file(infile, num_points_labels)` | `src/pq_flash_index.cpp:670` | Parses point-to-label assignments |
| `load_label_map(infile)` | `src/pq_flash_index.cpp:562` | Loads string→LabelT mapping |
| `generate_random_labels(labels, num, nthreads)` | `src/pq_flash_index.cpp:537` | Samples random labels from distribution |

---

## Build-Time Internal Functions (disk_utils.cpp)

| Function | Purpose |
|----------|---------|
| `add_new_file_to_single_index()` | Appends a new file's contents to an existing single-file index |
| `get_memory_budget()` | Converts RAM budget string/double to bytes, reserving space for cache |
| `calculate_num_pq_chunks()` | Computes optimal PQ chunks given RAM budget and point count |
| `generateRandomWarmup<T>()` | Creates random vectors for warmup when no sample file exists |
| `merge_shards()` | Merges per-shard Vamana graphs with ID remapping and deduplication |
| `breakup_dense_points<T>()` | Splits high-label-density points into dummy copies |
| `extract_shard_labels()` | Extracts per-shard label subsets from global label file |
| `build_merged_vamana_index()` | Full pipeline: partition → build shards → merge |
| `optimize_beamwidth()` | Binary-search style tuning of beam width for QPS |
| `create_disk_layout<T>()` | Converts in-memory graph + base vectors to sector-aligned disk format |
| `build_disk_index()` | Top-level orchestrator: parse params → preprocess → PQ → build → layout |
