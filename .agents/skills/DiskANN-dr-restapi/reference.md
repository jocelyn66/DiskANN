# DiskANN REST API — Detailed Reference

## Types

### `diskann::SearchResult`

**File**: `include/restapi/search_wrapper.h` (lines 18–56)
**Impl**: `src/restapi/search_wrapper.cpp` (lines 27–48)

```cpp
class SearchResult {
public:
    SearchResult(unsigned int K, unsigned int elapsed_time_in_ms,
                 const unsigned *const indices, const float *const distances,
                 const std::string *const tags = nullptr,
                 const unsigned *const partitions = nullptr);

    const std::vector<unsigned int> &get_indices() const;
    const std::vector<float> &get_distances() const;
    bool tags_enabled() const;
    const std::vector<std::string> &get_tags() const;
    bool partitions_enabled() const;
    const std::vector<unsigned> &get_partitions() const;
    unsigned get_time() const;

private:
    unsigned int _K;
    unsigned int _search_time_in_ms;
    std::vector<unsigned int> _indices;
    std::vector<float> _distances;
    bool _tags_enabled;
    std::vector<std::string> _tags;
    bool _partitions_enabled;
    std::vector<unsigned> _partitions;
};
```

**Constructor behavior**: Copies `K` elements from raw arrays into vectors. Sets `_tags_enabled` / `_partitions_enabled` to true only when the corresponding pointer is non-null.

---

### `diskann::SearchNotImplementedException`

**File**: `include/restapi/search_wrapper.h` (lines 58–75)

Thrown by default `BaseSearch::search()` overloads. Message includes the unsupported data type name.

---

### `diskann::BaseSearch`

**File**: `include/restapi/search_wrapper.h` (lines 82–101)
**Impl**: `src/restapi/search_wrapper.cpp` (lines 50–82)

```cpp
class BaseSearch {
public:
    BaseSearch(const std::string &tagsFile = nullptr);
    virtual SearchResult search(const float *query, unsigned int dimensions, unsigned int K, unsigned int Ls);
    virtual SearchResult search(const int8_t *query, unsigned int dimensions, unsigned int K, unsigned int Ls);
    virtual SearchResult search(const uint8_t *query, unsigned int dimensions, unsigned int K, unsigned int Ls);
    void lookup_tags(unsigned K, const unsigned *indices, std::string *ret_tags);

protected:
    bool _tags_enabled;
    std::vector<std::string> _tags_str;
};
```

**Constructor**: If `tagsFile` is non-empty, opens the file and reads one tag per line into `_tags_str`. Sets `_tags_enabled = true`. Otherwise sets `_tags_enabled = false`.

**`lookup_tags()`**: For each of `K` indices, copies `_tags_str[indices[k]]` into `ret_tags[k]`. Throws `std::runtime_error` if tags not enabled or index out of range.

**Default `search()` overloads**: All three throw `SearchNotImplementedException`. Concrete subclasses override only the matching type.

---

### `diskann::InMemorySearch<T>`

**File**: `include/restapi/search_wrapper.h` (lines 103–115)
**Impl**: `src/restapi/search_wrapper.cpp` (lines 86–125)

```cpp
template <typename T>
class InMemorySearch : public BaseSearch {
public:
    InMemorySearch(const std::string &baseFile, const std::string &indexFile,
                   const std::string &tagsFile, Metric m,
                   uint32_t num_threads, uint32_t search_l);
    virtual ~InMemorySearch();
    SearchResult search(const T *query, unsigned int dimensions, unsigned int K, unsigned int Ls);

private:
    unsigned int _dimensions, _numPoints;
    std::unique_ptr<diskann::Index<T>> _index;
};
```

**Constructor parameters**:
| Param | Description |
|---|---|
| `baseFile` | Binary data file — read for metadata via `diskann::get_bin_metadata()` |
| `indexFile` | Index path prefix — passed to `_index->load()` |
| `tagsFile` | Tags file (one per line) — forwarded to `BaseSearch` |
| `m` | Distance metric (`diskann::Metric::L2` or `INNER_PRODUCT`) |
| `num_threads` | Thread count for search |
| `search_l` | Default search list size L |

**Constructor flow**:
1. `BaseSearch(tagsFile)` — loads tags
2. `diskann::get_bin_metadata(baseFile, total_points, dimensions)` — reads dimensionality
3. `diskann::IndexSearchParams(search_l, num_threads)` — creates search params
4. `new diskann::Index<T>(m, dimensions, total_points, nullptr, search_params, 0, false)` — constructs index
5. `_index->load(indexFile.c_str(), num_threads, search_l)` — loads graph from disk

**`search()` flow**:
1. Allocates `unsigned int[K]` and `float[K]`
2. Calls `_index->search(query, K, Ls, indices, distances)`
3. Measures wall-clock time via `std::chrono::high_resolution_clock`
4. If `_tags_enabled`, allocates `string[K]` and calls `lookup_tags()`
5. Constructs and returns `SearchResult(K, duration, indices, distances, tags)`
6. Frees `indices` and `distances`

**Template instantiations** (line 190–192): `float`, `int8_t`, `uint8_t`

---

### `diskann::PQFlashSearch<T>`

**File**: `include/restapi/search_wrapper.h` (lines 117–132)
**Impl**: `src/restapi/search_wrapper.cpp` (lines 127–196)

```cpp
template <typename T>
class PQFlashSearch : public BaseSearch {
public:
    PQFlashSearch(const std::string &indexPrefix, unsigned num_nodes_to_cache,
                  unsigned num_threads, const std::string &tagsFile, Metric m);
    virtual ~PQFlashSearch();
    SearchResult search(const T *query, unsigned int dimensions, unsigned int K, unsigned int Ls);

private:
    unsigned int _dimensions, _numPoints;
    std::unique_ptr<diskann::PQFlashIndex<T>> _index;
    std::shared_ptr<AlignedFileReader> reader;
};
```

**Constructor parameters**:
| Param | Description |
|---|---|
| `indexPrefix` | Path prefix — `_disk.index` and `_sample_data.bin` are appended |
| `num_nodes_to_cache` | How many BFS nodes to cache around medoid(s) |
| `num_threads` | Thread count — also sets `omp_set_num_threads()` |
| `tagsFile` | Tags file — forwarded to `BaseSearch` |
| `m` | Distance metric |

**Constructor flow**:
1. Creates platform-specific `AlignedFileReader` (`LinuxAlignedFileReader` on Linux, `WindowsAlignedFileReader` on Windows, `BingAlignedFileReader` if `USE_BING_INFRA`)
2. `new diskann::PQFlashIndex<T>(reader, m)` — constructs flash index
3. `_index->load(num_threads, index_prefix_path.c_str())` — loads from disk
4. `_index->cache_bfs_levels(num_nodes_to_cache, node_list)` — computes BFS cache
5. `_index->load_cache_list(node_list)` — loads nodes into memory cache
6. `omp_set_num_threads(num_threads)`

**`search()` flow**:
1. Allocates `uint64_t[K]` (PQ flash uses 64-bit IDs), `unsigned[K]`, `float[K]`
2. Calls `_index->cached_beam_search(query, K, Ls, indices_u64, distances, DEFAULT_W)` where `DEFAULT_W = 1`
3. Downcasts `uint64_t` indices to `unsigned` (truncation)
4. Measures wall-clock time
5. If `_tags_enabled`, looks up tags
6. Returns `SearchResult`

**Constant**: `DEFAULT_W = 1` (line 25 of `search_wrapper.cpp`) — beam width for `cached_beam_search`

**Template instantiations** (line 194–196): `float`, `int8_t`, `uint8_t`

---

### `diskann::Server`

**File**: `include/restapi/server.h`
**Impl**: `src/restapi/server.cpp`

```cpp
class Server {
public:
    Server(web::uri &url, std::vector<std::unique_ptr<diskann::BaseSearch>> &multi_searcher,
           const std::string &typestring);
    virtual ~Server();
    pplx::task<void> open();
    pplx::task<void> close();

protected:
    template <class T> void handle_post(web::http::http_request message);

    template <typename T>
    web::json::value toJsonArray(const std::vector<T> &v,
                                  std::function<web::json::value(const T &)> valConverter);
    web::json::value prepareResponse(const int64_t &queryId, const int k);

    template <class T>
    void parseJson(const utility::string_t &body, unsigned int &k, int64_t &queryId,
                   T *&queryVector, unsigned int &dimensions, unsigned &Ls);

    web::json::value idsToJsonArray(const diskann::SearchResult &result);
    web::json::value distancesToJsonArray(const diskann::SearchResult &result);
    web::json::value tagsToJsonArray(const diskann::SearchResult &result);
    web::json::value partitionsToJsonArray(const diskann::SearchResult &result);

    SearchResult aggregate_results(unsigned K, const std::vector<diskann::SearchResult> &results);

private:
    bool _isDebug;
    std::unique_ptr<web::http::experimental::listener::http_listener> _listener;
    const bool _multi_search;
    std::vector<std::unique_ptr<diskann::BaseSearch>> _multi_searcher;
};
```

---

## Functions

### `Server::Server(uri, multi_searcher, typestring)`

**File**: `src/restapi/server.cpp` (lines 16–42)

**Behavior**:
1. Sets `_multi_search = (multi_searcher.size() > 1)`
2. Moves all searchers from the passed vector into `_multi_searcher`
3. Creates `http_listener` from URI
4. Dispatches on `typestring`:
   - `"float"` → binds `handle_post<float>` (uses `_listener->support()` without method — defaults to POST)
   - `"int8_t"` → binds `handle_post<int8_t>` (explicitly `methods::POST`)
   - `"uint8_t"` → binds `handle_post<uint8_t>` (explicitly `methods::POST`)
   - else → throws string literal `"Unsupported type in server constuctor"` (note: typo "constuctor" is in source)

**Note**: The `"float"` branch uses a different `support()` overload (no method specified) vs. `"int8_t"` / `"uint8_t"` (explicit `methods::POST`). Both should work but it's inconsistent.

---

### `Server::handle_post<T>(message)`

**File**: `src/restapi/server.cpp` (lines 79–136)

**Flow**:
1. `message.extract_string(true)` — extracts body as string (true = force extraction)
2. `.then()` continuation:
   - Calls `parseJson<T>()` to extract query parameters
   - Records start time
   - Iterates `_multi_searcher`, calling `search()` on each
   - Calls `aggregate_results()` to merge results
   - Frees query vector via `diskann::aligned_free()`
   - Builds JSON response via `prepareResponse()` + `idsToJsonArray()` etc.
   - Computes `time_taken_in_us` (microseconds from start)
   - Returns `(HTTP 200, response)` on success
   - Returns `(HTTP 500, {error: msg})` on exception
3. Second `.then()`: calls `message.reply()` with status + JSON

---

### `Server::parseJson<T>(body, k, queryId, queryVector, dimensions, Ls)`

**File**: `src/restapi/server.cpp` (lines 148–177)

**Behavior**:
1. Prints raw body to stdout (debug)
2. Parses JSON via `web::json::value::parse(body)`
3. Extracts `query` array, `query_id` (optional, default -1), `Ls` (optional, default `DEFAULT_L` = 100), `k` (required)
4. **Validation**: `k > 0 && k <= Ls`, query array not empty — throws `std::invalid_argument` on failure (note: uses `throw new` which leaks — known issue in source)
5. Computes `dimensions = queryArr.size()`
6. Allocates: `diskann::alloc_aligned((void**)&queryVector, ROUND_UP(dim, 8) * sizeof(T), 8 * sizeof(T))`
7. `memset` to zero (note: uses `sizeof(float)` instead of `sizeof(T)` — potential bug for non-float types if `sizeof(T) > sizeof(float)`)
8. Copies query values: `queryVector[i] = (float)queryArr[i].as_double()` — casts to float then assigns to T

---

### `Server::aggregate_results(K, results)`

**File**: `src/restapi/server.cpp` (lines 62–109)

**Multi-index mode** (`_multi_search == true`):
1. Allocates `best_indices[K]`, `best_distances[K]`, `best_partitions[K]`, optionally `best_tags[K]`
2. Maintains `pos[numsearchers]` position pointers (all start at 0)
3. For each of K output slots: scans all searchers to find minimum distance at current position
4. Records the best index, distance, partition, and tag
5. Advances the position pointer for the winning searcher
6. Sums all search times across searchers for total_time
7. Constructs merged `SearchResult`

**Single-index mode**: Returns `results[0]` directly.

**Assumption**: Each searcher's results are sorted by ascending distance. The merge is a standard K-way merge of sorted lists.

---

### `Server::prepareResponse(queryId, k)`

**File**: `src/restapi/server.cpp` (lines 138–145)

Creates a JSON object with `query_id` and `k` fields. Callers add `indices`, `distances`, etc.

---

### `Server::idsToJsonArray(result)` / `distancesToJsonArray` / `tagsToJsonArray` / `partitionsToJsonArray`

**File**: `src/restapi/server.cpp` (lines 185–222)

Each iterates the corresponding vector from `SearchResult` and builds a `web::json::value::array()`. Uses `web::json::value::number()` for numeric types and `web::json::value::string()` for tags.

---

## Server Binary Reference

### `inmem_server` (`apps/restapi/inmem_server.cpp`)

**CLI arguments** (via `boost::program_options`):
| Arg | Type | Required | Default | Description |
|---|---|---|---|---|
| `--data_type` | string | yes | — | `int8`, `uint8`, or `float` |
| `--address` | string | yes | — | HTTP address (e.g., `http://0.0.0.0:8080`) |
| `--data_file` | string | yes | — | Binary data file path |
| `--index_path_prefix` | string | yes | — | Index file path prefix |
| `--num_threads,-T` | uint32 | yes | — | Thread count |
| `--l_search` | uint32 | yes | — | Search list size L |
| `--dist_fn` | string | no | `"l2"` | `l2` or `mips` |
| `--tags_file` | string | no | `""` | Tags file (one tag per line) |

**Global state**: `g_httpServer` (Server), `g_inMemorySearch` (vector of BaseSearch)

---

### `ssd_server` (`apps/restapi/ssd_server.cpp`)

**CLI arguments**:
| Arg | Type | Required | Default | Description |
|---|---|---|---|---|
| `--data_type` | string | yes | — | `int8`, `uint8`, or `float` |
| `--address` | string | yes | — | HTTP address |
| `--index_path_prefix` | string | yes | — | Index prefix (disk.index derived from this) |
| `--num_nodes_to_cache` | uint32 | no | `0` | BFS cache size |
| `--num_threads,-T` | uint32 | no | `omp_get_num_procs()` | Thread count |
| `--dist_fn` | string | no | `"l2"` | `l2` or `mips` |
| `--tags_file` | string | no | `""` | Tags file |

**Global state**: `g_httpServer` (Server), `g_ssdSearch` (vector of BaseSearch)

---

### `multiple_ssdindex_server` (`apps/restapi/multiple_ssdindex_server.cpp`)

**CLI arguments**:
| Arg | Type | Required | Default | Description |
|---|---|---|---|---|
| `--data_type` | string | yes | — | `int8`, `uint8`, or `float` |
| `--address` | string | yes | — | HTTP address |
| `--index_prefix_paths` | string | yes | — | File containing index prefixes (one per line) |
| `--num_nodes_to_cache` | uint32 | no | `0` | BFS cache per index |
| `--num_threads,-T` | uint32 | no | `omp_get_num_procs()` | Thread count |
| `--dist_fn` | string | no | `"l2"` | `l2` or `mips` |
| `--tags_file` | string | yes (paired) | — | File containing tag file paths (one per line, must match index count) |

**Multi-index loading**: Reads `index_prefix_paths` and `tags_file` line-by-line in parallel, pairs them, creates one `PQFlashSearch<T>` per pair. All pushed into `g_ssdSearch`.

---

### `client` (`apps/restapi/client.cpp`)

**CLI arguments**:
| Arg | Type | Required | Default | Description |
|---|---|---|---|---|
| `--data_type` | string | yes | — | `int8`, `uint8`, or `float` |
| `--address` | string | yes | — | Server HTTP address |
| `--query_file` | string | yes | — | Binary query file |
| `--num_queries,-Q` | uint32 | yes | — | Number of queries to send |
| `--l_search` | uint32 | yes | — | Search list size |
| `--k_value,-K` | uint32 | no | `10` | Number of neighbors |

**Function**: `query_loop<T>(ip_addr_port, query_file, nq, Ls, k_value)` — loads binary query file via `diskann::load_aligned_bin<T>()`, iterates queries, builds JSON, POSTs to server, prints response.

---

### `main.cpp` (Legacy) (`apps/restapi/main.cpp`)

**CLI**: Positional args — `<ip_addr_and_port> <index_file> <base_file> <ids_file>`

Uses non-templated `diskann::InMemorySearch` via `restapi/in_memory_search.h` (header not found in current tree). Uses `diskann::L2` metric only. This binary is **likely deprecated** — it references a non-existent header and uses a simpler `Server` constructor overload (2 args instead of 3).

---

## Build Configuration (`apps/restapi/CMakeLists.txt`)

Four targets with identical link patterns:

| Target | Source | Description |
|---|---|---|
| `inmem_server` | `inmem_server.cpp` | In-memory server |
| `ssd_server` | `ssd_server.cpp` | SSD server |
| `multiple_ssdindex_server` | `multiple_ssdindex_server.cpp` | Multi-index server |
| `client` | `client.cpp` | HTTP client |

**Linux link libraries** (all server targets):
`${PROJECT_NAME}` `aio` `-ltcmalloc` `-lboost_system` `-lcrypto` `-lssl` `-lcpprest` `Boost::program_options`

**Client link libraries** (Linux):
`${PROJECT_NAME}` `-lboost_system` `-lcrypto` `-lssl` `-lcpprest` `Boost::program_options`
(No `aio` or `tcmalloc` — client doesn't load indices)

**MSVC**: Links against `diskann_dll.lib` (debug/release variants), `Boost::program_options`. CppRestSDK configured in parent `CMakeLists.txt` via `DISKANN_MSVC_PACKAGES`.

---

## Known Issues in Source

1. **`throw new std::invalid_argument(...)`** in `parseJson()` — uses `throw new` which leaks the exception object. Should be `throw std::invalid_argument(...)`.
2. **`memset(queryVector, 0, new_dim * sizeof(float))`** in `parseJson<T>()` — uses `sizeof(float)` regardless of `T`. For `int8_t`/`uint8_t`, this zeros more memory than needed (harmless but incorrect).
3. **`queryVector[i] = (float)queryArr[i].as_double()`** — casts JSON double to float then implicitly converts to `T`. For integer types, this truncates fractional parts.
4. **`_listener->support()` inconsistency** — the `"float"` branch omits `methods::POST` while other branches specify it explicitly.
5. **64→32 bit index truncation** in `PQFlashSearch::search()` — `indices_u64[k]` (uint64_t) is assigned to `indices[k]` (unsigned) which silently truncates for datasets with >4B points.
6. **`main.cpp` references missing header** `restapi/in_memory_search.h` — likely a build artifact from an older version.
