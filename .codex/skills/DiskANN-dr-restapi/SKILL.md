---
name: DiskANN-dr-restapi
description: Use when working with the REST API module of DiskANN — HTTP server exposing in-memory and disk-based index search endpoints
---

# DiskANN REST API Module

## 1. Module Purpose

The restapi module wraps DiskANN's core index types (in-memory Vamana and SSD-based PQ Flash) behind an HTTP JSON API, enabling nearest-neighbor search queries over the network.

### Capabilities

- **In-memory index server** (`apps/restapi/inmem_server.cpp`): Loads a `diskann::Index<T>` into RAM, serves ANN queries via HTTP POST. Supports `float`, `int8_t`, `uint8_t`.
- **SSD/disk index server** (`apps/restapi/ssd_server.cpp`): Loads a `diskann::PQFlashIndex<T>` for disk-based ANN search with configurable BFS node caching and beam search.
- **Multi-index server** (`apps/restapi/multiple_ssdindex_server.cpp`): Loads multiple `PQFlashIndex<T>` instances from a file listing index paths, searches all per query, merges results by distance via `aggregate_results()`.
- **Client** (`apps/restapi/client.cpp`): CLI HTTP client that reads queries from a binary file (`diskann::load_aligned_bin<T>()`), sends them as JSON POST requests, prints responses.
- **Legacy server** (`apps/restapi/main.cpp`): Older non-templated in-memory server using `diskann::InMemorySearch` (via `restapi/in_memory_search.h`, header no longer present in tree — likely deprecated).

### Protocol

| Aspect | Detail |
|---|---|
| Transport | HTTP over TCP via `cpprest` `http_listener` |
| Method | POST only |
| Request body | JSON: `{"query": [v0, v1, ...], "k": N, "Ls": M, "query_id": ID}` |
| Response body | JSON: `{"indices": [...], "distances": [...], "tags": [...], "partition": [...], "time_taken_in_us": N, "query_id": ID, "k": N}` |
| Error response | HTTP 500 with `{"error": "message", "query_id": ID, "k": N}` |

### Build System

- Guarded by the CMake `RESTAPI` option. When enabled, `src/CMakeLists.txt` (line 17–18) appends `restapi/search_wrapper.cpp` and `restapi/server.cpp` to the core `diskann` library sources.
- `CMakeLists.txt` (line 326) adds `apps/restapi/` subdirectory for server/client binaries.
- `apps/restapi/CMakeLists.txt` defines four targets: `inmem_server`, `ssd_server`, `multiple_ssdindex_server`, `client`.
- Link dependencies (Linux): `${PROJECT_NAME}` (diskann lib), `aio`, `tcmalloc`, `boost_system`, `crypto`, `ssl`, `cpprest`, `Boost::program_options`.

---

## 2. Core Design Logic

### Architecture: Facade + Strategy

```
Server binary (e.g. inmem_server.cpp)
  └─ parses CLI args via boost::program_options
  └─ creates concrete searcher(s) → pushes into global vector
  └─ instantiates Server with the searcher vector
       └─ Server binds handle_post<T> to http_listener POST
       └─ handle_post: parseJson → search each searcher → aggregate → respond
```

**Why this design:**
- The `BaseSearch` abstract class (`include/restapi/search_wrapper.h`) decouples HTTP handling from index implementation. `Server` operates on `BaseSearch*` pointers and never touches `diskann::Index<T>` or `PQFlashIndex<T>` directly.
- Template dispatch happens once at startup: the `Server` constructor (`src/restapi/server.cpp`, line 18–42) checks `typestring` and binds the correct `handle_post<float>`, `handle_post<int8_t>`, or `handle_post<uint8_t>`.
- Multi-index support is a boolean flag (`_multi_search`) set in the constructor when `multi_searcher.size() > 1`. The `aggregate_results()` method performs K-way merge only when this flag is true; otherwise it returns the single result directly.

### Why CppRestSDK

The module uses Microsoft's C++ REST SDK (Casablanca) for:
- Async HTTP listener (`web::http::experimental::listener::http_listener`)
- PPL tasks (`.then()` chaining) for non-blocking request handling
- Built-in JSON parser/serializer (`web::json::value`)
- Cross-platform support (Linux + Windows in `CMakeLists.txt`)

### Resilience Pattern

All server binaries share an identical infinite-loop pattern:
```cpp
while (1) {
    try {
        setup(address, data_type);  // create Server, open listener
        // wait for stdin "exit"
    } catch (...) {
        teardown(address);          // close listener
        // loop restarts server
    }
}
```
This auto-restarts the HTTP listener on uncaught exceptions.

---

## 3. Core Data Structures

### `diskann::SearchResult` (`include/restapi/search_wrapper.h`, lines 18–56)

Encapsulates results from a single search call.

| Field | Type | Description |
|---|---|---|
| `_K` | `unsigned int` | Number of neighbors requested |
| `_search_time_in_ms` | `unsigned int` | Wall-clock search duration |
| `_indices` | `vector<unsigned int>` | Point indices in the dataset |
| `_distances` | `vector<float>` | Distances to query |
| `_tags_enabled` | `bool` | Whether string tags were resolved |
| `_tags` | `vector<string>` | Human-readable tags per result |
| `_partitions_enabled` | `bool` | Whether partition info is present |
| `_partitions` | `vector<unsigned>` | Source partition index (multi-index mode) |

Constructor (`src/restapi/search_wrapper.cpp`, lines 27–48): copies indices/distances into vectors; conditionally copies tags and partitions. Sets `_tags_enabled` / `_partitions_enabled` based on whether those arrays are non-null.

### `diskann::BaseSearch` (`include/restapi/search_wrapper.h`, lines 82–101)

Abstract search backend.

| Member | Purpose |
|---|---|
| `_tags_enabled` | Whether tag lookup is active |
| `_tags_str` | `vector<string>` loaded from a tags file (one tag per line) |
| `search(const float*, ...)` | Virtual, default throws `SearchNotImplementedException` |
| `search(const int8_t*, ...)` | Virtual, default throws `SearchNotImplementedException` |
| `search(const uint8_t*, ...)` | Virtual, default throws `SearchNotImplementedException` |
| `lookup_tags(K, indices, ret_tags)` | Translates point indices to tag strings via `_tags_str[index]` |

Constructor (`src/restapi/search_wrapper.cpp`, lines 50–70): reads tags file line by line into `_tags_str` if non-empty.

### `diskann::InMemorySearch<T>` (`include/restapi/search_wrapper.h`, lines 103–115)

| Field | Type | Description |
|---|---|---|
| `_index` | `unique_ptr<diskann::Index<T>>` | Core Vamana in-memory index |
| `_dimensions` | `unsigned int` | Vector dimensionality |
| `_numPoints` | `unsigned int` | Number of points in index |

Constructor (`src/restapi/search_wrapper.cpp`, lines 86–97): calls `diskann::get_bin_metadata()` on `baseFile` to get dimensions/count, creates `diskann::Index<T>` with `IndexSearchParams(search_l, num_threads)`, then calls `_index->load()`.

`search()` (`src/restapi/search_wrapper.cpp`, lines 100–121): allocates `unsigned int[K]` and `float[K]`, calls `_index->search(query, K, Ls, indices, distances)`, optionally looks up tags, wraps into `SearchResult`.

### `diskann::PQFlashSearch<T>` (`include/restapi/search_wrapper.h`, lines 117–132)

| Field | Type | Description |
|---|---|---|
| `_index` | `unique_ptr<diskann::PQFlashIndex<T>>` | Core PQ flash disk index |
| `reader` | `shared_ptr<AlignedFileReader>` | Platform-specific aligned I/O reader |
| `_dimensions` | `unsigned int` | Vector dimensionality |
| `_numPoints` | `unsigned int` | Number of points in index |

Constructor (`src/restapi/search_wrapper.cpp`, lines 127–158): creates platform-appropriate `AlignedFileReader` (Linux: `LinuxAlignedFileReader`, Windows: `WindowsAlignedFileReader`), constructs `PQFlashIndex<T>`, calls `_index->load()`, then warms BFS node cache via `_index->cache_bfs_levels(num_nodes_to_cache, node_list)` + `_index->load_cache_list(node_list)`.

`search()` (`src/restapi/search_wrapper.cpp`, lines 161–186): allocates `uint64_t[K]` for indices (PQ flash uses 64-bit IDs), calls `_index->cached_beam_search(query, K, Ls, indices_u64, distances, DEFAULT_W)` where `DEFAULT_W = 1` (beam width). Downcasts indices to `unsigned` for `SearchResult`.

### `diskann::Server` (`include/restapi/server.h`)

| Field | Type | Description |
|---|---|---|
| `_listener` | `unique_ptr<http_listener>` | CppRestSDK HTTP listener |
| `_multi_searcher` | `vector<unique_ptr<BaseSearch>>` | Polymorphic search backends |
| `_multi_search` | `const bool` | True when multiple searchers loaded |
| `_isDebug` | `bool` | Debug flag |

Key methods in `src/restapi/server.cpp`:
- `handle_post<T>(message)`: extracts body string → `parseJson<T>` → iterates searchers → `aggregate_results` → JSON response → reply
- `parseJson<T>(body, k, queryId, queryVector, dimensions, Ls)`: parses JSON, validates `k > 0 && k <= Ls`, allocates aligned query vector via `diskann::alloc_aligned`, zero-pads to `ROUND_UP(dimensions, 8)`
- `aggregate_results(K, results)`: when `_multi_search`, performs K-way merge by scanning minimum distance across all result sets using position pointers per searcher
- JSON serializers: `idsToJsonArray`, `distancesToJsonArray`, `tagsToJsonArray`, `partitionsToJsonArray`

### JSON Key Constants (`include/restapi/common.h`)

| Constant | Value | Purpose |
|---|---|---|
| `VECTOR_KEY` | `"query"` | Query vector array |
| `K_KEY` | `"k"` | Number of neighbors |
| `L_KEY` | `"Ls"` | Search list size |
| `QUERY_ID_KEY` | `"query_id"` | Client echo ID |
| `INDICES_KEY` | `"indices"` | Result point indices |
| `DISTANCES_KEY` | `"distances"` | Result distances |
| `TAGS_KEY` | `"tags"` | Optional string tags |
| `PARTITION_KEY` | `"partition"` | Partition index per result |
| `TIME_TAKEN_KEY` | `"time_taken_in_us"` | Server-side latency |
| `ERROR_MESSAGE_KEY` | `"error"` | Error description |
| `UNKNOWN_ERROR` | `"unknown_error"` | Fallback error string |
| `DEFAULT_L` | `100` | Default Ls when client omits it |

### Template Instantiations

`src/restapi/search_wrapper.cpp` (lines 190–196) explicitly instantiates:
```
InMemorySearch<float>, InMemorySearch<int8_t>, InMemorySearch<uint8_t>
PQFlashSearch<float>, PQFlashSearch<int8_t>, PQFlashSearch<uint8_t>
```

---

## 4. State Flow

### Server Startup (inmem_server)

```
main()
  ├─ boost::program_options: data_type, address, data_file, index_path_prefix,
  │                          num_threads, l_search, dist_fn, tags_file
  ├─ resolve Metric (L2 or INNER_PRODUCT)
  ├─ switch(data_type):
  │     create InMemorySearch<T>(data_file, index_file, tags_file, metric, num_threads, l_search)
  │       ├─ BaseSearch(tagsFile) → read tags file line-by-line
  │       ├─ get_bin_metadata(baseFile) → dimensions, total_points
  │       └─ Index<T>::load(indexFile, num_threads, search_l)
  │     push into g_inMemorySearch
  └─ while(1):
       ├─ setup(address, data_type)
       │     ├─ Server(uri, g_inMemorySearch, typestring)
       │     │     ├─ move searchers into _multi_searcher
       │     │     ├─ set _multi_search = (size > 1)
       │     │     └─ bind handle_post<T> to POST via std::bind
       │     └─ listener->open().wait()
       ├─ getline(cin) → "exit" → teardown + exit
       └─ catch → teardown → loop restarts
```

### Server Startup (multiple_ssdindex_server)

```
main()
  ├─ parse: address, data_type, index_prefix_paths (file), tags_file (file),
  │         num_nodes_to_cache, num_threads, dist_fn
  ├─ read index_prefix_paths file line-by-line, read tags_file line-by-line
  │   → paired into vector<pair<string,string>> index_tag_paths
  │   → if tags_file has fewer lines than index_prefix_paths, prints
  │     "The number of tags specified does not match the number of indices specified"
  │     and calls exit(-1)
  ├─ for each (prefix, tagfile) in index_tag_paths:
  │     PQFlashSearch<T>(prefix, num_nodes_to_cache, num_threads, tagfile, metric)
  │       ├─ create AlignedFileReader (platform-specific)
  │       ├─ PQFlashIndex<T>::load(num_threads, prefix)
  │       ├─ cache_bfs_levels(num_nodes_to_cache) → node_list
  │       └─ load_cache_list(node_list)
  │     push into g_ssdSearch
  └─ same server loop
```

### Request Handling

```
HTTP POST → handle_post<T>(message)
  ├─ message.extract_string(true).then([](body):
  │     ├─ parseJson<T>(body, k, queryId, queryVector, dimensions, Ls)
  │     │     ├─ web::json::value::parse(body)
  │     │     ├─ extract query array, query_id (optional, default -1), Ls (optional, default 100)
  │     │     ├─ k = val.at("k").as_integer()
  │     │     ├─ validate: k > 0, k <= Ls, query not empty
  │     │     ├─ diskann::alloc_aligned(queryVector, ROUND_UP(dim, 8) * sizeof(T))
  │     │     └─ memset + copy query values from JSON array
  │     ├─ startTime = high_resolution_clock::now()
  │     ├─ for each searcher in _multi_searcher:
  │     │     results.push_back(searcher->search(queryVector, dim, K, Ls))
  │     ├─ aggregate_results(K, results)
  │     │     └─ if _multi_search: K-way merge picking min distance, tracking partition
  │     │        else: return results[0]
  │     ├─ diskann::aligned_free(queryVector)
  │     ├─ build JSON response: query_id, k, indices, distances, [tags], [partition], time_taken_in_us
  │     └─ return (HTTP 200, response)
  │  on exception:
  │     └─ return (HTTP 500, {error: msg, query_id, k})
  └─ .then([](response_status):
       └─ message.reply(status, json).wait()
```

### Client Flow

```
client main()
  ├─ parse: data_type, address, query_file, num_queries, l_search, k_value
  ├─ query_loop<T>(address, query_file, nq, Ls, k)
  │     ├─ http_client client(address)
  │     ├─ load_aligned_bin<T>(query_file) → data, npts, ndims, rounded_dim
  │     └─ for i in [0, nq):
  │           ├─ build JSON: {query_id: i, k, Ls, query: [vec[0..ndims)]}
  │           ├─ POST to server
  │           ├─ .then() extract response string
  │           └─ print response
```

---

## 5. Modification Scenarios

### Scenario 1: Adding a New HTTP Endpoint (e.g., GET /status)

**Goal**: Expose a health-check endpoint.

**Files to modify**:
- `include/restapi/server.h` — add `void handle_get_status(web::http::http_request message)` declaration
- `src/restapi/server.cpp` — implement handler; in `Server` constructor, add:
  ```cpp
  _listener->support(web::http::methods::GET,
      std::bind(&Server::handle_get_status, this, std::placeholders::_1));
  ```

**Key details**: `cpprest` `http_listener::support()` allows binding different HTTP methods independently. The handler can access `_multi_searcher.size()` and `_multi_search` to report status. No changes needed in `BaseSearch` or server binaries.

### Scenario 2: Adding a New Data Type (e.g., `float16`)

**Goal**: Serve indices with half-precision vectors.

**Files to modify**:
1. `include/restapi/search_wrapper.h` — add `virtual SearchResult search(const float16*, ...)` in `BaseSearch` (default throws)
2. `src/restapi/search_wrapper.cpp` — add `InMemorySearch<float16>` and `PQFlashSearch<float16>` implementations + explicit template instantiations at bottom
3. `src/restapi/server.cpp` — add `else if (typestring == "float16")` in `Server` constructor (line ~37) to bind `handle_post<float16>`
4. Server binaries (`inmem_server.cpp`, `ssd_server.cpp`, `multiple_ssdindex_server.cpp`) — add `float16` branch in `data_type` if-else chain

**Prerequisite**: Core `diskann::Index<float16>` and `diskann::PQFlashIndex<float16>` must exist.

### Scenario 3: Adding Batch Query Support

**Goal**: Accept multiple queries in a single HTTP request for higher throughput.

**Files to modify**:
1. `include/restapi/common.h` — define `QUERIES_KEY = "queries"`
2. `include/restapi/server.h` — add `handle_post_batch<T>` or modify `handle_post<T>`
3. `src/restapi/server.cpp` — parse JSON array of query objects, loop `parseJson` + `search()` per query, return array of response objects

**Gotcha**: `parseJson<T>()` allocates aligned memory via `diskann::alloc_aligned()` and the caller frees via `diskann::aligned_free()`. In batch mode, each query needs its own allocation/deallocation. Consider parallelizing with `std::async` since searches are independent.

### Scenario 4: Replacing CppRestSDK with a Different HTTP Framework

**Goal**: Swap to a lighter HTTP library (e.g., cpp-httplib, Crow).

**Files to modify**:
1. `include/restapi/server.h` — replace `web::http::experimental::listener::http_listener` and `web::json::value`
2. `src/restapi/server.cpp` — reimplement `open()`, `close()`, `handle_post<T>`, JSON parsing
3. `apps/restapi/CMakeLists.txt` — remove `-lcpprest -lboost_system -lcrypto -lssl`, add new library
4. `apps/restapi/client.cpp` — replace `web::http::client::http_client`

**Key invariant**: `BaseSearch`, `SearchResult`, and `search_wrapper.cpp` are HTTP-agnostic. They should require zero changes. The HTTP layer is fully contained in `Server` and the client binary.

### Scenario 5: Adding Response Compression or Middleware

**Goal**: Add gzip compression, authentication, or rate-limiting.

**Files to modify**:
- `include/restapi/server.h` — add middleware state (e.g., `unordered_map<string, token_bucket>` for rate limiting)
- `src/restapi/server.cpp` — add logic at the top of `handle_post<T>()`:
  - Auth: check `message.headers()` for `Authorization` header; return 401/403 on failure
  - Rate limit: key by `message.remote_address()`, check token bucket before processing
  - Compression: after building response JSON, compress body and set `Content-Encoding: gzip`

**Key detail**: All request processing in `handle_post<T>` runs inside a `.then()` continuation on the PPL task chain, so middleware must fit within that async flow.

---

## 6. File Reference

| File | Role |
|---|---|
| `include/restapi/common.h` | JSON key string constants, `DEFAULT_L` |
| `include/restapi/search_wrapper.h` | `SearchResult`, `BaseSearch`, `InMemorySearch<T>`, `PQFlashSearch<T>` class declarations |
| `include/restapi/server.h` | `Server` class declaration wrapping `http_listener` |
| `src/restapi/search_wrapper.cpp` | Implementations: index loading, search dispatch, tag lookup, template instantiations |
| `src/restapi/server.cpp` | `Server` implementation: constructor, `handle_post<T>`, `parseJson<T>`, `aggregate_results`, JSON serializers |
| `apps/restapi/inmem_server.cpp` | In-memory index HTTP server binary |
| `apps/restapi/ssd_server.cpp` | SSD/disk index HTTP server binary |
| `apps/restapi/multiple_ssdindex_server.cpp` | Multi-SSD-index HTTP server with K-way result merging |
| `apps/restapi/client.cpp` | CLI HTTP client: reads binary query file, sends JSON POSTs |
| `apps/restapi/main.cpp` | Legacy non-templated in-memory server (references missing `in_memory_search.h`) |
| `apps/restapi/CMakeLists.txt` | Build targets: `inmem_server`, `ssd_server`, `multiple_ssdindex_server`, `client` |
| `src/CMakeLists.txt` (line 17–18) | Conditionally includes `restapi/` sources when `RESTAPI` is ON |
