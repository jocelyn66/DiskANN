---
name: DiskANN-dr
description: Use when working with DiskANN codebase — provides comprehensive module knowledge, design logic, and modification guides (generated from cpp_main branch)
---

# DiskANN — Deep Read Skills Index

| Field | Value |
|-------|-------|
| Source | `/Users/jo/workspace/index/DiskANN` (local) |
| Branch | `cpp_main` |
| Commit | `2a26950d` |
| Generated | 2025-07-15 |

## Module Catalog

| Module | Skill Path | Purpose |
|--------|-----------|---------|
| **infrastructure** | `DiskANN-dr-infrastructure/` | Foundational plumbing — binary file I/O, aligned alloc, scratch spaces, neighbor structs, logging, math (MKL k-means), parameters, exceptions |
| **distance** | `DiskANN-dr-distance/` | All vector distance/similarity computation — L2, cosine, inner product, PQ lookup tables, SIMD auto-selection |
| **storage** | `DiskANN-dr-storage/` | Vector data stores (in-memory full-precision, PQ-compressed) and graph adjacency stores behind abstract interfaces |
| **filtering-partition** | `DiskANN-dr-filtering-partition/` | Label-based filtering, PQ/OPQ codebook training, k-means data partitioning for disk index construction |
| **index** | `DiskANN-dr-index/` | Core in-memory Vamana graph index — build, search, insert, delete, consolidate, filtered search |
| **disk-index** | `DiskANN-dr-disk-index/` | PQFlashIndex for SSD-based ANN search with beam search, caching, and disk layout utilities |
| **apps** | `DiskANN-dr-apps/` | CLI entry points — build/search memory & disk indices, dynamic insert/delete tests, format converters |
| **python** | `DiskANN-dr-python/` | pybind11 bindings exposing `diskannpy` — build/search/dynamic ops for memory & disk indices from Python |
| **restapi** | `DiskANN-dr-restapi/` | HTTP JSON server (via cpprest) for in-memory, disk, and multi-index ANN search over the network |

## Dependency Graph

```
infrastructure  (leaf — no internal dependencies)
    │
    ├──► distance
    │        ↑
    ├──► storage ───────────┐
    │                       │
    ├──► filtering-partition─┤  (also depends on distance)
    │                       │
    ├──► index ◄────────────┘  (depends on storage, distance, filtering-partition)
    │
    ├──► disk-index  (depends on distance, storage, filtering-partition, infrastructure I/O)
    │
    ├──► apps        (depends on index, disk-index — consumer of all core modules)
    ├──► python      (depends on index, disk-index — wraps C++ core via pybind11)
    └──► restapi     (depends on index, disk-index — HTTP layer over search)
```

**Layer summary:**

1. **Foundation**: infrastructure
2. **Computation**: distance
3. **Data management**: storage, filtering-partition
4. **Core indices**: index (in-memory), disk-index (SSD)
5. **Interfaces**: apps (CLI), python (pybind11), restapi (HTTP)

## Cross-Module Scenarios

### 1. Building an In-Memory Index from Raw Vectors

**Modules**: infrastructure → distance → storage → filtering-partition (optional) → index → apps/python

1. **apps** (`build_memory_index.cpp`) or **python** (`build_memory_index()`) parses parameters
2. **infrastructure** (`utils.h`) loads the binary vector file via `load_aligned_bin<T>()`
3. **distance** selects the metric-appropriate `Distance<T>` implementation (L2/cosine/MIPS) via `get_distance_function<T>()`
4. **storage** creates an `InMemDataStore<T>` to hold vectors in RAM and an `InMemGraphStore` for adjacency lists
5. If filtered, **filtering-partition** (`filter_utils`) parses label files and builds universal-label mappings
6. **index** `Index<T>::build()` runs the Vamana greedy-search-based insertion algorithm, wiring the graph through `iterate_to_fixed_point()` and `prune_neighbors()` / `occlude_list()`
7. **index** `Index<T>::save()` serializes the graph and data to disk

### 2. Building a Disk Index (SSD-Optimized)

**Modules**: infrastructure → distance → filtering-partition → index → disk-index → apps/python

1. **apps** (`build_disk_index.cpp`) or **python** (`build_disk_index()`) invokes `diskann::build_disk_index<T>()`
2. **filtering-partition** trains PQ codebooks (`generate_pq_pivots()` + `generate_pq_data_from_pivots()`) for compressed in-RAM representation
3. **filtering-partition** (`partition.cpp`) optionally shards data into RAM-budget-constrained partitions via `partition_with_ram_budget()`
4. Per shard, **index** builds a Vamana graph in memory
5. **disk-index** (`disk_utils.cpp`) merges shard graphs via `merge_shards()`, creates the sector-aligned disk layout (`create_disk_layout()`), and writes the final `.index` file with 4KB-sector-aligned nodes

### 3. Searching an In-Memory Index

**Modules**: infrastructure → distance → storage → index → apps/python/restapi

1. Caller provides query vector(s) and search parameters (L = search list size)
2. **index** `Index<T>::search()` acquires an `InMemQueryScratch` from the scratch pool
3. **index** runs `iterate_to_fixed_point()`: greedy beam search expanding neighbors, computing distances via **distance**, maintaining a sorted candidate list
4. If filtered search, `search_with_filters()` uses label bitmaps to restrict expansion
5. Results (IDs + distances) returned to caller; scratch returned to pool

### 4. Searching a Disk Index (SSD)

**Modules**: infrastructure → distance → disk-index → apps/python/restapi

1. **disk-index** `PQFlashIndex::cached_beam_search()` is the core entry point
2. PQ-compressed vectors in RAM provide approximate distances to guide beam search
3. Promising candidates trigger sector reads from SSD via **infrastructure** aligned I/O (`AlignedFileReader`)
4. Full-precision distances computed from disk-resident vectors via **distance**
5. Node cache (`cache_bfs_levels()`) keeps hot nodes in RAM to reduce I/O
6. Results returned through the same interface as in-memory search

### 5. Dynamic Insert / Delete / Consolidate

**Modules**: infrastructure → distance → storage → index → apps/python

1. `Index<T>::insert_point()` adds a new vector: stores in **storage**, assigns a location (possibly reusing a deleted slot from `_empty_slots`), links into the graph via `prune_neighbors()`
2. `Index<T>::lazy_delete()` marks a point as deleted in `_delete_set` (O(1), no graph modification)
3. `Index<T>::consolidate_deletes()` compacts the data store, removes deleted nodes from all adjacency lists, and rebuilds the location-to-tag mapping

### 6. REST API Search Request Lifecycle

**Modules**: infrastructure → distance → disk-index/index → restapi

1. **restapi** `Server` receives a POST with JSON `{query: [...], k: N, Ls: M}`
2. `Server::parseJson<T>()` validates and extracts parameters (Ls defaults to DEFAULT_L=100)
3. `Server::handle_post<T>()` dispatches to the registered `Searcher<T>::search()`
4. For SSD: **disk-index** `PQFlashSearch<T>` wraps `PQFlashIndex::cached_beam_search()` with DEFAULT_W=1 as beam width
5. For multi-index: `aggregate_results()` performs K-way merge across searchers by minimum distance
6. Results serialized to JSON and returned in the HTTP response

### 7. Python Workflow: Build + Search + Dynamic Ops

**Modules**: python → index → disk-index → distance → storage → infrastructure

1. `diskannpy.build_memory_index()` calls C++ `diskann::build_memory_index<T>()` via pybind11 module.cpp
2. `StaticMemoryIndex` wraps `diskann::Index<T>` — `.search()` and `.batch_search()` return numpy arrays
3. `DynamicMemoryIndex` supports `.insert()`, `.mark_deleted()`, `.save()` (auto-consolidates if needed)
4. `StaticDiskIndex` wraps `PQFlashIndex<T>` — `.search()` and `.batch_search()` for SSD indices
5. All Python classes handle dtype dispatch (float32/int8/uint8) internally, store metadata in `_metadata.bin`
