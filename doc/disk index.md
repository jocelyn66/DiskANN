Q:
- section 4B对齐是不是和8B页面很类似里, section怎么路由的?

lsm run:

-----

prompt: 我想了解磁盘模式的存储格式设计细节, 作为AI数据湖项目里的lsm run粒度向量索引的设计参考

## 🔍 DiskANN 磁盘索引存储格式——全景

DiskANN 构建磁盘索引时会生成 **一个核心文件** `_disk.index`，加上若干辅助文件。核心文件的结构如下：

```
┌───────────────────────────────────────────────────────┐
│  Sector 0: METADATA  (4096 bytes)                     │
│    npts, ndims, medoid, max_node_len,                 │
│    nnodes_per_sector, frozen_info, reorder_info...    │
├───────────────────────────────────────────────────────┤
│  Sectors 1..N: GRAPH + COORDINATES (交织存储)          │
│    每个 node = [T coords[ndims]] [uint32 nnbrs]       │
│                 [uint32 nbr_ids[nnbrs]]               │
│    多个 node 打包进一个 sector，或一个 node 跨多个 sector │
├───────────────────────────────────────────────────────┤
│  (可选) Sectors N+1..M: REORDER DATA                  │
│    全精度浮点向量，用于 PQ 搜索后重排                     │
└───────────────────────────────────────────────────────┘
```


关键设计点：
- **SECTOR_LEN = 4096 字节**，所有 I/O 按 sector 对齐
- **向量坐标 + 图邻居列表 交织存储在同一个 sector 里**（不是分开存）
- 两种寻址模式：小节点 → 多节点挤进一个 sector；大节点 → 一个节点跨多个 sector
- 搜索时用 **Linux AIO 批量读取**，每个线程预分配 128 × 4096 = 512KB scratch buffer

辅助文件包括 PQ 码本、PQ 压缩向量（常驻内存）、medoid 入口点、warmup 采样数据等。

todo: 
- what: warmup 采样数据

why design:
1. 一次读取全精度和邻居量化. 全精度向量算一次精确距离，放full_retset。最终返回给用户的 top-k 结果全部来自 full_retset，用的是精确距离排序。节省全精读重排.

---

prompt: 题1中的向量是原始向量还是量化后的向量

**磁盘 sector 里存的是原始向量（全精度），不是量化后的向量。**

证据在 disk_utils.cpp：

```cpp
// write coords of node first
base_reader.read((char *)cur_node_coords.get(), sizeof(T) * ndims_64);
memcpy(node_buf.get(), cur_node_coords.get(), ndims_64 * sizeof(T));
```

它直接从 `base_file`（原始向量文件）读取 `T[ndims]` 写入 node buffer，`T` 就是 `float`/`int8_t`/`uint8_t`。

DiskANN 的磁盘格式实际上是 **双轨设计**：

| 存储位置 | 内容 | 精度 | 用途 |
|---------|------|------|------|
| `_pq_compressed.bin` | PQ 压缩向量 | 量化 (uint8) | **常驻内存**，beam search 中做近似距离排序, 对邻居节点做 PQ 近似距离计算，决定哪些节点值得发起磁盘 I/O 去读取。 |
| `_disk.index` sectors | 原始向量 + 图邻居 | 全精度 (T) | **磁盘上**，按需读取做精确距离计算 |
| (可选) reorder data | 全精度 float 向量 | float | 磁盘上，用于 disk PQ 模式下的重排 |

PQ 量化向量（内存）	全精度向量（磁盘）
存储	_pq_compressed.bin，搜索时全量加载到 RAM	_disk.index 的每个 sector
作用	图遍历中对未访问邻居做近似距离排序，决定下一步读哪些节点	对已读回的节点做精确距离计算
影响	遍历方向和效率（I/O 模式）	最终返回结果的距离精度
调用时机	每次读回一个节点后，对其所有邻居批量 PQ 距离计算	每次磁盘 I/O 读回 sector 后，对该节点做一次精确计算

搜索流程是：
1. 用**内存中的 PQ 压缩向量**快速估算距离，筛选候选
2. 对候选节点从磁盘读 sector，拿到**原始向量**做精确距离计算
3. 同时从 sector 里拿到**图邻居列表**，继续 beam search

beam search:
- 只有在候选队列前 beam_width 名的节点，才会在下一轮被选中发起磁盘 I/O。

参数: 
- beam_width

---

变体：
当启用 use_reorder_data 时（pq_flash_index.cpp:1623），还会额外从 reorder 区域读取全精度 float 向量做二次重排，这是给 _use_disk_index_pq 模式准备的——磁盘上存的也是 PQ 压缩的，需要再读全精度数据来修正。

---

prompt: _use_disk_index_pq 模式的目的是什么

jo. 内存pq与磁盘pq可独立, 内存pq受内存规格约束

两种模式对比
默认模式（不启用 disk PQ）：
````
磁盘 sector: [float 向量 (ndims × 4B)] + [图邻居列表]
内存: [PQ 压缩向量 (num_chunks × 1B)]
````
Disk PQ 模式（_use_disk_index_pq = true）：
`````
This is the code block that represents the suggested code change:
```markdown
磁盘 sector: [PQ 压缩向量 (disk_pq_chunks × 1B)] + [图邻居列表]
内存: [PQ 压缩向量 (num_chunks × 1B)]  ← 这个还是在
```

```markdown
if (!use_disk_pq)
{
    diskann::create_disk_layout<T>(data_file_to_use, mem_index_path, disk_index_path);
}
else
{
    // 注意模板参数变成了 uint8_t，输入变成了 PQ 压缩后的数据
    diskann::create_disk_layout<uint8_t>(disk_pq_compressed_vectors_path, mem_index_path, disk_index_path);
}
```

为什么需要这个模式？
磁盘空间和 I/O 带宽优化。
128 维 float 向量	disk PQ 压缩到 32 字节
每个节点向量大小	512 字节	*32* 字节
加上图邻居（64 度）	512 + 260 = 772 字节	32 + 260 = 292 字节
每 sector (4096B) 能放几个节点	5 个	14 个
节点变小了，每次 I/O 读回更多节点，或者同样数据量需要更少磁盘空间。

总结：三种配置的权衡
配置	磁盘存什么	搜索时距离精度	磁盘空间	I/O 次数
默认	全精度向量 + 图	精确	大	标准
disk PQ	PQ 向量 + 图	近似	小	更少（节点更密）
disk PQ + reorder	PQ 向量 + 图 + 全精度向量（尾部）	最终精确	更大（但图区域更紧凑）	多一轮重排 I/O
本质上是一个经典的 空间-精度-I/O 三角权衡。

prompt: 重排模式, pq的作用是什么, 内存里的pq是直接加载辅助文件里的吧

重排模式（disk PQ + reorder）的完整搜索流程有 三层距离计算：

阶段	用什么	距离精度	目的
① 图遍历选邻居	内存 PQ（_pq_table）	粗略近似	决定哪些邻居值得发 I/O
② 读回节点后打分	磁盘 PQ（_disk_pq_table）	较好近似	对已访问节点排序，构建候选集 full_retset
③ 最终重排	reorder 区的全精度向量	精确	对 top 候选做最终精确排序
所以在重排模式下，有两套 PQ，各管一件事：

内存 PQ（_pq_table）：导航——给图遍历选方向
磁盘 PQ（_disk_pq_table）：打分——给已访问节点算一个比内存 PQ 可能更精细的近似距离
最终靠 reorder 的全精度向量兜底。

两套 PQ 可以不同
值得注意的是：内存 PQ 的 chunk 数由 search_DRAM_budget 推算（越少越省内存），磁盘 PQ 的 chunk 数由 disk_pq_dims 参数独立指定。两者的精度和压缩率可以独立调节——内存那套更粗（省 RAM），磁盘那套可以更细（反正存磁盘不占内存）。

