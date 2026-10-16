# Qdrant vs Milvus 面试题库（完整版）

---

## 目录

1. 项目背景
2. 高频必问（Q1-Q3）
3. 技术细节追问（Q4-Q8）
4. 实践细节追问（Q9-Q13）
5. 进阶追问（Q14-Q17）
6. 面试"杀手锏"（Q18-Q20）
7. 速记卡片
8. 面试时的一段话
9. 千万别说的
10. 附录：踩坑记录

---

## 一、项目背景

我项目主线用 **Milvus**，但**实际跑过 Qdrant**——用 Qdrant 做过 Day 1-2 的数据注入和混合检索，后来因为要模拟企业生产环境，切到 Milvus。

两个库都亲手用过、都踩过坑，面试能答得很实。

### 使用历程

| 阶段 | 用什么 | 原因 |
|------|-------|------|
| Day 1-2 | Qdrant | 部署简单，快速跑通 |
| Day 2 之后 | 切到 Milvus | 模拟企业生产环境 |

---

## 二、高频必问（100% 会问）

### Q1：你项目用的什么向量数据库？

**答**：
主线用 Milvus，因为我模拟的是企业生产场景。Milvus 是生产级向量库，支持十亿级向量，大厂用得多。

但我也实际跑过 Qdrant——用 Qdrant 做过 Day 1-2 的数据注入和混合检索，后来因为要模拟企业环境切到 Milvus。

**关键点**：两个都提，体现视野。

---

### Q2：Qdrant 和 Milvus 有什么区别？

| 维度 | Qdrant | Milvus |
|------|--------|--------|
| 开发语言 | Rust | Go + C++ |
| 部署 | 单容器 | 三容器（etcd + MinIO + Milvus） |
| 启动速度 | 5 秒 | 30 秒+ |
| 镜像大小 | ~200MB | ~1.5GB |
| 适合规模 | 中小（百万级） | 大（十亿级） |
| 混合检索 | 原生支持，一行配置 | 支持，但配置复杂 |
| 社区 | 中等 | 大厂背书，社区大 |
| 企业采用 | 中小厂 | 大厂多 |
| 运维难度 | 低 | 高 |

一句话：Qdrant 轻量好上手，Milvus 更适合大规模生产。

---

### Q3：为什么选 Milvus 而不是 Qdrant？

三点原因：

1. 我模拟的是**企业生产场景**，Milvus 在大厂用得多，更贴合真实环境。

2. **功能全**——支持多种索引、分区、多租户，适合企业级需求。

3. **生态成熟**——官方 pymilvus SDK 稳定，K8s 部署方案完善。

但我也知道 Qdrant 的优势：部署简单、混合检索原生支持。个人项目或中小规模，Qdrant 更划算。

---

## 三、技术细节追问（80% 会问）

### Q4：Milvus 的架构是什么？

Milvus 是**存算分离**架构，分四个组件：

- **Milvus 本体**：做向量检索
- **etcd**：存元数据（schema、索引配置）
- **MinIO**：存实际数据（向量、标量）
- **Pulsar**（集群版才用）：做消息队列

单机版（Standalone）三个容器就够：Milvus + etcd + MinIO。

数据流：

```
应用层 → Milvus 本体（检索）
          ↓ 元数据
        etcd
          ↓ 实际数据
        MinIO
```

---

### Q5：Qdrant 的架构是什么？

Qdrant 是**单体架构**，一个容器搞定所有：

- 元数据、数据、索引都在一个进程里
- 用 Rust 写，性能好
- 部署简单，docker run 一条命令就能起来

**核心特点**：没有外部依赖，一个二进制搞定所有。

---

### Q6：Milvus 的索引类型有哪些？

| 索引 | 特点 | 适合场景 |
|------|------|---------|
| FLAT | 暴力搜索，100% 准确 | 小数据集（万级） |
| IVF_FLAT | 聚类 + 倒排，快 | 中大数据集（百万级） |
| IVF_SQ8 | IVF + 8bit 量化，省内存 | 大数据集 |
| HNSW | 图索引，最快 | 高 QPS 场景 |
| IVF_PQ | 乘积量化，极致压缩 | 超大数据集 |

**我项目用的是 IVF_FLAT**，因为数据量小，精度优先。

**IVF_FLAT 原理**：
- 把所有向量聚成 `nlist` 个簇
- 检索时先找最近的几个簇（`nprobe`）
- 在簇内做暴力搜索

**我项目参数**：`nlist=128`，`nprobe=10`。

---

### Q7：Qdrant 的索引是什么？

Qdrant 默认用 **HNSW**（分层可导航小世界图），也支持量化索引。

HNSW 是图索引，检索时从上层图往下层逼近，速度快、精度高。

**HNSW 特点**：
- 多层图结构
- 上层稀疏（快速跳转），下层密集（精确匹配）
- 检索复杂度 O(log n)

---

### Q8：两库的相似度度量有哪些？

| 度量 | 含义 | 用途 |
|------|------|------|
| COSINE | 余弦相似度，看方向 | 文本向量（最常用） |
| L2 | 欧氏距离，看距离 | 图像向量 |
| IP | 内积 | 归一化后的向量 |

**我项目用 COSINE**，因为文本向量的"方向"代表语义。

**为什么文本用 COSINE 不用 L2**：因为向量归一化后，COSINE 和 IP 等价，且不受向量长度影响。

---

## 四、实践细节追问（60% 会问）

### Q9：Qdrant 的混合检索怎么做？

Qdrant 原生支持混合检索，一行配置：

```python
from langchain_qdrant import QdrantVectorStore, RetrievalMode

store = QdrantVectorStore(
    client=client,
    collection_name="enterprise_kb",
    embedding=embeddings,          # 稠密：BGE
    sparse_embedding=sparse,        # 稀疏：BM25
    retrieval_mode=RetrievalMode.HYBRID,
)
```

它底层自动做**双路召回**（稠密向量 + 稀疏向量），用 **RRF（倒数排名融合）** 合并。

存的时候两种向量一起存进 Collection，检索时自动跑两路。

---

### Q10：Milvus 的混合检索怎么做？

Milvus 也支持混合检索，但配置比 Qdrant 复杂：

1. schema 里要**单独声明**稠密向量字段和稀疏向量字段
2. 建索引时**两套参数**分别配
3. 检索时用 HybridSearch 或 RRFRanker

**示例**：

```python
from pymilvus import AnnSearchRequest, RRFRanker

# 稠密检索请求
dense_req = AnnSearchRequest(
    data=[dense_vector],
    anns_field="dense_vector",
    param={"metric_type": "COSINE"},
    limit=10,
)

# 稀疏检索请求
sparse_req = AnnSearchRequest(
    data=[sparse_vector],
    anns_field="sparse_vector",
    param={"metric_type": "IP"},
    limit=10,
)

# 融合
results = collection.hybrid_search(
    reqs=[dense_req, sparse_req],
    rerank=RRFRanker(),
    limit=3,
)
```

**我项目目前用的是纯稠密向量**（BGE），如果要做混合，会加 BM25 稀疏向量。

---

### Q11：两库的部署方式？

**Qdrant**：

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - ./qdrant_data:/qdrant/storage
```

一条命令：`docker compose up -d qdrant`

**Milvus**：

```yaml
services:
  etcd:
    image: quay.io/coreos/etcd:v3.5.5
    ...
  minio:
    image: minio/minio:latest
    ...
  milvus:
    image: milvusdb/milvus:v2.3.3
    ports:
      - "19530:19530"
    depends_on:
      - etcd
      - minio
```

三个容器，有依赖关系。

**我实际部署时**：
- Qdrant 5 秒起来
- Milvus 首次拉镜像 1.5GB，启动 30 秒
- Milvus 还踩了 MinIO 镜像的坑（Docker Hub 拒绝、Chainguard 镜像 command 格式不同）

**这些踩坑经历面试可讲。**

---

### Q12：数据怎么持久化？

**Qdrant**：

```yaml
volumes:
  - ./qdrant_data:/qdrant/storage
```

**Milvus**：三个容器都要挂载：

```yaml
volumes:
  - ./milvus_data/etcd:/etcd
  - ./milvus_data/minio:/minio_data
  - ./milvus_data/milvus:/var/lib/milvus
```

**Milvus 持久化更复杂**，因为数据分散在三个容器：
- etcd：存元数据
- MinIO：存实际数据
- Milvus：存索引缓存

**容器删了，数据还在**（因为挂载到宿主机目录）。

---

### Q13：你项目里怎么用 pymilvus 的？

我直接用 **pymilvus 官方 SDK**，没用 LangChain 封装。

**原因**：LangChain 的 `langchain-milvus` 依赖 `langchain-core 1.x`，和我项目的 `0.3.x` 冲突。

**实现**：

```python
from pymilvus import (
    connections, utility,
    Collection, CollectionSchema, FieldSchema, DataType
)

# 1. 连接
connections.connect(host="localhost", port="19530")

# 2. 定义 schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
]
schema = CollectionSchema(fields)
collection = Collection(name="enterprise_kb", schema=schema)

# 3. 建索引
collection.create_index(
    field_name="vector",
    index_params={
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 128},
    },
)

# 4. 插入
collection.insert([texts, sources, vectors])
collection.flush()
collection.load()

# 5. 检索
results = collection.search(
    data=[query_vector],
    anns_field="vector",
    param={"metric_type": "COSINE", "params": {"nprobe": 10}},
    limit=3,
    output_fields=["text", "source"],
)
```

**好处**：不依赖框架，可控性强，企业实际生产也用这种方式。

---

## 五、进阶追问（40% 会问）

### Q14：Milvus 的 Standalone 和 Cluster 区别？

| 模式 | 组件 | 适合场景 |
|------|------|---------|
| Standalone | Milvus + etcd + MinIO（单机） | 开发、测试、小规模 |
| Cluster | 加 Pulsar、多 Coordinator、多 Worker | 生产、大规模 |

**我项目用 Standalone**，因为数据量小。生产环境会升级到 Cluster，支持水平扩展。

**Cluster 的核心变化**：
- 计算和存储分离
- 支持水平扩展（加 Worker）
- 通过 Pulsar 做消息队列解耦

---

### Q15：Qdrant 支持分布式吗？

Qdrant 支持分布式，但相对 Milvus 起步晚。

- **单机版**：一个容器
- **分布式版**：多节点 + 分片 + 副本

**分布式特点**：
- 分片（Sharding）：把数据分到多个节点
- 副本（Replication）：每个分片有多个副本，高可用

**大规模生产用 Qdrant 分布式的案例相对少**，更多用 Milvus 或 Elasticsearch。

---

### Q16：如果数据量到 10 亿，你会选哪个？

**Milvus**。原因：

- Milvus 就是为十亿级设计的，Cluster 模式支持水平扩展
- 支持多种量化索引（IVF_PQ、SCANN），大幅压缩内存
- 存算分离，计算和存储独立扩展

Qdrant 也能扩，但十亿级场景 Milvus 更成熟。

**具体数据**：
- Milvus 单机能扛 1 亿+
- Cluster 能扛 100 亿+
- Qdrant 单机百万级，分布式千万级

---

### Q17：两库的延迟对比？

| 数据规模 | Qdrant | Milvus |
|---------|--------|--------|
| 小（万级） | 10-30ms | 10-30ms |
| 中（百万级） | 30-50ms | 50-100ms |
| 大（亿级） | 分布式略慢 | 分布式稳定 |

- **小数据集（万级）**：两者接近，都在 10-50ms
- **中数据集（百万级）**：Qdrant 略快（Rust 性能好）
- **大数据集（亿级）**：Milvus 更稳（分布式架构）

我项目里 14 条数据，检索延迟都在 50ms 内，差异不明显。

---

## 六、面试"杀手锏"问题

### Q18：你项目为什么从 Qdrant 切到 Milvus？

主要两个原因：

1. **模拟企业场景**。Milvus 在大厂用得多，我想让项目更贴近生产。

2. **想体验两个库的差异**。Qdrant 部署简单但混合检索封装太"黑盒"，Milvus 让我更能控制底层——schema、索引参数、检索参数都自己设计。

**迁移成本不高**：因为上层逻辑（检索接口）不变，只改了 `indexer.py` 和 `vector_store.py` 两个文件。

**切换后的收益**：
- 更懂底层（schema、索引、度量）
- 面试能讲选型逻辑
- 环境更接近企业

---

### Q19：你会推荐公司用哪个？

看场景：

| 场景 | 推荐 |
|------|------|
| 中小规模（<千万级）、快速迭代 | Qdrant |
| 大规模（亿级）、企业生产 | Milvus |
| 需要混合检索、不想配置 | Qdrant |
| 需要多租户、复杂权限 | Milvus |
| 追求部署简单 | Qdrant |
| 追求功能全 | Milvus |

**我项目选 Milvus** 是因为要模拟企业环境，但**个人项目我会选 Qdrant** 省事。

---

### Q20：如果面试官说"我们公司用 Qdrant"，你怎么接？

Qdrant 也是很好的选择。我在项目里**实际用过 Qdrant**，它的**混合检索原生支持**特别方便——一行 `RetrievalMode.HYBRID` 就搞定双路召回，底层自动 RRF 融合。部署也简单，一个容器。

我用 Milvus 是因为要对比两者，但 Qdrant 的开发体验确实好。如果你们用 Qdrant，我能快速上手。

**关键点**：不要贬低 Qdrant，要体现"两个都会"。

---

## 七、速记卡片（面试前扫一眼）

| 问题 | 核心答案 |
|------|---------|
| 用什么向量库 | Milvus 主线 + Qdrant 实操过 |
| 两库区别 | Qdrant 轻量，Milvus 企业级 |
| 为什么选 Milvus | 模拟企业场景 + 功能全 |
| Milvus 架构 | Milvus + etcd + MinIO |
| Qdrant 架构 | 单体 |
| 索引 | Milvus 用 IVF_FLAT，Qdrant 用 HNSW |
| 度量 | COSINE |
| 混合检索 | Qdrant 一行，Milvus 要配 |
| 为什么用 pymilvus | LangChain 冲突，SDK 更可控 |
| 大规模选哪个 | Milvus |
| 小规模选哪个 | Qdrant |

---

## 八、面试时的一段话（背诵版）

面试官问："你项目用的什么向量数据库？"

我主线用 **Milvus**，因为模拟的是企业生产场景。但我实际也跑过 **Qdrant**，用它做过 Day 1-2 的数据注入和混合检索。

两者主要区别：

- **部署上**，Qdrant 是单容器，Milvus 是三容器（etcd + MinIO + Milvus）
- **规模上**，Qdrant 适合百万级，Milvus 适合十亿级
- **混合检索上**，Qdrant 原生支持一行配置，Milvus 要单独配
- **性能上**，小数据集两者接近，大数据集 Milvus 更稳

我选 Milvus 是因为要模拟企业环境，但个人项目我会选 Qdrant——部署简单、开发体验好。两个库我都踩过坑，能快速上手任何一个。

---

## 九、千万别说的

| 别说 | 应该说 |
|------|--------|
| 我只用过 Milvus | Milvus 主线 + Qdrant 实操过 |
| 不知道 Qdrant 什么 | 两者架构、部署、规模都清楚 |
| 不知道为什么选 Milvus | 模拟企业环境 + 功能全 |
| 没用过 pymilvus | 直接用官方 SDK，可控性强 |
| Qdrant 不行 | 看场景，两者各有优势 |
| 两库差不多 | 架构、规模、生态差异明显 |
| 没用过就评价 | 两个都跑过、都踩过坑 |

---

## 十、附录：踩坑记录

### Milvus 部署踩的坑

| 坑 | 原因 | 解决 |
|----|------|------|
| MinIO 镜像拉取被拒 | Docker Hub 限制 | 换 Chainguard 镜像 |
| Chainguard 镜像启动失败 | command 格式不同 | 去掉开头的 `minio` |
| MinIO 显示 unhealthy | 镜像无 curl | 忽略，不影响使用 |

### 环境依赖冲突的坑

| 坑 | 原因 | 解决 |
|----|------|------|
| langchain-milvus 依赖冲突 | 需要 langchain-core 1.x | 改用 pymilvus SDK |
| LangChain 版本混乱 | 升级后 API 变了 | 锁死版本 |
| pymilvus 路径错误 | .venv 记录旧路径 | 重建虚拟环境 |

### 这些坑的价值

面试时可以讲：
- "我遇到过 LangChain 的版本依赖冲突，最后改用官方 SDK 解决"
- "Milvus 部署比 Qdrant 复杂，我踩了 MinIO 镜像的坑"
- "环境迁移时 .venv 的路径问题，我通过重建虚拟环境解决"

**每个坑都是面试素材。**

---

## 十一、总结

面试官问 Qdrant vs Milvus，本质是考：

1. 你**用过几个**向量库
2. 你**懂不懂**各自的架构和特点
3. 你**能不能**讲清选型逻辑
4. 你**踩过哪些坑**

**你两个都跑过、都踩过坑，这段能答 20 分钟。**