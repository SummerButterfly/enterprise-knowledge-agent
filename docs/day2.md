# Day 2：Milvus 版检索（完整版）

---

## 一、Day 2 目标

**用代码从 Milvus 里检索出最相关的文档块。**

一句话：问题 → 向量化 → Milvus 检索 → 返回 Top-K。

---

## 二、整体流程图

```
用户提问："年假怎么算？"
   ↓ ① 向量化（BGE）
查询向量（768 维）
   ↓ ② Milvus 检索（COSINE）
Top-K 相似向量
   ↓ ③ 返回原文
文档块列表
```

---

## 三、环境确认

```powershell
# 1. 确认 Milvus 在跑
docker ps
```
**预期**：3 个容器在 `Up`（`milvus-standalone`、`milvus-minio`、`milvus-etcd`）。注意 `milvus-minio` 显示 `unhealthy` 是正常的（健康检查脚本缺少 curl），不影响使用。

```powershell
# 2. 确认数据在
python -c "from pymilvus import connections, utility; connections.connect(host='localhost', port='19530'); print(utility.list_collections())"
```
**预期**：`['enterprise_kb']`

---

## 四、Day 2 涉及的 2 个文件

### 1. `src/retrieval/vector_store.py`（Milvus 版）

```python
from pymilvus import connections, Collection
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from config.settings import settings


class HybridRetriever:
    """Milvus 版检索器（用 pymilvus SDK）"""

    def __init__(self):
        # 1. 连接 Milvus
        connections.connect(host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)
        
        # 2. 加载集合到内存
        self.collection = Collection(settings.COLLECTION_NAME)
        self.collection.load()
        
        # 3. 加载 BGE 向量模型
        self.embeddings = HuggingFaceBgeEmbeddings(model_name=settings.EMBED_MODEL)

    def search(self, query: str, k: int = None):
        """检索 Top-K，返回 [{text, score, source}, ...]"""
        k = k or settings.TOP_K
        
        # 1. 把查询转成向量
        query_vector = self.embeddings.embed_query(query)

        # 2. Milvus 检索
        results = self.collection.search(
            data=[query_vector],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=k,
            output_fields=["text", "source"],
        )

        # 3. 格式化结果
        output = []
        for hit in results[0]:
            output.append({
                "text": hit.entity.get("text"),
                "score": hit.distance,
                "source": hit.entity.get("source"),
            })
        return output
```

**关键点**：
- 用 `pymilvus` 原生 API，不依赖 LangChain 的 Milvus 封装。
- `collection.load()` 检索前必须执行，否则报错 `collection not loaded`。
- `metric_type="COSINE"` 使用余弦相似度，对文本向量最合适。
- `output_fields=["text", "source"]` 指定需要返回的额外字段。

---

### 2. `tests/test_retrieval.py`

```python
from src.retrieval.vector_store import HybridRetriever


def main():
    retriever = HybridRetriever()

    queries = [
        "年假怎么算？",
        "报销需要什么材料？",
        "上班时间是什么时候？",
    ]

    for q in queries:
        print(f"\n{'='*60}")
        print(f"问题：{q}")
        print('='*60)

        results = retriever.search(q, k=3)
        for i, r in enumerate(results, 1):
            text = r["text"].replace("\n", " ")[:80]
            print(f"  {i}. [score={r['score']:.4f}] {text}...")


if __name__ == "__main__":
    main()
```

**关键点**：
- 返回是字典列表，用 `r["text"]`、`r["score"]`。
- 不再用 LangChain 的 `(doc, score)` 元组。

---

## 五、运行命令

```powershell
python -m tests.test_retrieval
```

**为什么加 `-m`**：`test_retrieval.py` 里有 `from src.retrieval...`，必须从项目根目录找包。

**实际执行顺序**：
```
test_retrieval.py 被运行
   ↓
初始化 HybridRetriever
   ↓
连接 Milvus → 加载 collection 到内存 → 加载 BGE 模型
   ↓
遍历 3 个问题
   ↓
每个问题：向量化 → Milvus 检索 → 打印 Top3
```

---

## 六、验证结果

### 方式 1：看命令行输出
```
============================================================
问题：年假怎么算？
============================================================
  1. [score=0.85] 第五条 年假按工龄计算：满 1 年 5 天...
  2. [score=0.72] ...
  3. [score=0.65] ...
```
**Top1 相关就对了。**

### 方式 2：命令行验证数据条数
```powershell
python -c "from pymilvus import connections, Collection; connections.connect(host='localhost', port='19530'); c = Collection('enterprise_kb'); c.load(); print('条数:', c.num_entities)"
```
**预期**：`条数: 14`

---

## 七、Day 2 踩过的坑（重点）

| 坑 | 原因 | 解决 |
|----|------|------|
| **`pymilvus` 未安装** | 依赖没装 | `pip install pymilvus -i https://pypi.tuna.tsinghua.edu.cn/simple` |
| **`Connection refused 19530`** | Milvus 没启动，或 Docker Desktop 没开 | 先启动 Docker Desktop，再 `docker compose up -d` |
| **`Collection not found`** | 还没建索引 | 先跑 `python -m src.ingestion.indexer` |
| **满屏 `DeprecationWarning`** | PyMilvus 旧 API 警告 | **忽略**（目前能用），未来可换 `MilvusClient` |
| **`.venv` 路径错误（Fatal error in launcher）** | 项目复制到新路径后，旧 `.venv` 记录的是旧路径 | 删掉旧 `.venv`，重新 `python -m venv .venv` |
| **找不到路径** | 在错误的目录（比如桌面）执行命令 | `cd D:\projects2026_10_15\enterprise-knowledge-agent` |
| **`python` 命令找不到** | 没激活虚拟环境 | 执行 `.venv\Scripts\activate`，或检查系统环境变量 |

---

## 八、核心概念速记

| 概念 | 一句话 |
|------|--------|
| **Milvus 检索** | 给查询向量，找最相似的 Top-K |
| **COSINE** | 余弦相似度，看向量"方向" |
| **nprobe** | 搜索多少个簇（IVF_FLAT 索引参数） |
| **output_fields** | 检索时额外返回的字段 |
| **collection.load()** | 把集合加载到内存才能检索 |
| **HybridRetriever** | 封装检索的类 |

---

## 九、Day 2 完成后的项目状态

```
enterprise-knowledge-agent/
├── src/
│   ├── ingestion/
│   │   ├── loader.py       ✅
│   │   ├── splitter.py     ✅
│   │   └── indexer.py      ✅ Milvus 版
│   └── retrieval/
│       ├── __init__.py     ✅
│       ├── vector_store.py ✅ Milvus 版（Day 2 完成）
│       └── vector_store_qdrant.py  ✅ 保留
├── tests/
│   ├── __init__.py         ✅
│   └── test_retrieval.py   ✅ Milvus 版
├── docs/
│   ├── day1.md             ✅
│   └── day2.md             ✅ 本文档
└── ...
```

---

## 十、Day 3 预告

**主题：Rerank 精排**

1. 加 Cross-Encoder 模型（`bge-reranker-base`）
2. 粗排 Top10 → 精排 Top3
3. 对比"有 Rerank"和"没 Rerank"的差异

**目标**：Top1 准确率进一步提升。

---

## 十一、面试可讲点

| 面试问题 | 你怎么答 |
|---------|---------|
| 检索怎么做的？ | 用 pymilvus SDK，collection.load 后 search |
| 用什么模型？ | BGE-base-zh-v1.5，768 维 |
| 相似度怎么算？ | COSINE，余弦相似度 |
| 为什么不用 LangChain 封装？ | 版本冲突，SDK 更可控 |
| 检索返回什么？ | 字典列表，含 text、score、source |
| 踩过什么坑？ | pymilvus 路径错误、Connection refused、依赖冲突 |

---

## 十二、和 Qdrant 版的区别

| 对比 | Qdrant 版[day2.md](day2.md)（Day 2） | Milvus 版（当前） |
|------|-------------------|------------------|
| 检索器类 | `QdrantVectorStore` | `pymilvus.Collection` |
| 返回格式 | `(Document, score)` 元组 | `{text, score, source}` 字典 |
| 索引类型 | HNSW（默认） | IVF_FLAT |
| 加载 | 自动 | 手动 `load()` |
| 代码量 | 少（封装好） | 多（自己控） |

**Qdrant 方便，Milvus 可控。**