# Day 3：Rerank 精排（完整版）

---

## 一、Day 3 目标

**给检索加一层"精排"，把 Milvus 粗排的 Top10 用 Cross-Encoder 重排为 Top3，提升准确率。**

一句话：**粗排负责"快"，精排负责"准"，两阶段配合达到生产级 RAG 标准。**

---

## 二、整体流程图

```
用户提问："年假怎么算？"
   ↓ ① Milvus 粗排（快）
Top 10 候选文档
   ↓ ② Cross-Encoder 精排（准）
Top 3 精准文档
   ↓ ③ 后续大模型生成
最终答案
```

**为什么需要两阶段？**
- **粗排（向量检索）**：速度快，但只看"语义大致相似"，容易混入无关内容。
- **精排（Rerank）**：模型逐个"精读"，速度慢，但判断极其精准。
- **粗排取 Top10 → 精排取 Top3**：既保证了速度，又保证了准确率。

---

## 三、环境确认

```powershell
# 1. 确认在项目目录
cd D:\projects2026_10_15\enterprise-knowledge-agent

# 2. 确认 Milvus 在跑
docker ps

# 3. 确认修复了之前的依赖冲突
python -c "import sentence_transformers; print('OK')"
```
**全部通过才能继续。**

---

## 四、Day 3 涉及的 3 个文件

### 1. `src/retrieval/reranker.py`（新增）

```python
from sentence_transformers import CrossEncoder
from config.settings import settings


class Reranker:
    """Cross-Encoder 精排器（基于 sentence-transformers）"""

    def __init__(self):
        # BAAI/bge-reranker-base 是 BGE 官方 Reranker 模型
        self.model = CrossEncoder("BAAI/bge-reranker-base")

    def rerank(self, query: str, candidates: list, top_k: int = None):
        """输入候选列表，计算精准相关性，返回排序后的 Top-K"""
        top_k = top_k or settings.RERANK_TOP_K

        if not candidates:
            return []

        # 构造 (query, text) 对
        pairs = [[query, c["text"]] for c in candidates]

        # 计算相关性分数
        scores = self.model.predict(pairs)

        # 把分数加到候选上
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)

        # 按精排分数重新排序
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

        return reranked[:top_k]
```

**关键点**：
- 使用 `CrossEncoder`，它会把 `query` 和 `text` 拼在一起送入模型打分。
- 打分通过 `predict(pairs)` 批量执行，效率高。
- 将打分结果写入 `c["rerank_score"]`，然后按该分数重新排序。

---

### 2. `src/retrieval/vector_store.py`（修改）

```python
from pymilvus import connections, Collection
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from src.retrieval.reranker import Reranker
from config.settings import settings


class HybridRetriever:
    """Milvus 版检索器 + Rerank 精排"""

    def __init__(self):
        # 1. 连接 Milvus
        connections.connect(host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)
        self.collection = Collection(settings.COLLECTION_NAME)
        self.collection.load()
        
        # 2. 加载 BGE 向量模型
        self.embeddings = HuggingFaceBgeEmbeddings(model_name=settings.EMBED_MODEL)
        
        # 3. 加载 Rerank 模型
        self.reranker = Reranker()

    def search(self, query: str, k: int = None, use_rerank: bool = True):
        """检索：1. Milvus 粗排 → 2. Rerank 精排"""
        k = k or settings.TOP_K
        query_vector = self.embeddings.embed_query(query)

        # ========== 第一步：Milvus 粗排 ==========
        results = self.collection.search(
            data=[query_vector],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=k,
            output_fields=["text", "source"],
        )

        candidates = []
        for hit in results[0]:
            candidates.append({
                "text": hit.entity.get("text"),
                "score": hit.distance,
                "source": hit.entity.get("source"),
            })

        # ========== 第二步：Rerank 精排 ==========
        if use_rerank:
            return self.reranker.rerank(query, candidates)
        
        return candidates[:settings.RERANK_TOP_K]
```

**关键点**：
- `use_rerank=True`（默认）时，执行完整的"粗排 + 精排"。
- `use_rerank=False` 时，只做粗排（方便测试对比）。
- 粗排调用 `k=10`，精排返回 Top3（`RERANK_TOP_K`）。

---

### 3. `tests/test_retrieval.py`（修改）

```python
from src.retrieval.vector_store import HybridRetriever


def main():
    retriever = HybridRetriever()

    queries = [
        "年假怎么算？",
        "报销需要什么材料？",
        "上班时间是什么时候？",
        "满3年工龄有多少年假？",
    ]

    for q in queries:
        print(f"\n{'='*70}")
        print(f"问题：{q}")
        print('='*70)

        # 1. 不 Rerank（粗排）
        print("\n【粗排（Milvus Top3）】")
        no_rerank = retriever.search(q, k=10, use_rerank=False)
        for i, r in enumerate(no_rerank[:3], 1):
            text = r["text"].replace("\n", " ")[:60]
            print(f"  {i}. [score={r['score']:.4f}] {text}...")

        # 2. 加 Rerank（精排）
        print("\n【精排（Rerank Top3）】")
        reranked = retriever.search(q, k=10, use_rerank=True)
        for i, r in enumerate(reranked, 1):
            text = r["text"].replace("\n", " ")[:60]
            print(f"  {i}. [rerank={r['rerank_score']:.4f}] {text}...")


if __name__ == "__main__":
    main()
```

**关键点**：
- 对比输出"粗排分数"和"精排分数"。
- 精排后 Top1 的分数通常会比粗排的更高、更明显（因为 Cross-Encoder 打分非常准）。

---

## 五、运行命令

```powershell
python -m tests.test_retrieval
```

**首次运行会下载 `bge-reranker-base` 模型**（约 1.1GB），配了 `HF_ENDPOINT` 镜像的会快一些，耐心等几分钟。

---

## 六、验证结果

**预期输出**：
```
======================================================================
问题：年假怎么算？
======================================================================

【粗排（Milvus Top3）】
  1. [score=0.8521] 第五条 年假按工龄计算...
  2. [score=0.7210] ...
  3. [score=0.6543] ...

【精排（Rerank Top3）】
  1. [rerank=0.9821] 第五条 年假按工龄计算...
  2. [rerank=0.4512] ...
  3. [rerank=0.1234] ...
```

**观察点**：
- 精排后 Top1 的分数极高（通常 > 0.9），说明 Cross-Encoder 极其确信。
- 无关候选的分数极低（可能 < 0.1）。

---

## 七、Day 3 踩过的坑（极其重要）

| 坑 | 原因 | 解决 |
|----|------|------|
| **`pyarrow requires NumPy 2.0`** | `pyarrow` 26.0 强制要求 NumPy 2.0，但项目基于 1.26.4 编译（PyTorch 等依赖 1.x） | **千万不能升级 NumPy！** 执行 `pip install "pyarrow<26"` 降级 |
| **`FlagEmbedding` 启动卡死/依赖重** | `FlagEmbedding` 默认拉取 `datasets`、`pyarrow` 等重型微调依赖 | **放弃 `FlagEmbedding`**，改用轻量的 `sentence-transformers`（代码见上） |
| **Reranker 下载极慢** | 没配 HuggingFace 国内镜像 | 确保 `.env` 有 `HF_ENDPOINT=https://hf-mirror.com`，且重启了 PyCharm |
| **CPU 跑 Rerank 很慢** | 默认没用 GPU，或者显存不够 | 能用 GPU 用 GPU，没 GPU 就 `use_fp16=True` 或耐心等 |
| **Top1 分数跟粗排一样** | 没调用 Rerank，或候选列表为空 | 检查 `use_rerank=True`，检查 Milvus 里有没有数据 |

**面试时可讲**："我遇到 `pyarrow` 和 NumPy 的版本冲突导致精排模型导入失败。我没有盲目升级 NumPy（那会导致 PyTorch 崩溃），而是通过降级 `pyarrow<26` 并替换轻量依赖解决了问题。"

---

## 八、核心概念速记

| 概念 | 一句话 |
|------|--------|
| **Rerank** | 对粗排结果做二次打分排序 |
| **Cross-Encoder** | 把 query 和 doc 拼在一起计算相似度，极其精准但慢 |
| **Bi-Encoder** | 分别把 query 和 doc 变向量计算相似度（Milvus 用的就是这个），快但略粗糙 |
| **粗排 + 精排** | 两阶段检索架构，生产级 RAG 标配 |
| **bge-reranker-base** | BAAI 开源的 Rerank 模型，效果极佳 |

---

## 九、Day 3 完成后的项目状态

```
enterprise-knowledge-agent/
├── src/
│   ├── ingestion/
│   │   └── indexer.py      ✅
│   └── retrieval/
│       ├── vector_store.py ✅ 已集成 Rerank
│       ├── reranker.py     ✅ Day 3 新增
│       └── vector_store_qdrant.py ✅ 保留
├── tests/
│   └── test_retrieval.py   ✅ 已支持对比测试
└── docs/
    ├── day1.md
    ├── day2.md
    └── day3.md             ✅ 本文档
```

---

## 十、Day 4 预告

**主题：LangGraph 基础**

1. 定义 `AgentState`
2. 写 `retrieve_node` 和 `generate_node`
3. 串成状态机
4. 跑通"检索 → 生成"

**目标**：把前面写的检索代码，接入大模型，输出最终答案。

---

## 十一、面试可讲点

| 面试问题 | 你怎么答 |
|---------|---------|
| 为什么要 Rerank？ | 粗排快但不准，精排准但慢。两阶段配合，取长补短 |
| Rerank 模型怎么选？ | `bge-reranker-base`，BGE 官方，中英文效果好 |
| Cross-Encoder 和 Bi-Encoder 区别？ | Bi-Encoder 分别编码求相似度，快；Cross-Encoder 交叉编码打分，准但慢 |
| Rerank 延迟多少？ | 10 个候选在 CPU 上大概 100-200ms，GPU 上更快 |
| 踩过什么坑？ | `pyarrow` 和 NumPy 冲突，降级解决；放弃 `FlagEmbedding` 换轻量依赖 |