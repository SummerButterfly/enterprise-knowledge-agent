from pymilvus import connections, Collection
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from src.retrieval.reranker import Reranker
from config.settings import settings


class HybridRetriever:
    """Milvus 版检索器 + Rerank 精排"""

    def __init__(self):
        # 1. 连接 Milvus 并加载集合
        connections.connect(host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)
        self.collection = Collection(settings.COLLECTION_NAME)
        self.collection.load()

        # 2. 加载 BGE 向量模型（用于把查询变成向量）
        self.embeddings = HuggingFaceBgeEmbeddings(model_name=settings.EMBED_MODEL)

        # 3. 加载 Rerank 模型（用于精排）
        self.reranker = Reranker()

    def search(self, query: str, k: int = None, use_rerank: bool = True):
        """
        检索流程：
        1. Milvus 粗排（召回 Top10 候选）
        2. Rerank 精排（从中挑出 Top3）
        """
        # 默认取 TOP_K（比如 10 个）去粗排
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

        # 把结果转成字典列表
        candidates = []
        for hit in results[0]:
            candidates.append({
                "text": hit.entity.get("text"),
                "score": hit.distance,
                "source": hit.entity.get("source"),
            })

        # ========== 第二步：Rerank 精排 ==========
        if use_rerank:
            # 交给 Reranker 去计算精准相关性，并返回 Top3
            return self.reranker.rerank(query, candidates)

        # 如果不需要精排（测试对比用），直接返回粗排的 Top3
        return candidates[:settings.RERANK_TOP_K]