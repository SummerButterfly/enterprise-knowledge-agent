from sentence_transformers import CrossEncoder
from config.settings import settings


class Reranker:
    """Cross-Encoder 精排器（基于 sentence-transformers）"""

    def __init__(self):
        self.model = CrossEncoder("BAAI/bge-reranker-base")

    def rerank(self, query: str, candidates: list, top_k: int = None):
        top_k = top_k or settings.RERANK_TOP_K

        if not candidates:
            return []

        pairs = [[query, c["text"]] for c in candidates]
        scores = self.model.predict(pairs)

        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]