from pymilvus import connections, Collection
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from config.settings import settings


class HybridRetriever:
    """Milvus 版检索器（用 pymilvus SDK）"""

    def __init__(self):
        connections.connect(host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)
        self.collection = Collection(settings.COLLECTION_NAME)
        self.collection.load()
        self.embeddings = HuggingFaceBgeEmbeddings(model_name=settings.EMBED_MODEL)

    def search(self, query: str, k: int = None):
        k = k or settings.TOP_K
        query_vector = self.embeddings.embed_query(query)

        results = self.collection.search(
            data=[query_vector],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=k,
            output_fields=["text", "source"],
        )

        output = []
        for hit in results[0]:
            output.append({
                "text": hit.entity.get("text"),
                "score": hit.distance,
                "source": hit.entity.get("source"),
            })
        return output