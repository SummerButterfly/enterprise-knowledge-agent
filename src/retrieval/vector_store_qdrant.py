from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from config.settings import settings

class HybridRetriever:
    """混合检索器：向量（语义）+ BM25（关键词）"""

    def __init__(self):
        self.client = QdrantClient(url=settings.QDRANT_URL)
        self.embeddings = HuggingFaceBgeEmbeddings(model_name=settings.EMBED_MODEL)
        self.sparse = FastEmbedSparse(model_name="Qdrant/bm25")

        self.store = QdrantVectorStore(
            client=self.client,
            collection_name=settings.COLLECTION_NAME,
            embedding=self.embeddings,
            sparse_embedding=self.sparse,
            retrieval_mode=RetrievalMode.HYBRID,
        )

    def search(self, query: str, k: int = None):
        """混合检索，返回 [(Document, score), ...]"""
        k = k or settings.TOP_K
        return self.store.similarity_search_with_score(query, k=k)

    def search_pure_vector(self, query: str, k: int = None):
        """纯向量检索（对比用）"""
        k = k or settings.TOP_K
        vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=settings.COLLECTION_NAME,
            embedding=self.embeddings,
            retrieval_mode=RetrievalMode.DENSE,
        )
        return vector_store.similarity_search_with_score(query, k=k)