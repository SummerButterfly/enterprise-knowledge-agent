from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from src.ingestion.loader import load_document
from src.ingestion.splitter import split_document
from config.settings import settings

def build_index(file_path: str, doc_type: str = "enterprise"):
    # 1. 连接 Qdrant
    client = QdrantClient(url=settings.QDRANT_URL)

    # 2. 加载模型
    embeddings = HuggingFaceBgeEmbeddings(model_name=settings.EMBED_MODEL)
    sparse = FastEmbedSparse(model_name="Qdrant/bm25")

    # 3. 读文档 + 切块
    text = load_document(file_path)
    chunks = split_document(text, doc_type)
    print(f"切出 {len(chunks)} 个块")

    # 4. 先删旧集合（避免重复累积）
    try:
        client.delete_collection(settings.COLLECTION_NAME)
        print(f"已删除旧集合：{settings.COLLECTION_NAME}")
    except Exception:
        pass

    # 5. 用 from_texts 自动建 collection + 写入
    store = QdrantVectorStore.from_texts(
        texts=chunks,
        embedding=embeddings,
        sparse_embedding=sparse,
        retrieval_mode=RetrievalMode.HYBRID,
        url=settings.QDRANT_URL,
        collection_name=settings.COLLECTION_NAME,
    )
    print(f"已写入 Qdrant：{settings.COLLECTION_NAME}")
    return chunks

if __name__ == "__main__":
    build_index(settings.PDF_PATH)