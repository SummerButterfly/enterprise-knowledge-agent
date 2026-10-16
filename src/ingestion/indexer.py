from pymilvus import (
    connections, utility,
    Collection, CollectionSchema, FieldSchema, DataType
)
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from src.ingestion.loader import load_document
from src.ingestion.splitter import split_document
from config.settings import settings

COLLECTION_NAME = settings.COLLECTION_NAME
DIM = 768


def build_index(file_path: str, doc_type: str = "enterprise"):
    connections.connect(host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)

    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
        print(f"已删除旧集合：{COLLECTION_NAME}")

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=DIM),
    ]
    schema = CollectionSchema(fields, description="企业知识库")
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    print(f"已创建集合：{COLLECTION_NAME}")

    collection.create_index(
        field_name="vector",
        index_params={
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 128},
        },
    )

    embeddings = HuggingFaceBgeEmbeddings(model_name=settings.EMBED_MODEL)

    text = load_document(file_path)
    chunks = split_document(text, doc_type)
    print(f"切出 {len(chunks)} 个块")

    vectors = embeddings.embed_documents(chunks)

    collection.insert([
        chunks,
        [file_path] * len(chunks),
        vectors,
    ])
    collection.flush()
    collection.load()
    print(f"已写入 Milvus：{COLLECTION_NAME}，共 {collection.num_entities} 条")

    return chunks


if __name__ == "__main__":
    build_index(settings.PDF_PATH)