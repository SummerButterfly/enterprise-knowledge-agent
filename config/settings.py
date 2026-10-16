import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PDF_PATH = os.getenv("PDF_PATH", "data/raw/test.pdf")

    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
    COLLECTION_NAME = "enterprise_kb"

    EMBED_MODEL = "BAAI/bge-base-zh-v1.5"
    LLM_MODEL = "deepseek-chat"

    CHUNK_SIZE = 300
    CHUNK_OVERLAP = 50

    TOP_K = 10
    RERANK_TOP_K = 3

    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_BASE_URL = "https://api.deepseek.com"

settings = Settings()