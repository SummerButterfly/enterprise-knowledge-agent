import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # 路径
    PDF_PATH = os.getenv("PDF_PATH", "data/raw/test.pdf")
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    COLLECTION_NAME = "enterprise_kb"

    # 模型
    EMBED_MODEL = "BAAI/bge-base-zh-v1.5"
    LLM_MODEL = "deepseek-chat"

    # 切块
    CHUNK_SIZE = 300
    CHUNK_OVERLAP = 50

    # 检索
    TOP_K = 10
    RERANK_TOP_K = 3

    # 密钥
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_BASE_URL = "https://api.deepseek.com"

settings = Settings()