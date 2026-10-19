from langchain_openai import ChatOpenAI
from config.settings import settings

def get_llm(temperature: float = 0.0):
    """获取支持工具调用的 LangChain LLM 客户端"""
    return ChatOpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.DEEPSEEK_API_KEY,
        base_url=settings.DEEPSEEK_BASE_URL,
        temperature=temperature,
    )