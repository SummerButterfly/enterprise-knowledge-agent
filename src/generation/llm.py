from openai import OpenAI
from config.settings import settings


class LLMClient:
    """封装 DeepSeek 大模型调用"""

    def __init__(self):
        self.client = OpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL,
        )
        self.model = settings.LLM_MODEL

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        """生成答案"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content