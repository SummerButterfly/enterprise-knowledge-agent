from typing import TypedDict, Annotated
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from src.retrieval.vector_store import HybridRetriever
from src.generation.llm import LLMClient


# ========== 1. 定义状态 ==========
class AgentState(TypedDict):
    """状态机中传递的数据"""
    messages: Annotated[list[BaseMessage], operator.add]  # 对话历史（累加）
    query: str                                            # 当前查询
    context: str                                          # 检索到的上下文


# ========== 2. 初始化组件（全局复用） ==========
retriever = HybridRetriever()
llm = LLMClient()


# ========== 3. 定义节点 ==========

def retrieve_node(state: AgentState) -> dict:
    """检索节点：调用 Rerank 检索"""
    query = state["query"]
    results = retriever.search(query, k=10, use_rerank=True)

    # 拼接 Top3 上下文
    context = "\n\n".join([r["text"] for r in results])

    return {"context": context}


def generate_node(state: AgentState) -> dict:
    """生成节点：调用大模型生成答案"""
    query = state["query"]
    context = state["context"]

    prompt = f"""你是一个专业的企业知识库助手。请严格根据以下资料回答问题。
如果资料中没有答案，请直接回答"资料中未提及"，不要编造。

资料：
{context}

问题：{query}

回答："""

    answer = llm.generate(prompt, temperature=0.0)

    return {"messages": [AIMessage(content=answer)]}