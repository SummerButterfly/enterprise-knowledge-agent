from typing import TypedDict, Annotated
import operator
from langchain_core.messages import BaseMessage, AIMessage

from src.retrieval.vector_store import HybridRetriever
from src.generation.llm import LLMClient


# ========== 1. 更新状态定义 ==========
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    query: str
    context: str
    is_relevant: bool   # 新增：评估结果
    retry_count: int    # 新增：重试次数


# ========== 2. 初始化组件 ==========
retriever = HybridRetriever()
llm = LLMClient()


# ========== 3. 原有节点 ==========
def retrieve_node(state: AgentState) -> dict:
    query = state["query"]
    results = retriever.search(query, k=10, use_rerank=True)
    context = "\n\n".join([r["text"] for r in results])
    return {"context": context}


def generate_node(state: AgentState) -> dict:
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


# ========== 4. 新增节点：评估资料相关性 ==========
def grade_node(state: AgentState) -> dict:
    query = state["query"]
    context = state["context"][:1000]  # 截断防止太长，节约 token

    prompt = f"""你是一个专业的检索质量评估员。请判断以下检索到的资料，是否与用户的问题相关。
如果资料里包含回答该问题的线索，输出 'yes'；如果毫无关联，输出 'no'。
只输出小写的 yes 或 no，不要输出其他任何文字。

资料：
{context}

问题：
{query}
"""
    ans = llm.generate(prompt, temperature=0.0).strip().lower()
    is_relevant = "yes" in ans
    return {"is_relevant": is_relevant}


# ========== 5. 新增节点：改写查询 ==========
def rewrite_node(state: AgentState) -> dict:
    query = state["query"]
    retry_count = state.get("retry_count", 0)

    prompt = f"""你是一个搜索专家。根据用户的问题，生成一个更适合在企业知识库中检索的精准查询词。
不要包含多余的解释，只输出改写后的查询语句。

原问题：
{query}
"""
    new_query = llm.generate(prompt, temperature=0.0).strip()
    print(f"\n[Agent 触发重试] 原查询：'{query}' -> 改写为：'{new_query}'")

    return {
        "query": new_query,
        "retry_count": retry_count + 1
    }