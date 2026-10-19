from typing import TypedDict, Annotated
import operator
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage

from src.retrieval.vector_store import HybridRetriever
from src.generation.llm import get_llm
from src.agent.tools import query_policy_detail


# ========== 1. 更新状态定义 ==========
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    query: str
    context: str
    is_relevant: bool  # 评估结果
    retry_count: int  # 重试次数


# ========== 2. 初始化组件 ==========
retriever = HybridRetriever()


# ========== 3. 原有节点 ==========
def retrieve_node(state: AgentState) -> dict:
    query = state["query"]
    results = retriever.search(query, k=10, use_rerank=True)
    context = "\n\n".join([r["text"] for r in results])
    return {"context": context}


def generate_node(state: AgentState) -> dict:
    query = state["query"]
    context = state["context"]

    # 获取模型并绑定工具
    llm = get_llm()
    tools = [query_policy_detail]
    llm_with_tools = llm.bind_tools(tools)

    prompt = f"""你是一个专业的企业知识库助手。请严格根据以下资料回答问题。
如果资料中没有答案，你可以调用工具查询。如果连工具都查不到，请直接回答"资料中未提及"，不要编造。

资料：
{context}

问题：{query}"""

    # 模型思考：直接回答 or 调用工具
    response = llm_with_tools.invoke(prompt)

    # 如果模型决定调用工具
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        print(f"\n[Agent 调用工具] {tool_name}，参数：{tool_args}")

        if tool_name == "query_policy_detail":
            tool_result = query_policy_detail.invoke(tool_args)

            # 把工具查询的结果喂回给大模型，生成最终回答
            final_prompt = f"根据工具查询的结果，回答用户问题。\n工具结果：{tool_result}\n问题：{query}"
            final_response = llm.invoke(final_prompt)
            return {"messages": [AIMessage(content=final_response.content)]}

    # 如果不需要调用工具，直接返回答案
    return {"messages": [AIMessage(content=response.content)]}


# ========== 4. 评估节点 ==========
def grade_node(state: AgentState) -> dict:
    query = state["query"]
    context = state["context"][:1000]
    llm = get_llm()

    prompt = f"""你是一个专业的检索质量评估员。请判断以下检索到的资料，是否与用户的问题相关。
如果资料里包含回答该问题的线索，输出 'yes'；如果毫无关联，输出 'no'。
只输出小写的 yes 或 no，不要输出其他任何文字。

资料：
{context}

问题：
{query}
"""
    ans = llm.invoke(prompt).content.strip().lower()
    is_relevant = "yes" in ans
    return {"is_relevant": is_relevant}


# ========== 5. 改写节点 ==========
def rewrite_node(state: AgentState) -> dict:
    query = state["query"]
    retry_count = state.get("retry_count", 0)
    llm = get_llm()

    prompt = f"""你是一个搜索专家。根据用户的问题，生成一个更适合在企业知识库中检索的精准查询词。
不要包含多余的解释，只输出改写后的查询语句。

原问题：
{query}
"""
    new_query = llm.invoke(prompt).content.strip()
    print(f"\n[Agent 触发重试] 原查询：'{query}' -> 改写为：'{new_query}'")

    return {
        "query": new_query,
        "retry_count": retry_count + 1
    }