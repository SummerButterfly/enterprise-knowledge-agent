from langgraph.graph import StateGraph, START, END

from src.agent.nodes import (
    AgentState, retrieve_node, generate_node, grade_node, rewrite_node
)


# ========== 条件路由函数 ==========
def should_retry(state: AgentState) -> str:
    """决定下一步走向"""
    # 如果资料相关，或者已经重试了 2 次，就放弃重试，直接生成
    if state.get("is_relevant") or state.get("retry_count", 0) >= 2:
        return "generate"
    # 否则去改写查询
    return "rewrite"


def build_agent():
    workflow = StateGraph(AgentState)

    # 添加 4 个节点
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_node)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("generate", generate_node)

    # 串联连线
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade")

    # 核心：条件边（根据评估结果分流）
    workflow.add_conditional_edges(
        "grade",
        should_retry,
        {
            "generate": "generate",  # 相关 -> 去生成
            "rewrite": "rewrite"     # 不相关 -> 去改写
        }
    )

    # 改写完之后，回到检索重新查
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", END)

    return workflow.compile()


agent = build_agent()