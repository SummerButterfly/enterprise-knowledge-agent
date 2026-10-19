from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.agent.nodes import (
    AgentState, retrieve_node, generate_node, grade_node, rewrite_node
)


def should_retry(state: AgentState) -> str:
    if state.get("is_relevant") or state.get("retry_count", 0) >= 2:
        return "generate"
    return "rewrite"


def build_agent():
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_node)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("generate", generate_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade")

    workflow.add_conditional_edges(
        "grade",
        should_retry,
        {"generate": "generate", "rewrite": "rewrite"}
    )

    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", END)

    # 核心：编译时传入内存检查点，实现多轮对话记忆
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


agent = build_agent()