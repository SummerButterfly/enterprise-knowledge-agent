from langgraph.graph import StateGraph, START, END

from src.agent.nodes import AgentState, retrieve_node, generate_node


def build_agent():
    """构建 RAG 状态机"""
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    # 添加边（连线）
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


# 全局 agent 实例
agent = build_agent()