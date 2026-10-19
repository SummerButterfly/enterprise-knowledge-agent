from src.agent.graph import agent


def main():
    config = {"configurable": {"thread_id": "tool_test_001"}}

    # 问一个知识库里没有，但工具能查到的问题
    q = "帮我查一下保单 P123 的详情"

    print(f"用户：{q}")
    result = agent.invoke(
        {
            "query": q,
            "messages": [],
            "context": "",
            "is_relevant": False,
            "retry_count": 0
        },
        config=config
    )
    print(f"Agent：{result['messages'][-1].content}")


if __name__ == "__main__":
    main()