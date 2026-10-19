from src.agent.graph import agent

def main():
    # thread_id 相当于用户的 Session ID，用于区分不同用户的对话
    config = {"configurable": {"thread_id": "user_session_001"}}

    queries = [
        "年假怎么算？",           # 第一轮
        "那满3年有多少天？",      # 第二轮（依赖上下文）
    ]

    for q in queries:
        print(f"\n用户：{q}")
        result = agent.invoke(
            {
                "query": q,
                "messages": [],
                "context": "",
                "is_relevant": False,
                "retry_count": 0
            },
            config=config  # 传入记忆配置
        )
        print(f"Agent：{result['messages'][-1].content}")


if __name__ == "__main__":
    main()