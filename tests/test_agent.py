from src.agent.graph import agent


def main():
    queries = [
        "年假怎么算？",
        "报销需要什么材料？",
        "公司股票代码是多少？",   # 应该拒答
    ]

    for q in queries:
        print(f"\n{'='*70}")
        print(f"问题：{q}")
        print('='*70)

        result = agent.invoke({
            "query": q,
            "messages": [],
            "context": "",
        })

        # 最后一条消息就是答案
        answer = result["messages"][-1].content
        print(f"回答：{answer}")


if __name__ == "__main__":
    main()