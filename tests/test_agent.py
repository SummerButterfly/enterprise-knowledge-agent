from src.agent.graph import agent


def main():
    queries = [
        "年假怎么算？",               # 正常命中
        "我没发票能不能报销差旅费？",   # 刁钻问题，需要改写查询重试
        "公司股票代码是多少？",        # 拒答测试
    ]

    for q in queries:
        print(f"\n{'='*70}")
        print(f"用户提问：{q}")
        print('='*70)

        result = agent.invoke({
            "query": q,
            "messages": [],
            "context": "",
            "is_relevant": False,
            "retry_count": 0
        })

        answer = result["messages"][-1].content
        print(f"最终回答：{answer}")


if __name__ == "__main__":
    main()