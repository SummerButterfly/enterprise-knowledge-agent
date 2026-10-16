from src.retrieval.vector_store import HybridRetriever


def main():
    retriever = HybridRetriever()

    queries = [
        "年假怎么算？",
        "报销需要什么材料？",
        "上班时间是什么时候？",
    ]

    for q in queries:
        print(f"\n{'='*60}")
        print(f"问题：{q}")
        print('='*60)

        results = retriever.search(q, k=3)
        for i, r in enumerate(results, 1):
            text = r["text"].replace("\n", " ")[:80]
            print(f"  {i}. [score={r['score']:.4f}] {text}...")


if __name__ == "__main__":
    main()