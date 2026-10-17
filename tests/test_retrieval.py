from src.retrieval.vector_store import HybridRetriever


def main():
    retriever = HybridRetriever()

    queries = [
        "年假怎么算？",
        "报销需要什么材料？",
        "上班时间是什么时候？",
        "满3年工龄有多少年假？",
    ]

    for q in queries:
        print(f"\n{'='*70}")
        print(f"问题：{q}")
        print('='*70)

        # 1. 不 Rerank（粗排）
        print("\n【粗排（Milvus Top3）】")
        no_rerank = retriever.search(q, k=10, use_rerank=False)
        for i, r in enumerate(no_rerank[:3], 1):
            text = r["text"].replace("\n", " ")[:60]
            print(f"  {i}. [score={r['score']:.4f}] {text}...")

        # 2. 加 Rerank（精排）
        print("\n【精排（Rerank Top3）】")
        reranked = retriever.search(q, k=10, use_rerank=True)
        for i, r in enumerate(reranked, 1):
            text = r["text"].replace("\n", " ")[:60]
            print(f"  {i}. [rerank={r['rerank_score']:.4f}] {text}...")


if __name__ == "__main__":
    main()