from src.retrieval.vector_store import HybridRetriever

def main():
    retriever = HybridRetriever()

    queries = [
        "年假怎么算？",
        "报销需要什么材料？",
        "上班时间是什么时候？",
        "员工什么时候上下班？",  # 语义型：没用"上班时间"字样
        "满3年工龄有多少年假？",  # 关键词型：含"年假""3年"
        "出差怎么报销？",  # 语义型：没提"发票"字样
    ]

    for q in queries:
        print(f"\n{'='*60}")
        print(f"问题：{q}")
        print('='*60)

        print("\n【混合检索 Top3】")
        results = retriever.search(q, k=3)
        for i, (doc, score) in enumerate(results, 1):
            text = doc.page_content.replace("\n", " ")[:80]
            print(f"  {i}. [score={score:.4f}] {text}...")

        print("\n【纯向量 Top3】")
        vec_results = retriever.search_pure_vector(q, k=3)
        for i, (doc, score) in enumerate(vec_results, 1):
            text = doc.page_content.replace("\n", " ")[:80]
            print(f"  {i}. [score={score:.4f}] {text}...")

if __name__ == "__main__":
    main()