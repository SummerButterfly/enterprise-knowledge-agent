from src.retrieval.vector_store import HybridRetriever
from tests.eval_dataset import EVAL_DATASET
import time


def evaluate():
    retriever = HybridRetriever()

    hit_count = 0
    total_count = 0
    refuse_correct = 0
    refuse_total = 0

    print("开始评估 30 条测试集...")
    start_time = time.time()

    for item in EVAL_DATASET:
        q = item["question"]
        gt = item["ground_truth"]
        total_count += 1

        # 1. 调用检索（这里把 k 设为 5，看前 5 个有没有命中）
        results = retriever.search(q, k=5)

        # 提取检索到的文本，并拼接
        retrieved_texts = [r["text"] for r in results]
        combined_text = "".join(retrieved_texts)

        # 2. 核心优化：文本清洗（去除所有空格、换行、制表符等空白字符）
        clean_text = "".join(combined_text.split())
        clean_gt = "".join(gt.split())

        # 3. 判断逻辑
        if gt == "未提及":
            # 拒答类：看检索出来的内容里有没有明显的超纲词汇
            refuse_total += 1
            if "股票" not in combined_text and "营收" not in combined_text:
                refuse_correct += 1
        else:
            # 正样本类：用清洗后的长句进行精确匹配
            if clean_gt in clean_text:
                hit_count += 1
            else:
                print(f"[Miss] 问题: {q} | 期望: {gt}")

    end_time = time.time()

    # 计算指标
    recall = hit_count / (total_count - refuse_total) * 100
    print("\n" + "=" * 40)
    print("评估结果")
    print("=" * 40)
    print(f"总问题数: {total_count}")
    print(f"正样本数量: {total_count - refuse_total}")
    print(f"正样本命中数: {hit_count}")
    print(f"Top5 召回率: {recall:.2f}%")
    print(f"耗时: {end_time - start_time:.2f} 秒")


if __name__ == "__main__":
    evaluate()