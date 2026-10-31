import json
import random

# 模拟你 PDF 里的真实业务数据
topics = [
    {"q": "满3年有多少天年假？", "a": "满3年有10天", "source": "第五条"},
    {"q": "满1年有多少天年假？", "a": "满1年有5天", "source": "第五条"},
    {"q": "满5年有多少天年假？", "a": "满5年有15天", "source": "第五条"},
    {"q": "上班时间是什么时候？", "a": "上午 9:00 至下午 18:00，中午休息 1 小时", "source": "第三条"},
    {"q": "请假需要提前多久申请？", "a": "提前一天在系统提交申请", "source": "第四条"},
    {"q": "报销需要什么材料？", "a": "发票、行程单、审批单，三者缺一不可", "source": "第六条"},
    {"q": "超过5000元的报销找谁审批？", "a": "财务总监", "source": "第七条"},
    {"q": "报销申请要在几天内提交？", "a": "费用发生后 30 天内", "source": "第八条"},
]

# 超纲拒答数据
refuse_questions = [
    "公司股票代码是多少？", "公司有几个员工？", "CEO叫什么名字？",
    "公司总部在哪里？", "辞职流程是什么？", "加班有加班费吗？"
]

dataset = []

# 1. 生成正常回答的样本 (150条)
for _ in range(150):
    item = random.choice(topics)
    output_json = json.dumps({
        "answer": item["a"],
        "source": item["source"],
        "confidence": 0.95
    }, ensure_ascii=False)

    dataset.append({
        "instruction": "你是一个企业知识库助手。请严格根据资料回答问题，并且只输出合法的 JSON 格式，包含 answer、source、confidence 三个字段。",
        "input": f"资料：{item['q']}的相关规定。\n问题：{item['q']}",
        "output": output_json
    })

# 2. 生成拒答的样本 (50条)
for _ in range(50):
    q = random.choice(refuse_questions)
    output_json = json.dumps({
        "answer": "资料中未提及",
        "source": "无",
        "confidence": 1.0
    }, ensure_ascii=False)

    dataset.append({
        "instruction": "你是一个企业知识库助手。请严格根据资料回答问题，并且只输出合法的 JSON 格式，包含 answer、source、confidence 三个字段。",
        "input": f"资料：未提供相关制度。\n问题：{q}",
        "output": output_json
    })

# 打乱顺序并保存
random.shuffle(dataset)
with open("train_data.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"成功生成 {len(dataset)} 条微调数据，已保存为 train_data.json")