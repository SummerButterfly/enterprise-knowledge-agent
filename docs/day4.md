# Day 4：LangGraph 基础（完整版）

---

## 一、Day 4 目标

**把"检索 → 生成"串成一个状态机（StateGraph），跑通 RAG 最小闭环。**

一句话：用户提问 → 检索资料 → 拼接上下文 → 大模型生成回答。

---

## 二、整体流程图

```
用户提问："年假怎么算？"
   ↓ ① 进入状态机
START 节点
   ↓ ② 流转到检索节点
retrieve_node（Milvus 粗排 + Rerank 精排）
   ↓ ③ 流转到生成节点
generate_node（拼接 Prompt → 调用 DeepSeek）
   ↓ ④ 输出结果
END 节点（返回最终答案）
```

---

## 三、环境确认

```powershell
# 1. 确认在项目目录
cd D:\projects2026_10_15\enterprise-knowledge-agent

# 2. 确认 Milvus 在跑
docker ps

# 3. 确认 LangGraph 已安装
python -c "import langgraph; print('OK')"

# 4. 确认 .env 里填了真实的 DeepSeek Key
# （不能用 sk-placeholder）
```

**全部通过才能继续。**

---

## 四、Day 4 涉及的 4 个文件

### 1. `src/generation/llm.py`（新增）

```python
from openai import OpenAI
from config.settings import settings


class LLMClient:
    """封装 DeepSeek 大模型调用"""

    def __init__(self):
        self.client = OpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL,
        )
        self.model = settings.LLM_MODEL

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        """生成答案"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content
```

**关键点**：
- 使用 OpenAI 兼容接口调用 DeepSeek。
- `temperature=0.0` 确保输出稳定，不随机发挥。

---

### 2. `src/agent/nodes.py`（新增）

```python
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import BaseMessage, AIMessage

from src.retrieval.vector_store import HybridRetriever
from src.generation.llm import LLMClient


# ========== 1. 定义状态 ==========
class AgentState(TypedDict):
    """状态机中传递的数据"""
    messages: Annotated[list[BaseMessage], operator.add]  # 对话历史
    query: str                                            # 当前查询
    context: str                                          # 检索到的上下文


# ========== 2. 初始化组件 ==========
retriever = HybridRetriever()
llm = LLMClient()


# ========== 3. 定义节点 ==========

def retrieve_node(state: AgentState) -> dict:
    """检索节点：调用 Rerank 检索"""
    query = state["query"]
    results = retriever.search(query, k=10, use_rerank=True)

    # 拼接 Top3 上下文
    context = "\n\n".join([r["text"] for r in results])

    return {"context": context}


def generate_node(state: AgentState) -> dict:
    """生成节点：调用大模型生成答案"""
    query = state["query"]
    context = state["context"]

    prompt = f"""你是一个专业的企业知识库助手。请严格根据以下资料回答问题。
如果资料中没有答案，请直接回答"资料中未提及"，不要编造。

资料：
{context}

问题：{query}

回答："""

    answer = llm.generate(prompt, temperature=0.0)

    return {"messages": [AIMessage(content=answer)]}
```

**关键点**：
- `AgentState` 用 `TypedDict` 定义，`Annotated` 和 `operator.add` 让 `messages` 可以自动累加。
- `retrieve_node` 调用 Day 3 写的 `HybridRetriever`。
- `generate_node` 用 Prompt 严加约束，防止幻觉。

---

### 3. `src/agent/graph.py`（新增）

```python
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
```

**关键点**：
- 用 `StateGraph(AgentState)` 创建状态机。
- `add_node` 添加节点，`add_edge` 串联节点。
- `compile()` 编译为可执行实例。

---

### 4. `tests/test_agent.py`（新增）

```python
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
```

**关键点**：
- `agent.invoke` 接收初始状态。
- 返回的 `result["messages"][-1].content` 是最终答案。

---

## 五、运行命令

```powershell
python -m tests.test_agent
```

**为什么加 `-m`**：文件里有 `from src.agent.graph...`，必须从项目根目录找包。

**实际执行顺序**：
```
test_agent.py 被运行
   ↓
初始化 agent（调用 build_agent）
   ↓
遍历 3 个问题
   ↓
每个问题：invoke 状态机 → retrieve → generate → 打印回答
```

---

## 六、验证结果

**预期输出**：
```
======================================================================
问题：年假怎么算？
======================================================================
回答：根据资料第五条，年假按工龄计算：满 1 年 5 天，满 3 年 10 天，满 5 年 15 天。

======================================================================
问题：报销需要什么材料？
======================================================================
回答：报销需要提供：发票、行程单、审批单，三者缺一不可。

======================================================================
问题：公司股票代码是多少？
======================================================================
回答：资料中未提及。
```

**看到"资料中未提及"，说明防幻觉生效。**

---

## 七、Day 4 踩过的坑

| 坑 | 原因 | 解决 |
|----|------|------|
| **API 认证失败 (401)** | `.env` 里的 Key 还是 `sk-placeholder`，或改了 Key 没重启终端 | 填真实 Key，**重启 PyCharm** 让环境变量生效 |
| **模型瞎编** | Prompt 没约束 | 在 Prompt 里加"资料中没有请回答未提及"，并把 `temperature` 设为 0 |
| **`context` 为空** | Milvus 没启动或数据没了 | `docker ps` 确认容器在跑，`list_collections` 确认数据在 |
| **`KeyError: 'messages'`** | 状态定义错，或没传 `messages` | 确保 `AgentState` 里有 `messages`，`invoke` 时传入 `"messages": []` |
| **找不到模块 `langgraph`** | 没装 | `pip install langgraph==0.2.45` |

---

## 八、核心概念速记

| 概念 | 一句话 |
|------|--------|
| **StateGraph** | LangGraph 的状态机，管理流程 |
| **State** | 节点间传递的数据（`AgentState`） |
| **Node** | 每一步操作（`retrieve_node`、`generate_node`） |
| **Edge** | 节点之间的连接（`add_edge`） |
| **START / END** | 状态机的起点和终点 |
| **compile()** | 把状态机编译成可执行实例 |

---

## 九、Day 4 完成后的项目状态

```
enterprise-knowledge-agent/
├── src/
│   ├── ingestion/
│   │   └── indexer.py      ✅
│   ├── retrieval/
│   │   ├── vector_store.py ✅
│   │   └── reranker.py     ✅
│   ├── generation/
│   │   └── llm.py          ✅ Day 4 新增
│   └── agent/
│       ├── nodes.py        ✅ Day 4 新增
│       └── graph.py        ✅ Day 4 新增
├── tests/
│   ├── test_retrieval.py   ✅
│   └── test_agent.py       ✅ Day 4 新增
└── docs/
    ├── day1.md
    ├── day2.md
    ├── day3.md
    └── day4.md             ✅ 本文档
```

---

## 十、Day 5 预告

**主题：Agentic RAG 核心（含金量最高）**

1. 给状态机加"评估节点"
2. 检索不相关时，自动"改写查询"
3. 重试检索，直到相关
4. 防止死循环（加 `retry_count` 上限）

**这是普通 RAG 和"企业级 Agentic RAG"的分水岭。**

---

## 十一、面试可讲点

| 面试问题 | 你怎么答 |
|---------|---------|
| 为什么要用 LangGraph？ | 普通 if-else 写循环会变面条，LangGraph 状态机清晰可控 |
| 状态机怎么流转的？ | START → retrieve → generate → END |
| State 里存什么？ | query、context、messages |
| 怎么防幻觉？ | Prompt 约束 + temperature=0 |
| 踩过什么坑？ | API Key 没生效（没重启终端）、模型瞎编（Prompt 没约束） |