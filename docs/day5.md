# Day 5：Agentic RAG（评估+改写重试）完整版

---

## 一、Day 5 目标

**把 Day 4 的“直线生成”变成“带有自我反思和纠错能力的智能体循环”。**

一句话：检索完资料后，先让大模型评估一下资料相不相关；如果不相关，自动改写查询重新检索，直到相关或达到最大重试次数。

---

## 二、整体流程图

```text
START → retrieve（检索）
          ↓
        grade（评估相关性）
          ├── [相关] ──────────→ generate（生成回答） → END
          └── [不相关] → rewrite（改写查询） → 重新检索（最多 2 次）
```

---

## 三、Day 5 涉及的 2 个核心文件修改

### 1. 修改 `src/agent/nodes.py`

新增了 `grade_node`（评估节点）和 `rewrite_node`（改写节点），并在 `AgentState` 中增加了评估结果和重试次数。

```python
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import BaseMessage, AIMessage

from src.retrieval.vector_store import HybridRetriever
from src.generation.llm import LLMClient


# ========== 1. 更新状态定义 ==========
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    query: str
    context: str
    is_relevant: bool   # 新增：评估结果
    retry_count: int    # 新增：重试次数


# ========== 2. 初始化组件 ==========
retriever = HybridRetriever()
llm = LLMClient()


# ========== 3. 原有节点 ==========
def retrieve_node(state: AgentState) -> dict:
    query = state["query"]
    results = retriever.search(query, k=10, use_rerank=True)
    context = "\n\n".join([r["text"] for r in results])
    return {"context": context}


def generate_node(state: AgentState) -> dict:
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


# ========== 4. 新增节点：评估资料相关性 ==========
def grade_node(state: AgentState) -> dict:
    query = state["query"]
    context = state["context"][:1000]  # 截断防止太长，节约 token

    prompt = f"""你是一个专业的检索质量评估员。请判断以下检索到的资料，是否与用户的问题相关。
如果资料里包含回答该问题的线索，输出 'yes'；如果毫无关联，输出 'no'。
只输出小写的 yes 或 no，不要输出其他任何文字。

资料：
{context}

问题：
{query}
"""
    ans = llm.generate(prompt, temperature=0.0).strip().lower()
    is_relevant = "yes" in ans
    return {"is_relevant": is_relevant}


# ========== 5. 新增节点：改写查询 ==========
def rewrite_node(state: AgentState) -> dict:
    query = state["query"]
    retry_count = state.get("retry_count", 0)

    prompt = f"""你是一个搜索专家。根据用户的问题，生成一个更适合在企业知识库中检索的精准查询词。
不要包含多余的解释，只输出改写后的查询语句。

原问题：
{query}
"""
    new_query = llm.generate(prompt, temperature=0.0).strip()
    print(f"\n[Agent 触发重试] 原查询：'{query}' -> 改写为：'{new_query}'")

    return {
        "query": new_query,
        "retry_count": retry_count + 1
    }
```

---

### 2. 修改 `src/agent/graph.py`

引入 `should_retry` 条件路由，并在状态机中加入条件边。

```python
from langgraph.graph import StateGraph, START, END

from src.agent.nodes import (
    AgentState, retrieve_node, generate_node, grade_node, rewrite_node
)


# ========== 条件路由函数 ==========
def should_retry(state: AgentState) -> str:
    """决定下一步走向"""
    # 如果资料相关，或者已经重试了 2 次，就放弃重试，直接生成
    if state.get("is_relevant") or state.get("retry_count", 0) >= 2:
        return "generate"
    # 否则去改写查询
    return "rewrite"


def build_agent():
    workflow = StateGraph(AgentState)

    # 添加 4 个节点
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_node)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("generate", generate_node)

    # 串联连线
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade")

    # 核心：条件边（根据评估结果分流）
    workflow.add_conditional_edges(
        "grade",
        should_retry,
        {
            "generate": "generate",  # 相关 -> 去生成
            "rewrite": "rewrite"     # 不相关 -> 去改写
        }
    )

    # 改写完之后，回到检索重新查
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", END)

    return workflow.compile()


agent = build_agent()
```

---

## 四、验证结果

运行 `python -m tests.test_agent`，终端输出如下（核心看重试逻辑）：

```text
用户提问：我没发票能不能报销差旅费？
[Agent 触发重试] 原查询：'我没发票能不能报销差旅费？' -> 改写为：'差旅费报销 无发票 规定'
最终回答：资料中未提及“无发票”情形下的具体规定。资料仅明确：差旅费报销需提供发票、行程单、审批单，三者缺一不可。

用户提问：公司股票代码是多少？
[Agent 触发重试] 原查询：'公司股票代码是多少？' -> 改写为：'公司股票代码'
[Agent 触发重试] 原查询：'公司股票代码' -> 改写为：'公司股票代码'
最终回答：资料中未提及
```

**验证点：**
1. 刁钻问题触发了改写，成功找到了相关描述。
2. 超纲问题触发了 2 次改写后，被 `retry_count >= 2` 拦截，优雅拒答，**没有陷入死循环**。

---

## 五、Day 5 踩过的坑

| 坑 | 原因 | 解决 |
|----|------|------|
| 运行测试时没有任何输出，直接返回了终端提示符 | 新建了测试文件，但忘记写 `if __name__ == "__main__": main()` | 补上主函数调用 |
| Agent 陷入无限重试 | 没有写重试上限 | 在 `should_retry` 中加入 `retry_count >= 2` |
| 评估节点很慢或很贵 | 传入的 context 太长 | 在 `grade_node` 里对 `context` 做了 `[:1000]` 的截断 |

---

## 六、面试提炼（为什么这个功能含金量最高？）

**面试官：** “你项目里的 RAG 和普通 RAG 有什么区别？”

**你的回答：**
> “我实现的是 **Agentic RAG**。普通的 RAG 检索完直接丢给大模型，一旦检索到无关信息，大模型就会产生幻觉。
> 
> 我基于 **LangGraph** 引入了状态机循环：加入了**评估节点（Grade）** 和**改写节点（Rewrite）**。
> 如果检索到的资料不相关，Agent 会自动反思，把用户的原始 Query 改写成更适合检索的关键词，重新召回。同时我设置了最大重试阈值（2次），防止死循环。
> 实测发现，对于口语化或刁钻的问题（如‘没发票能不能报销’），系统能自动改写为‘差旅费报销规定’，大幅提升了检索的鲁棒性和最终回答的准确性。”