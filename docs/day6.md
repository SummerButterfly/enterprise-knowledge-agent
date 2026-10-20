# Day 6：Memory（多轮记忆）与 Tools（工具调用）完整版

---

## 一、Day 6 目标

**让 Agent 拥有“记忆”和“手脚”。**

1. **Memory（多轮记忆）**：解决大模型“金鱼记忆”的问题，让它在多轮对话中能理解上下文（例如：能听懂“那满3年呢？”是在问年假）。
2. **Tools（工具调用）**：赋予 Agent 调用外部 API 的能力。当本地知识库查不到时（如查保单），能自主调用外部工具（Function Calling）。

---

## 二、整体流程图

```text
用户输入 → 检索(Milvus) → 评估(不相关则重试/相关则继续)
                                 ↓
                           生成节点(判断是否需要工具)
                                 ├── [需要] → 触发外部工具 → 拿到结果喂回大模型 → 生成回答
                                 └── [不需要] → 直接生成回答
                                 ↓
                          通过 MemorySaver 记住本轮对话，供下一轮使用
```

---

## 三、Day 6 涉及的 4 个核心文件修改

### 1. 替换 `src/generation/llm.py`
> **目的**：原生 SDK 不支持 `bind_tools`，换成 LangChain 标准的 `ChatOpenAI`，以支持工具绑定。

```python
from langchain_openai import ChatOpenAI
from config.settings import settings

def get_llm(temperature: float = 0.0):
    """获取支持工具调用的 LangChain LLM 客户端"""
    return ChatOpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.DEEPSEEK_API_KEY,
        base_url=settings.DEEPSEEK_BASE_URL,
        temperature=temperature,
    )
```

---

### 2. 新建 `src/agent/tools.py`
> **目的**：定义 Agent 可以调用的工具函数，用 `@tool` 装饰器标注。

```python
from langchain_core.tools import tool

@tool
def query_policy_detail(policy_no: str) -> str:
    """根据保单号查询保单详情，当用户问到具体保单信息时使用此工具。"""
    # 模拟去数据库查询
    mock_data = {
        "P123": "保单号 P123，险种：重疾险，保额：50万，生效日期：2023-01-01。",
        "P456": "保单号 P456，险种：医疗险，保额：200万，生效日期：2024-05-20。"
    }
    return mock_data.get(policy_no.upper(), f"抱歉，未找到保单 {policy_no} 的记录。")
```

---

### 3. 替换 `src/agent/nodes.py`
> **目的**：在生成节点中绑定工具，判断是否需要调用工具；并适配新的 ChatOpenAI 客户端。

```python
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import BaseMessage, AIMessage

from src.retrieval.vector_store import HybridRetriever
from src.generation.llm import get_llm
from src.agent.tools import query_policy_detail


# ========== 1. 更新状态定义 ==========
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    query: str
    context: str
    is_relevant: bool   # 评估结果
    retry_count: int    # 重试次数


# ========== 2. 初始化组件 ==========
retriever = HybridRetriever()


# ========== 3. 原有节点 ==========
def retrieve_node(state: AgentState) -> dict:
    query = state["query"]
    results = retriever.search(query, k=10, use_rerank=True)
    context = "\n\n".join([r["text"] for r in results])
    return {"context": context}


def generate_node(state: AgentState) -> dict:
    query = state["query"]
    context = state["context"]
    
    # 获取模型并绑定工具
    llm = get_llm()
    tools = [query_policy_detail]
    llm_with_tools = llm.bind_tools(tools)

    prompt = f"""你是一个专业的企业知识库助手。请严格根据以下资料回答问题。
如果资料中没有答案，你可以调用工具查询。如果连工具都查不到，请直接回答"资料中未提及"，不要编造。

资料：
{context}

问题：{query}"""
    
    # 模型思考：直接回答 or 调用工具
    response = llm_with_tools.invoke(prompt)
    
    # 如果模型决定调用工具
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        print(f"\n[Agent 调用工具] {tool_name}，参数：{tool_args}")
        
        if tool_name == "query_policy_detail":
            tool_result = query_policy_detail.invoke(tool_args)
            
            # 把工具查询的结果喂回给大模型，生成最终回答
            final_prompt = f"根据工具查询的结果，回答用户问题。\n工具结果：{tool_result}\n问题：{query}"
            final_response = llm.invoke(final_prompt)
            return {"messages": [AIMessage(content=final_response.content)]}

    # 如果不需要调用工具，直接返回答案
    return {"messages": [AIMessage(content=response.content)]}


# ========== 4. 评估节点 ==========
def grade_node(state: AgentState) -> dict:
    query = state["query"]
    context = state["context"][:1000]
    llm = get_llm()

    prompt = f"""你是一个专业的检索质量评估员。请判断以下检索到的资料，是否与用户的问题相关。
如果资料里包含回答该问题的线索，输出 'yes'；如果毫无关联，输出 'no'。
只输出小写的 yes 或 no，不要输出其他任何文字。

资料：
{context}

问题：
{query}
"""
    ans = llm.invoke(prompt).content.strip().lower()
    is_relevant = "yes" in ans
    return {"is_relevant": is_relevant}


# ========== 5. 改写节点 ==========
def rewrite_node(state: AgentState) -> dict:
    query = state["query"]
    retry_count = state.get("retry_count", 0)
    llm = get_llm()

    prompt = f"""你是一个搜索专家。根据用户的问题，生成一个更适合在企业知识库中检索的精准查询词。
不要包含多余的解释，只输出改写后的查询语句。

原问题：
{query}
"""
    new_query = llm.invoke(prompt).content.strip()
    print(f"\n[Agent 触发重试] 原查询：'{query}' -> 改写为：'{new_query}'")

    return {
        "query": new_query,
        "retry_count": retry_count + 1
    }
```

---

### 4. 替换 `src/agent/graph.py`
> **目的**：引入 `MemorySaver` 实现记忆持久化。

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.agent.nodes import (
    AgentState, retrieve_node, generate_node, grade_node, rewrite_node
)


def should_retry(state: AgentState) -> str:
    if state.get("is_relevant") or state.get("retry_count", 0) >= 2:
        return "generate"
    return "rewrite"


def build_agent():
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_node)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("generate", generate_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade")
    
    workflow.add_conditional_edges(
        "grade",
        should_retry,
        {"generate": "generate", "rewrite": "rewrite"}
    )
    
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", END)

    # 核心：编译时传入内存检查点，实现多轮对话记忆
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


agent = build_agent()
```

---

## 四、验证与测试

### 1. 测试多轮记忆 `tests/test_memory.py`

```python
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
```

### 2. 测试工具调用 `tests/test_tools.py`

```python
from src.agent.graph import agent

def main():
    config = {"configurable": {"thread_id": "tool_test_001"}}

    # 问一个知识库里没有，但工具能查到的问题
    q = "帮我查一下保单 P123 的详情"
    
    print(f"用户：{q}")
    result = agent.invoke(
        {
            "query": q,
            "messages": [],
            "context": "",
            "is_relevant": False,
            "retry_count": 0
        },
        config=config
    )
    print(f"Agent：{result['messages'][-1].content}")

if __name__ == "__main__":
    main()
```

---

## 五、验证结果

### 1. Memory 测试结果
```text
用户：年假怎么算？
Agent：根据资料第五条，年假按工龄计算：满1年5天，满3年10天，满5年15天。

用户：那满3年有多少天？
Agent：满3年有10天。  <-- 成功结合上一轮语境，记住了“年假”
```

### 2. Tools 测试结果
```text
用户：帮我查一下保单 P123 的详情

[Agent 调用工具] query_policy_detail，参数：{'policy_no': 'P123'}
Agent：保单号 P123 的详情如下：
- **险种**：重疾险
- **保额**：50 万
- **生效日期**：2023-01-01  <-- 成功触发外部函数调用
```

---

## 六、Day 6 踩过的坑

| 坑 | 原因 | 解决 |
|----|------|------|
| 跑测试时没有任何输出直接退出 | 新建测试文件时忘记写 `if __name__ == "__main__": main()` | 补上主函数调用 |
| 原生 SDK 不支持 `bind_tools` | 之前是手写的 OpenAI 类，不是 LangChain 标准对象 | 将 `llm.py` 升级，改用 `ChatOpenAI` |
| 工具调用报错或无法触发 | 绑定的 Prompt 里没有告诉模型“你可以调用工具” | 在 Prompt 增加“如果资料中没有答案，你可以调用工具查询”的引导 |

---

## 七、面试提炼（为什么这个功能含金量高？）

**面试官**：“你项目里的 Agent 具备哪些能力？”

**你的回答**：
> “我基于 **LangGraph** 构建了带有 **Memory** 和 **Tools** 的 Agent。
> 
> 1. **多轮记忆**：通过 `MemorySaver` 和 `thread_id`，实现了对话历史持久化。用户问完‘年假怎么算’，下一句只说‘那满3年呢？’，Agent 也能结合上下文精准回答。
> 2. **工具调用（Function Calling）**：通过 LangChain 的 `bind_tools` 绑定外部函数。当遇到知识库范围外的问题（如查特定保单号），大模型能自主决定调用对应的 Python 函数，拿到结果后再汇总生成自然语言回答。这极大扩展了 RAG 系统的能力边界。”