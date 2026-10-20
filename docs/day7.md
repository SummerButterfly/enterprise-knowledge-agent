# Day 7：FastAPI 封装（将 Agent 变成 Web 服务）完整版

---

## 一、Day 7 目标

**将写好的 LangGraph Agent 包装成 HTTP RESTful 接口，让前端或第三方系统可以通过发 POST 请求与它对话。**

目前的痛点：之前只能通过在终端跑 `python -m tests.test_xxx` 脚本来测试，这在企业级应用中是不可行的。我们需要一个 Web 服务，实现前后端分离。

---

## 二、技术栈与核心文件

- **Web 框架**：FastAPI（高性能，支持异步，自带 Swagger 文档）
- **数据校验**：Pydantic（强制要求传入/传出的 JSON 格式）
- **ASGI 服务器**：Uvicorn

本次新增了三个文件：
1. `api/schemas.py`：定义请求和响应的数据模型。
2. `api/routes.py`：编写业务逻辑路由。
3. `api/main.py`：FastAPI 应用启动入口。

---

## 三、完整代码实现

### 1. 定义数据模型 `api/schemas.py`

```python
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    question: str = Field(..., description="用户的问题")
    thread_id: str = Field(default="default_user", description="会话ID，用于区分多轮记忆")

class ChatResponse(BaseModel):
    answer: str = Field(..., description="Agent 的回答")
```

### 2. 编写路由逻辑 `api/routes.py`

```python
from fastapi import APIRouter, HTTPException
from api.schemas import ChatRequest, ChatResponse
from src.agent.graph import agent

router = APIRouter()

@router.post("/chat", response_model=ChatResponse, summary="与 RAG Agent 对话")
async def chat(request: ChatRequest):
    try:
        # 1. 构造 LangGraph 所需的配置（传入 thread_id 以激活记忆）
        config = {"configurable": {"thread_id": request.thread_id}}
        
        # 2. 构造初始状态
        initial_state = {
            "query": request.question,
            "messages": [],
            "context": "",
            "is_relevant": False,
            "retry_count": 0
        }

        # 3. 同步调用 Agent（底层包含检索、评估、重试、生成、工具调用等）
        result = agent.invoke(initial_state, config=config)
        
        # 4. 提取最后一条消息作为最终回答
        answer = result["messages"][-1].content
        
        return ChatResponse(answer=answer)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 3. 编写启动入口 `api/main.py`

```python
from fastapi import FastAPI
from api.routes import router

app = FastAPI(
    title="Enterprise Knowledge Agent API",
    description="基于 Milvus + LangGraph 的企业知识库问答接口",
    version="1.0.0"
)

# 注册路由，统一加上 /api/v1 前缀
app.include_router(router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"message": "Welcome to Enterprise Knowledge Agent API"}
```

---

## 四、启动与验证

### 1. 启动 Uvicorn 服务
在项目根目录执行（确保虚拟环境已激活，Milvus 容器在跑）：
```powershell
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```
*看到 `Application startup complete.` 说明启动成功。*

### 2. 使用 Swagger UI 在线测试
1. 打开浏览器访问：`http://localhost:8000/docs`
2. 展开 `POST /api/v1/chat`，点击 **Try it out**。
3. 在 Request body 中输入：
   ```json
   {
     "question": "年假怎么算？",
     "thread_id": "web_user_001"
   }
   ```
4. 点击 **Execute**。
5. **预期响应（HTTP 200）**：
   ```json
   {
     "answer": "根据资料第五条，年假按工龄计算：满1年5天，满3年10天，满5年15天。"
   }
   ```

### 3. 多轮记忆测试
在 Request body 中输入（保持 thread_id 不变）：
   ```json
   {
     "question": "那满3年有多少天？",
     "thread_id": "web_user_001"
   }
   ```
   *预期响应*：Agent 能结合上一轮语境，回答出“满3年有10天”。

---

## 五、Day 7 踩过的坑

| 坑 | 原因 | 解决 |
|----|------|------|
| `ModuleNotFoundError: No module named 'api'` | 没在项目根目录启动 Uvicorn | 在终端执行 `cd` 切换到项目根目录再跑 |
| `ModuleNotFoundError: No module named 'fastapi'` | 环境缺少 Web 框架依赖 | 执行 `pip install fastapi "uvicorn[standard]" pydantic` |
| `500 Internal Server Error` | Agent 内部执行报错（如 Milvus 没启动、API Key 无效） | 查看启动 Uvicorn 的终端窗口，看红字报错信息 |

---

## 六、面试提炼

**面试官**：“你这个项目是只能在终端跑吗？”

**你的回答**：
> “不是的，我已经用 **FastAPI** 完成了服务化封装。
> 
> 我定义了基于 **Pydantic** 的请求和响应数据模型，并编写了 `/api/v1/chat` 的 POST 接口。通过 Uvicorn 部署后，它能自动生成 Swagger UI 在线调试文档。
> 
> 前端只需传入 `question` 和 `thread_id`，后端就能驱动 LangGraph 状态机完成检索、生成与记忆回溯，并返回标准 JSON。这为后续接入 Vue/React 前端或者第三方系统调用，提供了标准化的通信协议。”