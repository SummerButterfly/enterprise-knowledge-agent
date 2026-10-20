from fastapi import APIRouter, HTTPException
from api.schemas import ChatRequest, ChatResponse
from src.agent.graph import agent

router = APIRouter()


@router.post("/chat", response_model=ChatResponse, summary="与 RAG Agent 对话")
async def chat(request: ChatRequest):
    try:
        # 1. 构造 LangGraph 所需的配置
        config = {"configurable": {"thread_id": request.thread_id}}

        # 2. 构造初始状态
        initial_state = {
            "query": request.question,
            "messages": [],
            "context": "",
            "is_relevant": False,
            "retry_count": 0
        }

        # 3. 调用 Agent
        result = agent.invoke(initial_state, config=config)

        # 4. 提取最后一条消息作为回答
        answer = result["messages"][-1].content

        return ChatResponse(answer=answer)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))