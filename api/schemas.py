from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    question: str = Field(..., description="用户的问题")
    thread_id: str = Field(default="default_user", description="会话ID，用于区分多轮记忆")

class ChatResponse(BaseModel):
    answer: str = Field(..., description="Agent 的回答")