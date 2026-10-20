from fastapi import FastAPI
from api.routes import router

app = FastAPI(
    title="Enterprise Knowledge Agent API",
    description="基于 Milvus + LangGraph 的企业知识库问答接口",
    version="1.0.0"
)

# 注册路由，并加上 /api/v1 前缀
app.include_router(router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"message": "Welcome to Enterprise Knowledge Agent API"}