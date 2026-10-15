# Enterprise Knowledge Agent

企业知识库智能问答系统 —— 基于 LangChain + LangGraph 的 Agentic RAG。

## 功能
- 多格式文档索引（PDF/Word/TXT）
- 混合检索（向量 + BM25）+ Rerank
- Agentic RAG：自主判断检索质量，改写查询重试
- Function Calling + 多轮记忆
- FastAPI 接口 + LangSmith 观测

## 快速开始
1. 安装依赖：`pip install -r requirements.txt`
2. 配置密钥：复制 `.env.example` 为 `.env` 填入
3. 启动 Qdrant：`docker compose up -d qdrant`
4. 建索引：`python -m src.ingestion.indexer`

## 目录结构
- `config/` 配置
- `src/ingestion/` 数据注入
- `src/retrieval/` 检索
- `src/agent/` Agent 编排
- `api/` 接口
- `tests/` 测试