# Enterprise Knowledge Agent

企业知识库智能问答系统 —— 基于 Milvus + LangGraph 的 Agentic RAG。

## 核心功能
- **数据注入**：支持 PDF 文档解析，基于正则表达式按“章节/条款”结构化切块，使用 BGE 模型进行向量化。
- **混合检索**：基于 Milvus（原生 pymilvus SDK）实现，包含粗排（COSINE 相似度）与 Rerank 精排（Cross-Encoder）。
- **Agentic RAG**：基于 LangGraph 状态机，具备自我反思能力。检索不相关时，自动改写查询重试（防止死循环）。
- **多轮记忆 (Memory)**：基于 LangGraph MemorySaver 和 thread_id 实现多轮上下文对话。
- **工具调用 (Tools)**：集成 Function Calling，遇到超纲问题（如查保单）可自主调用外部 API。
- *(FastAPI 接口正在开发中)*

## 技术栈
- **核心框架**：LangChain, LangGraph
- **向量数据库**：Milvus (Docker 部署)
- **向量模型**：BAAI/bge-base-zh-v1.5
- **精排模型**：BAAI/bge-reranker-base
- **大语言模型**：DeepSeek API

## 快速开始

1. **安装依赖**：
   ```bash
   pip install -r requirements.txt
2. **配置环境变量**：
   复制项目根目录的 `.env.example` 文件，重命名为 `.env`，并填入你的真实 API Key：
   ```bash
   DEEPSEEK_API_KEY=sk-你的真实key
   HF_ENDPOINT=https://hf-mirror.com
   MILVUS_HOST=localhost
   MILVUS_PORT=19530
3. **启动 Milvus 向量库**：
   确保电脑已启动 Docker Desktop，然后在项目根目录执行：
   ```bash
   docker compose up -d
## 进阶功能（LoRA 微调）

针对 RAG 生成侧 JSON 格式不稳定的问题，项目引入了 LoRA 微调辅助模块。

- **基座模型**：Qwen2.5-7B-Instruct
- **微调框架**：LLaMA-Factory
- **训练数据**：200 条自建 Alpaca 格式问答对（含 150 条正常回答，50 条拒答样本）
- **微调参数**：LoRA Rank=8, Alpha=16, lr=1e-4, Epoch=3
- **训练结果**：Loss 降至 0.1697，耗时 1 分 14 秒，生成 LoRA 权重约 20MB。
- **实测效果**：JSON 格式合规率从 65% 提升至 98%。

> **⚠️ 依赖隔离说明**：
> 主 RAG 系统仅依赖基础 `requirements.txt`。LoRA 微调模块（`finetuned_llm.py`）需要 GPU 环境，其重型依赖（PyTorch、PEFT、Transformers）已独立抽离至 `src/generation/requirements-gpu.txt`，微调权重放于 `finetune_weights/` 目录中，**避免污染本地 CPU 环境**。