# Day 1：数据注入

## 完成内容
- PDF 加载（loader.py）
- 按章节/条款切块（splitter.py）
- 向量化 + 写入 Qdrant（indexer.py）

## 结果
- 切出 14 个块
- 存入 Qdrant collection: enterprise_kb
- Dashboard: http://localhost:6333/dashboard

## 踩坑记录
- sentence-transformers 需要单独装
- HF_ENDPOINT 要设镜像，否则 HuggingFace 下载超时
- from_documents 需要 Document 对象，改用了 from_texts
- QdrantClient 不能被 pickle，改用 url= 连接

## 关键命令
- 启动 Qdrant：docker compose up -d qdrant
- 建索引：python -m src.ingestion.indexer