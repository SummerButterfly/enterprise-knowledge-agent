# Day 1：数据注入（Milvus 版）

---

## 一、Day 1 目标

**把一份 PDF 拆成小块，转成向量，存进 Milvus 数据库。**

一句话：PDF → 切块 → 向量化 → 存进 Milvus。

---

## 二、整体流程图

```
PDF 文件
   ↓ ① 读取
纯文本
   ↓ ② 切块
文本块列表（14 个）
   ↓ ③ 向量化（BGE）
向量（768 维）
   ↓ ④ 入库
Milvus 数据库（enterprise_kb 集合）
```

---

## 三、环境前置

| 项目 | 状态 |
|------|------|
| Python 虚拟环境 `.venv` | ✅ |
| 依赖包 | ✅ |
| WSL 2 + Ubuntu | ✅ |
| Docker Desktop | ✅ |
| 镜像加速 | ✅ |
| Milvus 三容器 | ✅ |

**Milvus 容器**：

```powershell
docker ps
```

**预期看到 3 个**：
- `milvus-standalone`（Milvus 本体）
- `milvus-minio`（存数据）
- `milvus-etcd`（存元数据）

---

## 四、Day 1 涉及的 6 个文件

### 1. `.env`（环境变量）

```bash
DEEPSEEK_API_KEY=sk-placeholder
PDF_PATH=data/raw/test.pdf
HF_ENDPOINT=https://hf-mirror.com
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

**作用**：
- 存密钥（不写死在代码里）
- 存路径和地址
- `HF_ENDPOINT` 是 HuggingFace 国内镜像，加速模型下载

**注意**：
- 等号两边不加空格
- 值不加引号
- 不写 `export`

---

### 2. `config/settings.py`（全局配置）

```python
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PDF_PATH = os.getenv("PDF_PATH", "data/raw/test.pdf")

    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
    COLLECTION_NAME = "enterprise_kb"

    EMBED_MODEL = "BAAI/bge-base-zh-v1.5"
    LLM_MODEL = "deepseek-chat"

    CHUNK_SIZE = 300
    CHUNK_OVERLAP = 50

    TOP_K = 10
    RERANK_TOP_K = 3

    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_BASE_URL = "https://api.deepseek.com"

settings = Settings()
```

**作用**：把"配置"和"代码"分开。改地址、改模型名，只改这一处。

---

### 3. `docker-compose.yml`（启动 Milvus）

```yaml
services:
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - ./milvus_data/etcd:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    healthcheck:
      test: ["CMD", "etcdctl", "endpoint", "health"]
      interval: 30s
      timeout: 20s
      retries: 3

  minio:
    container_name: milvus-minio
    image: cgr.dev/chainguard/minio:latest
    user: "0:0"
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - ./milvus_data/minio:/minio_data
    command: server /minio_data --console-address ":9001"

  milvus:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.3.3
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - ./milvus_data/milvus:/var/lib/milvus
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - etcd
      - minio
```

**作用**：告诉 Docker 启动 Milvus 三个容器。

**启动命令**：

```powershell
docker compose up -d
```

---

### 4. `src/ingestion/loader.py`（读 PDF）

```python
from pdfminer.high_level import extract_text as pdf_extract
from pathlib import Path
from docx import Document
def load_document(path: str) -> str:
    """支持 PDF / Word / TXT"""
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".pdf":
        text = pdf_extract(str(p))
    elif suffix == ".docx":
        doc = Document(str(p))
        text = "\n".join(para.text for para in doc.paragraphs)
    elif suffix == ".txt":
        with open(str(p), "r", encoding="utf-8") as f:
            text = f.read()
    else:
        raise ValueError(f"不支持的文件格式：'{suffix}'（支持 .pdf / .docx / .txt）")

    if text.strip() == "":
        raise ValueError(f"文件无文本内容（可能是扫描版）：{path}")
    return text
```

**作用**：把 PDF 文件变成一段**纯文本字符串**。

---

### 5. `src/ingestion/splitter.py`（切块）

```python
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_document(text: str, doc_type: str = "enterprise"):
    patterns = [
        r"(?=第[一二三四五六七八九十百零\d]+[章节条])",
        r"(?=[一二三四五六七八九十]+、)",
        r"(?=\d+\.\d+\s)",
        r"(?=\d+、)",
    ]
    combined = "|".join(patterns)
    sections = [s.strip() for s in re.split(combined, text) if s.strip()]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "；", ""]
    )

    chunks = []
    for sec in sections:
        if len(sec) <= 300:
            chunks.append(sec)
        else:
            chunks.extend(splitter.split_text(sec))

    return chunks
```

**作用**：把长文本切成**语义完整的小块**。

**切块策略**：
- 优先按"第X章/条"、"一、"、"1.1"等**标题**切
- 每块不超过 300 字
- 超长的再按句子切

---

### 6. `src/ingestion/indexer.py`（Milvus 版）

```python
from pymilvus import (
    connections, utility,
    Collection, CollectionSchema, FieldSchema, DataType
)
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from src.ingestion.loader import load_document
from src.ingestion.splitter import split_document
from config.settings import settings

COLLECTION_NAME = settings.COLLECTION_NAME
DIM = 768


def build_index(file_path: str, doc_type: str = "enterprise"):
    # 1. 连接 Milvus
    connections.connect(host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)

    # 2. 如果集合已存在，删掉重建
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
        print(f"已删除旧集合：{COLLECTION_NAME}")

    # 3. 定义 schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=DIM),
    ]
    schema = CollectionSchema(fields, description="企业知识库")
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    print(f"已创建集合：{COLLECTION_NAME}")

    # 4. 建向量索引
    collection.create_index(
        field_name="vector",
        index_params={
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 128},
        },
    )

    # 5. 加载 BGE 模型
    embeddings = HuggingFaceBgeEmbeddings(model_name=settings.EMBED_MODEL)

    # 6. 读文档 + 切块
    text = load_document(file_path)
    chunks = split_document(text, doc_type)
    print(f"切出 {len(chunks)} 个块")

    # 7. 向量化
    vectors = embeddings.embed_documents(chunks)

    # 8. 插入
    collection.insert([
        chunks,
        [file_path] * len(chunks),
        vectors,
    ])
    collection.flush()
    collection.load()
    print(f"已写入 Milvus：{COLLECTION_NAME}，共 {collection.num_entities} 条")

    return chunks


if __name__ == "__main__":
    build_index(settings.PDF_PATH)
```

---

## 五、运行命令

```powershell
python -m src.ingestion.indexer
```

**为什么加 `-m`**：让 Python 从项目根目录找包，这样 `from src.xxx` 和 `from config.xxx` 才能导入成功。

**实际执行顺序**：

```
indexer.py 被运行
   ↓
连接 Milvus
   ↓
删除旧集合（如存在）
   ↓
创建新集合 + schema + 索引
   ↓
加载 BGE 模型（首次下载约 400MB）
   ↓
调用 load_document() → 读 PDF
   ↓
调用 split_document() → 切出 14 块
   ↓
向量化 14 个块
   ↓
插入 Milvus
   ↓
flush + load
```

---

## 六、验证结果

### 方式 1：看命令行输出

```
已创建集合：enterprise_kb
切出 14 个块
已写入 Milvus：enterprise_kb，共 14 条
```

### 方式 2：命令行查询

```powershell
python -c "from pymilvus import connections, utility; connections.connect(host='localhost', port='19530'); print(utility.list_collections())"
```

**预期输出**：

```
['enterprise_kb']
```

### 方式 3：查数据条数

```powershell
python -c "from pymilvus import connections, Collection; connections.connect(host='localhost', port='19530'); c = Collection('enterprise_kb'); c.load(); print('条数:', c.num_entities)"
```

**预期输出**：

```
条数: 14
```

---

## 七、Day 1 踩过的坑

| 坑 | 原因 | 解决 |
|----|------|------|
| requirements.txt 报编码错 | 记事本存成 GBK | 用 VS Code 建，存 UTF-8 |
| docker 命令找不到 | Docker 没装 | 装 Docker Desktop |
| 拉 Milvus 镜像超时 | 没配镜像加速 | Docker 加 registry-mirrors |
| MinIO 镜像被拒 | Docker Hub 限制 | 换 Chainguard 镜像 |
| Chainguard MinIO 启动失败 | command 格式不同 | 去掉开头的 minio |
| MinIO 显示 unhealthy | 镜像无 curl | 忽略，不影响使用 |
| sentence_transformers 缺失 | requirements 故意没放 | 单独 pip install |
| HuggingFace 下载超时 | 没设国内镜像 | .env 加 HF_ENDPOINT |
| langchain-milvus 依赖冲突 | 需要 langchain-core 1.x | 改用 pymilvus SDK |
| pymilvus 路径错误 | .venv 记录旧路径 | 重建虚拟环境 |

**这些坑在面试里都是加分项**——说明你真的动手跑过。

---

## 八、核心概念速记

| 概念 | 一句话 |
|------|--------|
| **RAG** | 检索 + 生成，先找资料再让大模型回答 |
| **向量** | 把文字变成一串数字，方便机器比较相似度 |
| **BGE** | 中文向量化模型，把文字变向量 |
| **切块** | 把长文切成语义完整的小块 |
| **Collection** | Milvus 里的"集合"，相当于数据库的表 |
| **Schema** | 集合的字段定义 |
| **IVF_FLAT** | 向量索引类型，聚类 + 倒排 |
| **COSINE** | 余弦相似度，度量向量"方向" |

---

## 九、Day 1 完成后的项目状态

```
enterprise-knowledge-agent/
├── .env                    ✅ 含 Milvus 配置
├── .gitignore              ✅
├── requirements.txt        ✅
├── docker-compose.yml      ✅ Milvus 三容器
├── config/
│   ├── __init__.py         ✅
│   └── settings.py         ✅ Milvus 版
├── src/
│   ├── __init__.py         ✅
│   ├── ingestion/
│   │   ├── __init__.py     ✅
│   │   ├── loader.py       ✅
│   │   ├── splitter.py     ✅
│   │   └── indexer.py      ✅ Milvus 版
│   └── retrieval/
│       └── __init__.py     ⏸ Day 2
├── data/raw/test.pdf       ✅
└── milvus_data/            ✅ 持久化数据
```

---

## 十、Day 2 预告

**主题：Milvus 版检索**

1. 封装 `HybridRetriever` 类
2. 用 `collection.search()` 检索
3. 测试几个问题，看 Top3 结果

**目标**：问"年假怎么算？"，返回年假相关条款。

---

## 十一、面试可讲点

| 面试问题 | 你怎么答 |
|---------|---------|
| 数据注入怎么做的？ | PDF 加载 → 按标题切块 → BGE 向量化 → 存 Milvus |
| 为什么按标题切？ | 条款是语义完整单元，混切会割裂定义 |
| 为什么用 Milvus？ | 模拟企业生产场景，支持大规模 |
| 为什么用 pymilvus 不用 LangChain？ | LangChain 版本冲突，SDK 更可控 |
| Milvus schema 怎么设计的？ | id + text + source + vector，4 个字段 |
| 索引怎么配的？ | IVF_FLAT，COSINE，nlist=128 |
| 踩过什么坑？ | MinIO 镜像、HuggingFace 下载、LangChain 冲突 |