# Day 9：Docker 全栈容器化部署（完整版）

---

## 一、Day 9 目标

**把 FastAPI 业务代码和 Milvus 向量库打包在一起，实现“一条命令启动整个系统”。**

在前面的开发中，我们只把 Milvus 放进了 Docker，业务代码是在本地 PyCharm 里跑的。为了达到企业级交付标准，今天要把所有服务容器化，并解决在 Windows 环境下打包 AI 项目时遇到的各种坑（依赖过大、网络超时、跨系统不兼容等）。

---

## 二、核心文件编写

### 1. 纯净版 `requirements.txt`

> **避坑点**：绝对不能把本地虚拟环境 `.venv` 直接复制进容器。Windows 下的 `.pyd` 或 `pywin32` 文件在 Linux 容器里会报错崩溃。必须让容器根据纯净的依赖清单，在 Linux 环境重新下载。

```text
langchain==0.3.7
langchain-core==0.3.15
langchain-community==0.3.5
langgraph==0.2.45
langchain-openai==0.2.5
langchain-text-splitters==0.3.2
pymilvus==2.4.4
fastapi==0.115.4
uvicorn[standard]==0.32.0
python-multipart==0.0.12
python-dotenv==1.0.1
pdfminer.six==20231228
sentence-transformers==3.2.1
```

### 2. 标准 `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# 使用清华源加速下载依赖包
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 3. 全栈编排 `docker-compose.yml`

在原有 Milvus 三个服务的基础上，追加业务服务 `rag-api`：

```yaml
services:
  # ========== Milvus 依赖组件 ==========
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

  # ========== 业务服务 ==========
  rag-api:
    container_name: rag-api-service
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    environment:
      # 注意：容器内互连不能写 localhost，必须写 Milvus 的服务名
      - MILVUS_HOST=milvus
      - MILVUS_PORT=19530
    depends_on:
      - milvus
```

---

## 三、启动与验证

执行以下命令进行构建和启动：

```powershell
docker compose up -d --build
```

**验证容器状态**：
```powershell
docker ps
```
预期看到 4 个容器（etcd、minio、milvus、rag-api-service）全部为 `Up` 状态。

**验证接口**：
浏览器打开 `http://localhost:8000/docs`，发送 POST 请求测试，若返回正确 JSON 答案，则代表全栈部署成功。

---

## 四、Day 9 踩过的史诗级坑（面试重点）

这一天是环境报错的重灾区，正是这些坑让你理解了 Docker 底层原理：

1. **WSL 2 磁盘爆满引发 Docker 崩溃（`EOF` 报错）**
   *   **现象**：构建镜像时报 `failed to receive status: rpc error: code = Unavailable desc = error reading from server: EOF`。
   *   **原因**：依赖包（PyTorch 等）体积过大，撑爆了 Windows 下 WSL 2 虚拟磁盘，导致 Docker 守护进程 I/O 断开。
   *   **解决**：清理无用镜像缓存（`docker system prune`），在 Docker Desktop 设置中把虚拟磁盘路径迁移到空间充足的 D 盘。
2. **Windows 与 Linux 的跨系统不兼容（`pywin32` 崩溃）**
   *   **现象**：尝试直接把本地 `.venv` 复制进容器以逃避下载，结果容器启动闪退，日志报 `ModuleNotFoundError: No module named 'pywin32_bootstrap'`。
   *   **原因**：`pywin32` 是 Windows 专属底层库，Linux 容器无法识别该模块的 `.pth` 文件。
   *   **解决**：放弃本地包复制大法，回归 `Dockerfile` 标准构建，让 Linux 容器根据 `requirements.txt` 重新拉取正确的 Linux 二进制包。
3. **网络超时导致 `pip install` 假死**
   *   **现象**：终端刷屏 `[+] Building 4500s...`，依赖包迟迟下不下来。
   *   **解决**：将 `pip` 的镜像源从默认的官方源或清华源切换为速度更快的阿里云源，或者开启 `--no-cache` 强制重新走网络。

---

## 五、面试提炼（降维打击话术）

**面试官**：“你在 Docker 部署时遇到过什么棘手的问题？”

**你的回答**：
> “在打包包含大模型的 Python 服务时，遇到了经典的**工程环境地狱**。
> 
> 由于 PyTorch 等依赖体积极大，一开始我在构建时，直接撑爆了 WSL 2 的虚拟磁盘，导致 Docker 守护进程崩溃，报 `EOF` 错误。后来我又尝试直接把本地的虚拟环境挂载/复制进容器，结果因为 Windows 和 Linux 的底层差异，触发了 `pywin32` 崩溃。
> 
> 经过排查，我放弃了这些取巧的路径。我的解决方案是：清理所有的损坏缓存，将 WSL 数据盘迁移到 D 盘，然后用纯净的 `python:3.11-slim` 镜像配合国内加速源重新构建。现在，任何人拉取我的代码，只需要一条 `docker compose up -d --build` 命令，就能一键拉起 Milvus 和业务代码。这让我认识到，**容器化必须严格遵循系统的隔离规范，对资源限制和环境差异保持敬畏。**”