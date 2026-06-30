# myRAG

基于 Python 的本地 RAG 应用，包含 PDF 解析、分块、向量索引、混合召回、RRF 融合、Cross-Encoder 重排序、LLM 生成回答与 RAGAS 评估。前端使用 Vue，后端使用 FastAPI，默认都运行在 `127.0.0.1`。

## 功能范围

1. 数据加载
   - 支持上传多个 PDF 文件到本地目录
   - 文档解析默认使用本地 `MinerU`
   - 递归字符分块，支持 `chunk_size` 和 `chunk_overlap`
2. 索引构建
   - 使用 `Qwen/Qwen3-Embedding-8B` 进行向量化
   - 向量写入 `ChromaDB`
3. 检索与排序
   - Dense Recall: Chroma 向量召回
   - BM25 Recall: `rank-bm25`
   - Sparse Recall: TF-IDF
   - 融合: RRF
   - 重排序: `cross-encoder/ms-marco-MiniLM-L-6-v2`
4. 生成回答
   - 使用 `ChatOpenAI` 调用 DeepSeek 大模型
5. 评估
   - 使用 `RAGAS`
6. 前后端
   - 后端: FastAPI
   - 前端: Vue 3 + Vite

## 目录结构

```text
backend/
  app/
    api/
    core/
    models/
    services/
frontend/
data/
requirements.txt
environment.yml
README.md
```

## 运行说明

### 1. 创建 Python 环境

在项目根目录执行：

```bash
conda env create -f environment.yml
conda activate rag0628
pip install -r requirements.txt
```

如果 `conda env create` 失败，也可以改用：

```bash
conda create -n rag0628 python=3.11 -y
conda activate rag0628
pip install -r requirements.txt
```

### 2. 安装前端依赖

```bash
cd frontend
npm install
cd ..
```

### 3. 配置 `.env`

先复制模板：

```bash
copy .env.example .env
```

推荐的本地运行配置如下：

```env
HOST=127.0.0.1
PORT=8000
CORS_ORIGINS=["http://127.0.0.1:5173"]

MINERU_BACKEND=mineru
MINERU_MODEL_PATH=./models/OpenDataLab/MinerU2___5-Pro-2605-1___2B
MINERU_DEVICE_MAP=auto
MINERU_IMAGE_ANALYSIS=false
MINERU_PDF_DPI=144

EMBEDDING_MODEL_NAME=./models/Qwen/Qwen3-Embedding-0___6B
EMBEDDING_DEVICE=cpu
EMBEDDING_BATCH_SIZE=8

RERANKER_MODEL_NAME=./models/cross-encoder/ms-marco-MiniLM-L6-v2
RERANKER_DEVICE=cpu

LLM_API_BASE=https://api.deepseek.com
LLM_API_KEY=your_deepseek_api_key
LLM_MODEL_NAME=deepseek-chat
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=4096
```

如果你的 PyTorch 已经能识别 GPU，可以把这些值改成：

```env
MINERU_DEVICE_MAP=cuda:0
EMBEDDING_DEVICE=cuda
RERANKER_DEVICE=cuda
```

### 4. 启动后端

```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

后端启动成功后，健康检查地址：

```text
http://127.0.0.1:8000/api/health
```

### 5. 启动前端

新开一个终端：

```bash
cd frontend
npm run dev
```

前端默认地址：

```text
http://127.0.0.1:5173
```

### 6. 使用流程

1. 打开前端页面
2. 选择一个或多个 PDF
3. 选择分块方式：`递归` 或 `固定大小`
4. 设置 `chunk size` 和 `chunk overlap`
5. 点击“上传并建库”
6. 等待前端显示解析、切分、向量化、入库进度
7. 入库完成后，在右侧输入问题并提问


## 后端实现说明

### 1. 文档解析

- `backend/app/services/parser.py`
- 默认通过本地 `ModelScope + Qwen2VLForConditionalGeneration + MinerUClient` 解析 PDF
- 后端会先把 PDF 按页渲染为图片，再逐页调用 `two_step_extract`
- 需要本地安装 `modelscope`、`PyMuPDF`，并将 `MINERU_MODEL_PATH` 指向已下载的 `MinerU2.5-Pro` 模型目录

### 2. 分块

- `backend/app/services/chunking.py`
- 递归字符分块
- 支持自定义分隔符列表

### 3. 向量化与索引

- `backend/app/services/embeddings.py`
- `backend/app/services/vector_store.py`

### 4. 混合召回

- `backend/app/services/retriever.py`
- Dense + BM25 + Sparse 三路召回
- RRF 融合
- Cross-Encoder 精排

### 5. 回答生成

- `backend/app/services/generator.py`
- 通过 `langchain-openai` 中的 `ChatOpenAI` 调用 DeepSeek

### 6. 评估

- `backend/app/services/evaluator.py`

## API 简例

### 1. 上传并建库

```bash
curl -X POST "http://127.0.0.1:8000/api/ingest" \
  -F "files=@demo.pdf" \
  -F "chunk_size=800" \
  -F "chunk_overlap=100"
```

### 2. 提问

```bash
curl -X POST "http://127.0.0.1:8000/api/ask" \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"这份文档的核心结论是什么？\",\"top_k\":5}"
```

### 3. 评估

```bash
curl -X POST "http://127.0.0.1:8000/api/evaluate" \
  -H "Content-Type: application/json" \
  -d "{\"samples\":[{\"question\":\"Q1\",\"answer\":\"A1\",\"contexts\":[\"C1\"],\"ground_truth\":\"G1\"}]}"
```

## 当前实现边界

1. 本项目已经切到本地 Python 方式调用 MinerU 模型，不再走云端接口，也不依赖 `mineru` CLI。
2. Qwen3-Embedding-8B 和 Cross-Encoder 首次加载模型体积较大，建议使用 GPU 或本地模型服务。
3. DeepSeek API Key 请通过环境变量注入，不要直接写入代码仓库。
4. 当前索引状态保存在进程内内存和 Chroma 中；服务重启后，BM25/TF-IDF 语料需要重新装载。若要长期运行，建议再增加本地 chunk 元数据持久化与启动恢复逻辑。
