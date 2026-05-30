# Enterprise RAG System

企业级智能知识库问答助手 — 基于 RAG（Retrieval-Augmented Generation）架构。

上传 PDF/TXT 文档 → 自动向量化入库 → 用自然语言提问 → DeepSeek 生成精准回答。

---

## 快速开始

### 1. 环境要求

- Python **3.10+**
- Windows / macOS / Linux
- 硬盘 **2GB+** 可用空间（模型约 95MB + 依赖约 1GB）

### 2. 克隆项目

```bash
git clone https://github.com/RDJ-stank/Enterprise-RAG-System.git
cd Enterprise-RAG-System/enterprise_rag
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 下载 Embedding 模型

```bash
python setup_model.py
```

> 模型约 95MB，从 HuggingFace 镜像下载。如果网络受限，可手动从 [BAAI/bge-small-zh-v1.5](https://hf-mirror.com/BAAI/bge-small-zh-v1.5) 下载全部文件放入 `models/bge-small-zh-v1.5/`。

### 5. 配置 API Key

```bash
copy .env.example .env
```

用文本编辑器打开 `.env`，填入你的 [DeepSeek API Key](https://platform.deepseek.com)：

```
DEEPSEEK_API_KEY=sk-你的真实Key
```

### 6. 启动后端

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

看到 `Application startup complete.` 即启动成功。

### 7. 启动前端（打开新终端）

```bash
streamlit run streamlit_app.py --server.port 8501
```

### 8. 开始使用

浏览器打开 **http://localhost:8501**

1. 左侧上传文档（支持 PDF / TXT / DOCX / CSV / XLSX / XLS）
2. 等待解析入库（显示 `N 个文本块已入库`）
3. 在聊天框输入问题，回车

---

## 项目架构

```
enterprise_rag/
├── main.py                      # FastAPI 应用入口
├── config.py                    # 全局配置
├── streamlit_app.py             # Streamlit 聊天界面
├── setup_model.py               # 首次启动前下载模型
│
├── api/                         # API 层
│   ├── routes/
│   │   ├── upload.py            # POST /upload
│   │   ├── chat.py              # POST /chat
│   │   └── documents.py         # GET/DELETE /documents
│   ├── schemas.py               # Pydantic 数据模型
│   └── dependencies.py          # 依赖注入 & 单例管理
│
├── services/                    # 业务编排层
│   ├── ingest_service.py        # 文档摄入流水线
│   ├── retrieval_service.py     # 检索流水线
│   └── generation_service.py    # RAG Prompt 组装 & LLM 生成
│
├── infrastructure/              # 基础设施层
│   ├── embedding.py             # Embedding 引擎
│   ├── vector_store.py          # ChromaDB 向量存储
│   ├── llm_client.py            # DeepSeek API 客户端
│   ├── document_loader.py       # PDF/TXT 加载器
│   └── text_splitter.py         # 文本分块器
│
├── models/                      # Embedding 模型（setup_model.py 下载）
├── chroma_db/                   # ChromaDB 数据（自动生成）
├── .env                         # 环境变量（需手动配置）
└── requirements.txt             # Python 依赖清单
```

## 技术栈

| 组件 | 选型 |
|------|------|
| Web 框架 | FastAPI |
| 向量数据库 | ChromaDB（本地持久化） |
| 文档处理 | LangChain（PyMuPDF + PDFPlumber + fitz 三级回退） |
| Embedding | BAAI/bge-small-zh-v1.5（512 维，本地运行） |
| LLM | DeepSeek API（deepseek-chat） |
| 前端 | Streamlit |

## 数据流

```
文档上传 → DocumentLoader → TextSplitter → Embedding → ChromaDB
                                                          ↑
用户提问 → Embedding → ChromaDB 检索 → RAG Prompt 组装 → DeepSeek → 回答
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查 |
| `POST` | `/upload` | 上传文档（multipart/form-data） |
| `POST` | `/chat` | 知识库问答 |
| `GET` | `/documents` | 文档列表 |
| `DELETE` | `/documents/{id}` | 删除指定文档 |
| `DELETE` | `/documents` | 一键删除所有文档 |

启动后端后访问 `http://localhost:8000/docs` 查看完整 API 文档（Swagger UI）。

## 常见问题

| 现象 | 解决方法 |
|------|----------|
| 上传 PDF 后块数为 0 | PDF 是扫描件，不含文字层，请更换可提取文字的 PDF |
| 回答"无法回答" | 知识库无相关文档，或上传的文档块数为 0 |
| 端口被占用 | `netstat -ano \| findstr :8000` 找到 PID 后结束 |
| DeepSeek 报错 | 检查 `.env` 中 API Key 是否正确 |

## License

MIT
