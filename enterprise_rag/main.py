import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import (
    get_embedding_service,
    get_vector_store,
    init_services,
)
from api.routes import upload, chat, documents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理。

    启动时：初始化 Embedding 模型和 ChromaDB 连接。
    关闭时：释放资源（sentence-transformers 和 ChromaDB 由 Python GC 处理）。
    """
    logger.info("===== 应用启动中 =====")
    init_services()
    logger.info("Embedding 模型已加载，Vector Store 已连接")
    logger.info("===== 应用就绪 =====")
    yield
    logger.info("===== 应用关闭 =====")


app = FastAPI(
    title="Enterprise RAG System",
    description="企业级智能知识库问答助手 — 基于 RAG 架构的文档检索与智能问答后端服务",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, tags=["文档管理"])
app.include_router(chat.router, tags=["智能问答"])
app.include_router(documents.router, tags=["文档管理"])


@app.get("/health", tags=["系统"])
async def health_check():
    """健康检查接口。"""
    try:
        emb = get_embedding_service()
        vs = get_vector_store()
        return {
            "status": "healthy",
            "embedding_model": emb.dimension,
            "vector_count": vs.count(),
        }
    except Exception:
        return {"status": "unhealthy"}
