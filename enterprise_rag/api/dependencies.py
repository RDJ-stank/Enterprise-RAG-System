from infrastructure.embedding import EmbeddingService
from infrastructure.vector_store import VectorStoreService

from services.ingest_service import IngestService
from services.retrieval_service import RetrievalService
from services.generation_service import GenerationService

# ── 全局单例（应用启动时初始化） ──────────────────────────────
_embedding_service: EmbeddingService | None = None
_vector_store_service: VectorStoreService | None = None


def get_embedding_service() -> EmbeddingService:
    """返回 EmbeddingService 全局单例。"""
    if _embedding_service is None:
        raise RuntimeError("EmbeddingService 尚未初始化")
    return _embedding_service


def get_vector_store() -> VectorStoreService:
    """返回 VectorStoreService 全局单例。"""
    if _vector_store_service is None:
        raise RuntimeError("VectorStoreService 尚未初始化")
    return _vector_store_service


def get_ingest_service() -> IngestService:
    """返回 IngestService 实例。"""
    return IngestService(
        embedding_service=get_embedding_service(),
        vector_store=get_vector_store(),
    )


def get_retrieval_service() -> RetrievalService:
    """返回 RetrievalService 实例。"""
    return RetrievalService(
        embedding_service=get_embedding_service(),
        vector_store=get_vector_store(),
    )


def get_generation_service() -> GenerationService:
    """返回 GenerationService 实例。"""
    return GenerationService()


def init_services() -> None:
    """应用启动时调用，初始化所有基础设施单例。"""
    global _embedding_service, _vector_store_service
    _embedding_service = EmbeddingService()
    _vector_store_service = VectorStoreService()
