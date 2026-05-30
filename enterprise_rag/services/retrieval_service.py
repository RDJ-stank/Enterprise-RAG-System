import logging

from infrastructure.embedding import EmbeddingService
from infrastructure.vector_store import VectorStoreService

logger = logging.getLogger(__name__)


class RetrievalService:
    """检索编排服务。

    将用户问题向量化，并在向量库中检索最相关的文档片段。
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStoreService,
    ) -> None:
        self._embedding = embedding_service
        self._vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
    ) -> list[dict]:
        """检索与查询最相关的文档片段。

        Args:
            query: 用户原始提问。
            top_k: 返回的最相关片段数量。

        Returns:
            [{chunk_text, filename, score, doc_id, chunk_index}, ...]
        """
        query_embedding = self._embedding.embed_texts([query])[0]
        results = self._vector_store.search(query_embedding, top_k=top_k)
        logger.info("检索完成: query_len=%d, top_k=%d, hits=%d", len(query), top_k, len(results))
        return results
