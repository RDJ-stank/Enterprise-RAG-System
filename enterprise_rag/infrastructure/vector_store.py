import logging
import os
import uuid
from datetime import datetime, timezone

import chromadb
from chromadb.api.types import QueryResult
from chromadb.config import Settings as ChromaSettings

from config import CHROMA_PERSIST_DIR

logger = logging.getLogger(__name__)

COLLECTION_NAME = "enterprise_knowledge"


class VectorStoreService:
    """ChromaDB 向量存储封装。

    管理文档向量的增、删、查操作，使用本地持久化存储。

    生命周期：应用启动时单例初始化，内部持有 ChromaDB 客户端和 Collection 引用。
    """

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        """初始化 ChromaDB 客户端并获取或创建 Collection。"""
        directory = persist_dir or CHROMA_PERSIST_DIR
        os.makedirs(directory, exist_ok=True)
        logger.info("正在连接 ChromaDB，持久化目录: %s", directory)

        self._collection_name = collection_name
        self._persist_dir = directory

        try:
            self._client = chromadb.PersistentClient(
                path=directory,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception:
            logger.exception("ChromaDB 初始化失败")
            raise RuntimeError("ChromaDB 初始化失败") from None

        logger.info(
            "ChromaDB 连接成功，Collection: %s，当前条目数: %d",
            collection_name,
            self._collection.count(),
        )

    # ── 对外查询接口 ──────────────────────────────────────────

    def add_documents(
        self,
        doc_id: str,
        chunks: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> int:
        """将文档的 chunks、向量和元数据写入 Collection。

        Args:
            doc_id: 文档唯一标识符。
            chunks: 文本块列表。
            embeddings: 每个文本块对应的向量，与 chunks 一一对应。
            metadatas: 每个文本块的元数据（含 doc_id、filename、chunk_index）。

        Returns:
            实际写入的向量条目数量。
        """
        n = len(chunks)
        if n == 0:
            return 0

        ids = [f"{doc_id}_{i}" for i in range(n)]

        try:
            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
            )
            logger.info("文档 %s 写入完成，共 %d 条向量", doc_id, n)
            return n
        except Exception:
            logger.exception("文档 %s 写入 ChromaDB 失败", doc_id)
            raise

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 4,
    ) -> list[dict]:
        """执行向量相似度检索。

        Args:
            query_embedding: 用户问题经 EmbeddingService 编码后的向量。
            top_k: 返回的最相似文档片段数量。

        Returns:
            按相似度降序排列的结果列表，每项包含:
            {chunk_text, filename, score, doc_id, chunk_index}
        """
        try:
            results: QueryResult = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            logger.exception("向量检索失败")
            raise

        # ChromaDB 返回的是嵌套列表，取第一个 query 的结果
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        hits: list[dict] = []
        for chunk_text, meta, dist in zip(docs, metas, distances):
            if chunk_text is None:
                continue
            score = 1.0 - dist if dist is not None else 0.0
            hits.append(
                {
                    "chunk_text": chunk_text,
                    "filename": (meta or {}).get("filename", "未知"),
                    "score": round(score, 4),
                    "doc_id": (meta or {}).get("doc_id", ""),
                    "chunk_index": (meta or {}).get("chunk_index", 0),
                }
            )
        return hits

    def delete_by_doc_id(self, doc_id: str) -> int:
        """删除指定文档的所有向量条目。

        Args:
            doc_id: 要删除的文档标识符。

        Returns:
            实际删除的向量条目数量。
        """
        try:
            existing = self._collection.get(
                where={"doc_id": doc_id},
                include=[],
            )
            ids_to_delete = existing.get("ids", [])
            if not ids_to_delete:
                logger.warning("文档 %s 不存在，跳过删除", doc_id)
                return 0

            self._collection.delete(ids=ids_to_delete)
            n = len(ids_to_delete)
            logger.info("文档 %s 已删除，移除 %d 条向量", doc_id, n)
            return n
        except Exception:
            logger.exception("删除文档 %s 失败", doc_id)
            raise

    def delete_all(self) -> dict:
        """删除 Collection 中的所有向量条目。

        通过删除整个 Collection 再重建的方式，确保彻底清空，
        不受 ChromaDB get() 默认返回数量限制的影响。

        Returns:
            {documents_removed, chunks_removed}
        """
        try:
            doc_count = len(self.list_documents(max_results=100_000))
            chunk_count = self._collection.count()

            if chunk_count > 0:
                self._client.delete_collection(self._collection_name)
                self._collection = self._client.get_or_create_collection(
                    name=self._collection_name,
                    metadata={"hnsw:space": "cosine"},
                )

            logger.info("已清空 Collection，共删除 %d 个文档 / %d 条向量", doc_count, chunk_count)
            return {"documents_removed": doc_count, "chunks_removed": chunk_count}
        except Exception:
            logger.exception("清空 Collection 失败")
            raise

    def list_documents(self, max_results: int = 100_000) -> list[dict]:
        """列出 Collection 中所有不重复的文档信息。

        Args:
            max_results: 最大返回的元数据条目数（防止 ChromaDB 默认限制导致漏报）。

        Returns:
            文档信息列表，每项包含: {doc_id, filename, chunk_count, uploaded_at, file_size}
        """
        try:
            all_metas = self._collection.get(
                include=["metadatas"],
                limit=max_results,
            )
        except Exception:
            logger.exception("获取文档列表失败")
            raise

        metas_list = all_metas.get("metadatas", [])
        if not metas_list:
            return []

        grouped: dict[str, dict] = {}
        for meta in metas_list:
            if meta is None:
                continue
            did = meta.get("doc_id")
            if did is None:
                continue
            if did not in grouped:
                grouped[did] = {
                    "doc_id": did,
                    "filename": meta.get("filename", "未知"),
                    "chunk_count": 0,
                    "uploaded_at": meta.get("uploaded_at", ""),
                    "file_size": meta.get("file_size", 0),
                }
            grouped[did]["chunk_count"] += 1

        return sorted(
            grouped.values(),
            key=lambda d: d["uploaded_at"],
            reverse=True,
        )

    def count(self) -> int:
        """返回 Collection 中当前向量条目总数。"""
        try:
            return self._collection.count()
        except Exception:
            logger.exception("获取 Collection 条目数失败")
            raise
