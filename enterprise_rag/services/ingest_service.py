import logging
import uuid
from datetime import datetime, timezone

from infrastructure.document_loader import DocumentLoaderService
from infrastructure.embedding import EmbeddingService
from infrastructure.text_splitter import TextSplitterService
from infrastructure.vector_store import VectorStoreService

logger = logging.getLogger(__name__)


class IngestService:
    """文档摄入编排服务。

    串联 加载 → 分块 → 向量化 → 存储 的完整流水线。
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStoreService,
    ) -> None:
        self._embedding = embedding_service
        self._vector_store = vector_store

    def ingest(
        self,
        file_path: str,
        filename: str,
        file_size: int,
    ) -> dict:
        """执行完整文档摄入流水线。

        Args:
            file_path: 临时文件的物理路径。
            filename: 用户上传的原始文件名。
            file_size: 文件大小（字节）。

        Returns:
            {doc_id, filename, file_size, chunk_count, status}
        """
        doc_id = uuid.uuid4().hex
        uploaded_at = datetime.now(timezone.utc).isoformat()

        # 1. 加载文档
        raw_docs = DocumentLoaderService.load(file_path)
        if not raw_docs:
            logger.warning("文档 %s 解析后无内容", filename)
            return {
                "doc_id": doc_id,
                "filename": filename,
                "file_size": file_size,
                "chunk_count": 0,
                "status": "empty",
            }

        # 2. 文本分块
        chunks = TextSplitterService.split(raw_docs)

        # 3. 向量化
        chunk_texts = [chunk.page_content for chunk in chunks]
        embeddings = self._embedding.embed_texts(chunk_texts)

        # 4. 构建元数据并写入向量库
        metadatas = [
            {
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": i,
                "uploaded_at": uploaded_at,
                "file_size": file_size,
                "source_page": chunks[i].metadata.get("page", -1),
            }
            for i in range(len(chunks))
        ]
        self._vector_store.add_documents(
            doc_id=doc_id,
            chunks=chunk_texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(
            "文档摄入完成: doc_id=%s, filename=%s, chunks=%d",
            doc_id,
            filename,
            len(chunks),
        )

        return {
            "doc_id": doc_id,
            "filename": filename,
            "file_size": file_size,
            "chunk_count": len(chunks),
            "status": "indexed",
        }
