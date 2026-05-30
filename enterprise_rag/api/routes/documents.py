import logging

from fastapi import APIRouter, HTTPException

from api.dependencies import get_vector_store
from api.schemas import DeleteResponse, DocumentInfo, DocumentListResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """列出所有已入库文档及其元数据。"""
    vector_store = get_vector_store()
    try:
        docs = vector_store.list_documents()
    except RuntimeError as exc:
        logger.exception("获取文档列表失败")
        raise HTTPException(status_code=500, detail=f"数据库异常: {exc}") from exc

    return DocumentListResponse(
        documents=[DocumentInfo(**d) for d in docs],
        total=len(docs),
    )


@router.delete("/documents/{doc_id}", response_model=DeleteResponse)
async def delete_document(doc_id: str):
    """删除指定文档及其所有向量数据。"""
    vector_store = get_vector_store()
    try:
        removed = vector_store.delete_by_doc_id(doc_id)
    except RuntimeError as exc:
        logger.exception("删除文档失败: %s", doc_id)
        raise HTTPException(status_code=500, detail=f"数据库异常: {exc}") from exc

    if removed == 0:
        raise HTTPException(status_code=404, detail="文档不存在")

    return DeleteResponse(doc_id=doc_id, status="deleted", chunks_removed=removed)
