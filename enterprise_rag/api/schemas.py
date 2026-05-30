from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ── /upload ─────────────────────────────────────────────────

class UploadResponse(BaseModel):
    doc_id: str = Field(..., description="唯一文档标识符 (UUID)")
    filename: str = Field(..., description="原始文件名")
    file_size: int = Field(..., description="文件大小（字节）")
    chunk_count: int = Field(..., description="切分后的文本块数量")
    status: str = Field(..., description="处理状态: indexed")


# ── /chat ───────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="用户提问内容")
    top_k: int = Field(default=4, ge=1, le=10, description="检索返回的最相关文档片段数量")


class SourceInfo(BaseModel):
    chunk_text: str = Field(..., description="检索到的文档片段原文")
    filename: str = Field(..., description="来源文件名")
    score: float = Field(..., description="相似度分数 (0~1)")


class ChatResponse(BaseModel):
    answer: str = Field(..., description="基于文档生成的回答")
    sources: list[SourceInfo] = Field(default_factory=list, description="参考来源列表")


# ── /documents ──────────────────────────────────────────────

class DocumentInfo(BaseModel):
    doc_id: str = Field(..., description="文档唯一标识符")
    filename: str = Field(..., description="文件名")
    chunk_count: int = Field(..., description="文本块数量")
    uploaded_at: datetime = Field(..., description="上传时间")
    file_size: int = Field(..., description="文件大小（字节）")


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo] = Field(default_factory=list, description="文档列表")
    total: int = Field(..., description="文档总数")


# ── /documents/{doc_id} ─────────────────────────────────────

class DeleteResponse(BaseModel):
    doc_id: str = Field(..., description="被删除的文档标识符")
    status: str = Field(..., description="操作状态: deleted")
    chunks_removed: int = Field(..., description="移除的向量条目数量")


# ── Error ───────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    detail: str = Field(..., description="错误详情描述")
