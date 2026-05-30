import logging

from fastapi import APIRouter, HTTPException

from api.dependencies import get_retrieval_service, get_generation_service
from api.schemas import ChatRequest, ChatResponse, SourceInfo

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    """接收用户提问，执行向量检索，调用 DeepSeek 生成答案。

    返回回答及参考文档片段，供前端展示引用来源。
    """
    retrieval_service = get_retrieval_service()
    generation_service = get_generation_service()

    try:
        sources = retrieval_service.retrieve(
            query=body.question,
            top_k=body.top_k,
        )
    except RuntimeError as exc:
        logger.exception("检索失败")
        raise HTTPException(status_code=500, detail=f"检索服务异常: {exc}") from exc

    try:
        answer = await generation_service.generate(
            question=body.question,
            sources=sources,
        )
    except RuntimeError as exc:
        logger.exception("LLM 生成失败")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception:
        logger.exception("LLM 生成未知错误")
        raise HTTPException(status_code=502, detail="LLM 服务暂不可用，请稍后重试")

    source_infos = [
        SourceInfo(
            chunk_text=src["chunk_text"],
            filename=src["filename"],
            score=src["score"],
        )
        for src in sources
    ]

    return ChatResponse(answer=answer, sources=source_infos)
