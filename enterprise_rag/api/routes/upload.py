import logging
import os
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from api.dependencies import get_ingest_service
from api.schemas import UploadResponse
from config import MAX_UPLOAD_SIZE_MB

logger = logging.getLogger(__name__)

router = APIRouter()
ALLOWED_EXTENSIONS = {".pdf", ".txt"}
MAX_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024


@router.post("/upload", response_model=UploadResponse, status_code=200)
async def upload(file: UploadFile = File(...)):
    """上传文档并触发解析、分块、向量化和入库操作。

    支持 PDF 和 TXT 格式，文件大小不超过配置上限。
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名为空")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型 '{ext}'。仅支持: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    content = await file.read()
    if len(content) > MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"文件大小超过 {MAX_UPLOAD_SIZE_MB}MB 限制",
        )

    suffix = ext if ext else None
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(content)
        tmp.close()

        ingest_service = get_ingest_service()
        result = await run_in_threadpool(
            ingest_service.ingest,
            file_path=tmp.name,
            filename=file.filename,
            file_size=len(content),
        )
        return UploadResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("文档摄入失败: %s", file.filename)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception:
        logger.exception("文档摄入未知错误: %s", file.filename)
        raise HTTPException(status_code=500, detail="服务器内部错误，请稍后重试")
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
