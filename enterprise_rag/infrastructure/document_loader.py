import logging
from pathlib import Path

import fitz  # PyMuPDF
from langchain_community.document_loaders import (
    PDFPlumberLoader,
    PyMuPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


class DocumentLoaderService:
    """统一文档加载器。

    PDF 采用三级回退策略：
      1. PyMuPDFLoader — 最快，覆盖 90% 的 PDF
      2. PDFPlumberLoader — 兼容更多内部格式
      3. fitz 逐页容错 — 跳过损坏的页面，提取剩余内容

    TXT 使用 TextLoader (UTF-8)。
    """

    @staticmethod
    def load(file_path: str | Path) -> list[Document]:
        """加载文档，提取原始文本。"""
        path = Path(file_path)
        if not path.exists():
            raise ValueError(f"文件不存在: {file_path}")

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"不支持的文件格式 '{ext}'，"
                f"仅支持: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        logger.info("加载文档: %s (格式=%s)", path.name, ext)

        if ext == ".txt":
            docs = _load_txt(str(path))
        else:
            docs = _load_pdf(str(path))

        # 过滤空页面
        docs = [d for d in docs if d.page_content and d.page_content.strip()]
        if not docs:
            logger.warning(
                "文档 %s 未提取到任何有效文字（可能为扫描件或图片PDF）",
                path.name,
            )
            return []

        # 为每个 Document 注入源文件名
        for doc in docs:
            doc.metadata["filename"] = path.name

        logger.info(
            "文档加载完成: %s, 共 %d 页/片段",
            path.name,
            len(docs),
        )
        return docs


# ── internal helpers ────────────────────────────────────────

def _load_txt(file_path: str) -> list[Document]:
    try:
        loader = TextLoader(file_path, encoding="utf-8")
        return loader.load()
    except Exception:
        logger.exception("TXT 解析失败: %s", file_path)
        raise RuntimeError(f"TXT 文件解析失败: {file_path}") from None


def _load_pdf(file_path: str) -> list[Document]:
    """三级回退加载 PDF。"""

    # 级别 1：PyMuPDFLoader（最快）
    try:
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        logger.info("PDF 由 PyMuPDFLoader 成功加载")
        return docs
    except Exception:
        logger.warning("PyMuPDFLoader 失败，切换到 PDFPlumberLoader ...")

    # 级别 2：PDFPlumberLoader（兼容性更好）
    try:
        loader = PDFPlumberLoader(file_path)
        docs = loader.load()
        logger.info("PDF 由 PDFPlumberLoader 成功加载")
        return docs
    except Exception:
        logger.warning("PDFPlumberLoader 也失败，切换到 fitz 逐页容错 ...")

    # 级别 3：fitz 逐页提取，跳过损坏的页面
    try:
        docs = _load_pdf_page_by_page(file_path)
        if docs:
            logger.info("PDF 由 fitz 逐页容错成功加载 (%d 页)", len(docs))
            return docs
    except Exception:
        pass

    raise RuntimeError(
        f"PDF 解析失败: {Path(file_path).name}。"
        f"已尝试 PyMuPDF / PDFPlumber / fitz 三种方式均无法解析，"
        f"该文件可能已损坏或为纯图片扫描件。"
    )


def _load_pdf_page_by_page(file_path: str) -> list[Document]:
    """使用 fitz 逐页提取文本，出错页面跳过，不拖累其他页面。"""
    docs: list[Document] = []
    failed_pages: list[int] = []

    try:
        pdf = fitz.open(file_path)
    except Exception:
        raise RuntimeError(f"fitz 无法打开 PDF 文件: {file_path}")

    for page_num in range(len(pdf)):
        try:
            page = pdf[page_num]
            text = page.get_text()
            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={"page": page_num, "source": file_path},
                ))
        except Exception:
            failed_pages.append(page_num + 1)  # 1-based for user

    pdf.close()

    if failed_pages:
        logger.warning(
            "fitz 逐页提取时 %d 页失败（页码: %s），已跳过",
            len(failed_pages),
            failed_pages,
        )

    return docs
