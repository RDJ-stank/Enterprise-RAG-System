import csv
import io
import logging
from pathlib import Path

import fitz  # PyMuPDF
import openpyxl
from docx import Document as DocxDocument
from langchain_community.document_loaders import (
    PDFPlumberLoader,
    PyMuPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".pdf", ".txt", ".docx", ".csv", ".xlsx", ".xls",
}


class DocumentLoaderService:
    """统一文档加载器。

    支持格式：
      PDF  — 三级回退策略 (PyMuPDF → PDFPlumber → fitz 逐页)
      TXT  — TextLoader (UTF-8)
      DOCX — python-docx 直接提取段落文字
      CSV  — Python csv 模块逐行读取
      XLSX — openpyxl 逐单元格提取
      XLS  — openpyxl (仅支持新版 .xls，旧版 97-2003 需 xlrd)
    """

    @staticmethod
    def load(file_path: str | Path) -> list[Document]:
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

        # 路由到对应 loader
        if ext == ".pdf":
            docs = _load_pdf(str(path))
        elif ext == ".txt":
            docs = _load_txt(str(path))
        elif ext == ".docx":
            docs = _load_docx(str(path))
        elif ext == ".csv":
            docs = _load_csv(str(path))
        elif ext in (".xlsx", ".xls"):
            docs = _load_xlsx(str(path))
        else:
            docs = []

        # 过滤空页面
        docs = [d for d in docs if d.page_content and d.page_content.strip()]
        if not docs:
            logger.warning(
                "文档 %s 未提取到任何有效文字", path.name,
            )
            return []

        for doc in docs:
            doc.metadata["filename"] = path.name

        logger.info("文档加载完成: %s, 共 %d 片段", path.name, len(docs))
        return docs


# ── TXT ────────────────────────────────────────────────────

def _load_txt(file_path: str) -> list[Document]:
    try:
        loader = TextLoader(file_path, encoding="utf-8")
        return loader.load()
    except Exception:
        logger.exception("TXT 解析失败: %s", file_path)
        raise RuntimeError(f"TXT 文件解析失败: {file_path}") from None


# ── PDF ────────────────────────────────────────────────────

def _load_pdf(file_path: str) -> list[Document]:
    try:
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        logger.info("PDF 由 PyMuPDFLoader 成功加载")
        return docs
    except Exception:
        logger.warning("PyMuPDFLoader 失败，切换到 PDFPlumberLoader ...")

    try:
        loader = PDFPlumberLoader(file_path)
        docs = loader.load()
        logger.info("PDF 由 PDFPlumberLoader 成功加载")
        return docs
    except Exception:
        logger.warning("PDFPlumberLoader 也失败，切换到 fitz 逐页容错 ...")

    try:
        docs = _load_pdf_page_by_page(file_path)
        if docs:
            logger.info("PDF 由 fitz 逐页容错成功加载 (%d 页)", len(docs))
            return docs
    except Exception:
        pass

    raise RuntimeError(
        f"PDF 解析失败: {Path(file_path).name}。"
        f"已尝试 PyMuPDF / PDFPlumber / fitz 三种方式均无法解析。"
    )


def _load_pdf_page_by_page(file_path: str) -> list[Document]:
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
            failed_pages.append(page_num + 1)

    pdf.close()

    if failed_pages:
        logger.warning("fitz 逐页提取时 %d 页失败，已跳过", len(failed_pages))

    return docs


# ── DOCX ───────────────────────────────────────────────────

def _load_docx(file_path: str) -> list[Document]:
    try:
        doc = DocxDocument(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        if not paragraphs:
            logger.warning("DOCX 文件无文字: %s", file_path)
            return []
        text = "\n\n".join(paragraphs)
        return [Document(page_content=text, metadata={"source": file_path})]
    except Exception:
        logger.exception("DOCX 解析失败: %s", file_path)
        raise RuntimeError(f"DOCX 文件解析失败: {Path(file_path).name}") from None


# ── CSV ────────────────────────────────────────────────────

def _load_csv(file_path: str) -> list[Document]:
    try:
        # 先尝试 UTF-8，失败则用 GBK
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, encoding="gbk") as f:
                content = f.read()

        reader = csv.reader(io.StringIO(content))
        rows = list(reader)
        if not rows:
            return []

        # 每行用 tab 连接各列，行与行之间换行
        lines = [" | ".join(r) for r in rows if any(c.strip() for c in r)]
        text = "\n".join(lines)
        return [Document(page_content=text, metadata={"source": file_path})]
    except Exception:
        logger.exception("CSV 解析失败: %s", file_path)
        raise RuntimeError(f"CSV 文件解析失败: {Path(file_path).name}") from None


# ── XLSX / XLS ─────────────────────────────────────────────

def _load_xlsx(file_path: str) -> list[Document]:
    sheets_text: list[str] = []
    try:
        wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
    except Exception:
        logger.exception("openpyxl 无法打开: %s", file_path)
        raise RuntimeError(f"Excel 文件解析失败: {Path(file_path).name}") from None

    try:
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            rows = []
            row_count = 0
            for row in ws.iter_rows(values_only=True):
                cells = [str(c) if c is not None else "" for c in row]
                if any(c.strip() for c in cells):
                    rows.append(" | ".join(cells))
                    row_count += 1
            if rows:
                sheet_header = f"--- 工作表: {sheet_name} ---"
                sheets_text.append(sheet_header + "\n" + "\n".join(rows))
            logger.info("工作表 '%s': %d 行", sheet_name, row_count)
    finally:
        wb.close()

    if not sheets_text:
        return []
    full_text = "\n\n".join(sheets_text)
    return [Document(page_content=full_text, metadata={"source": file_path})]
