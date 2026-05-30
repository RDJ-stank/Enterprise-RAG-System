import logging
import tempfile
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}
LOADER_MAP = {
    ".pdf": PyMuPDFLoader,
    ".txt": TextLoader,
}


class DocumentLoaderService:
    """统一文档加载器。

    根据文件扩展名自动路由到对应的 LangChain Loader，
    支持 PDF 和 TXT 格式，返回 LangChain Document 列表。
    """

    @staticmethod
    def load(file_path: str | Path) -> list[Document]:
        """加载文档，提取原始文本。

        Args:
            file_path: 待加载的文件路径。

        Returns:
            LangChain Document 对象列表。PDF 的每页为一个 Document，
            TXT 整体为一个 Document。

        Raises:
            ValueError: 文件格式不支持或文件不存在。
            RuntimeError: 文档解析失败。
        """
        path = Path(file_path)
        if not path.exists():
            raise ValueError(f"文件不存在: {file_path}")

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"不支持的文件格式 '{ext}'，仅支持: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        loader_cls = LOADER_MAP[ext]
        logger.info("加载文档: %s (格式=%s)", path.name, ext)

        try:
            if ext == ".txt":
                loader = loader_cls(str(path), encoding="utf-8")
            else:
                loader = loader_cls(str(path))
            docs = loader.load()
        except Exception:
            logger.exception("文档解析失败: %s", path.name)
            raise RuntimeError(f"文档解析失败: {path.name}") from None

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
