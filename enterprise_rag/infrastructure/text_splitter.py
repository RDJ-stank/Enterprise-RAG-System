import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

# 中文友好的分隔符优先级：段落 → 换行 → 句号 → 逗号 → 空格 → 字符
CN_SEPARATORS = ["\n\n", "\n", "。", "，", " ", ""]


class TextSplitterService:
    """文本分块服务。

    基于 LangChain RecursiveCharacterTextSplitter，采用中文友好的分隔符层级，
    将长文档切分为适合 Embedding 模型的文本片段。
    """

    @staticmethod
    def split(
        documents: list[Document],
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[Document]:
        """将文档列表切分为文本块。

        Args:
            documents: 待切分的 LangChain Document 列表。
            chunk_size: 每个文本块的最大字符数。默认从 config.CHUNK_SIZE 读取 (500)。
            chunk_overlap: 相邻文本块的重叠字符数。默认从 config.CHUNK_OVERLAP 读取 (50)。

        Returns:
            切分后的 Document 列表，每个 Document 的 metadata 继承自原始文档。
        """
        cs = chunk_size or CHUNK_SIZE
        co = chunk_overlap or CHUNK_OVERLAP

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cs,
            chunk_overlap=co,
            separators=CN_SEPARATORS,
            length_function=len,
            is_separator_regex=False,
        )

        chunks = splitter.split_documents(documents)
        logger.info(
            "文本分块完成: %d 个原始文档 → %d 个文本块 (chunk_size=%d, overlap=%d)",
            len(documents),
            len(chunks),
            cs,
            co,
        )
        return chunks
