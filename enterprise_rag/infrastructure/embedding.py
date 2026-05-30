import logging
from pathlib import Path

from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME, PROJECT_ROOT

logger = logging.getLogger(__name__)


class EmbeddingService:
    """本地 Embedding 引擎。

    使用 sentence-transformers 加载开源中文模型，
    在本地执行文本向量化，不依赖外部 API。

    生命周期：应用启动时单例初始化，全局复用，避免重复加载模型。
    """

    def __init__(self, model_name: str | None = None) -> None:
        """初始化 Embedding 模型。

        Args:
            model_name: HuggingFace 模型名称或本地路径。
                        默认从 config.EMBEDDING_MODEL_NAME 读取。
                        若为本地相对路径，相对于 PROJECT_ROOT 解析。
        """
        name = model_name or EMBEDDING_MODEL_NAME

        # 若为本地相对路径，解析为绝对路径
        model_path = name
        if name.startswith("./") or name.startswith("../"):
            model_path = str(Path(PROJECT_ROOT / name).resolve())
        elif Path(name).exists():
            model_path = str(Path(name).resolve())

        logger.info("正在加载 Embedding 模型: %s ...", name)
        try:
            self._model = SentenceTransformer(model_path, trust_remote_code=True)
        except Exception:
            logger.exception("加载 Embedding 模型失败: %s", name)
            raise RuntimeError(f"无法加载 Embedding 模型: {name}") from None
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(
            "Embedding 模型加载完成: %s (维度=%d)", name, self._dimension
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """将文本列表编码为向量列表。

        Args:
            texts: 待编码的文本列表。

        Returns:
            每个文本对应的向量，维度由模型决定（m3e-base 为 768）。
            返回顺序与输入顺序一致。
        """
        if not texts:
            return []
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        """返回当前模型的向量维度。"""
        return self._dimension
