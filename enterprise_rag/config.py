import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Project Root ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()

# ── DeepSeek API ────────────────────────────────────────────
DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

def _resolve_dir(raw: str, default: Path) -> str:
    """将可能为相对路径的目录解析为绝对路径。"""
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            return str((PROJECT_ROOT / p).resolve())
        return str(p.resolve())
    return str(default.resolve())


# ── ChromaDB ────────────────────────────────────────────────
CHROMA_PERSIST_DIR: str = _resolve_dir(
    os.getenv("CHROMA_PERSIST_DIR", ""),
    PROJECT_ROOT / "chroma_db",
)

# ── Embedding Model ─────────────────────────────────────────
EMBEDDING_MODEL_NAME: str = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "moka-ai/m3e-base",
)

# ── File Upload ─────────────────────────────────────────────
UPLOAD_DIR: str = _resolve_dir(
    os.getenv("UPLOAD_DIR", ""),
    PROJECT_ROOT / "uploads",
)
MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "20"))

# ── Text Splitting ──────────────────────────────────────────
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 50

# ── Retrieval ───────────────────────────────────────────────
DEFAULT_TOP_K: int = 4
MAX_TOP_K: int = 10

# ── LLM Generation ──────────────────────────────────────────
LLM_MODEL: str = "deepseek-chat"
LLM_TEMPERATURE: float = 0.3
