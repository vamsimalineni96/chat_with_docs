import os
from dotenv import load_dotenv

load_dotenv()


def _bool(env_var: str, default: bool = False) -> bool:
    v = os.getenv(env_var)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


# --- Database / Redis ---
DATABASE_URL = os.getenv("DATABASE_URL")
REDIS_URL = os.getenv("REDIS_URL")

# --- NVIDIA / NIM ---
NVIDIA_API_KEY = os.environ["NVIDIA_API_KEY"]
# `NVIDIA_BASE_URL` and `NVIDIA_RERANK_URL` are only consumed if you
# explicitly wire them into self-hosted NIM clients; for the hosted API
# the langchain-nvidia-ai-endpoints library resolves URLs internally.
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL")
NVIDIA_RERANK_URL = os.getenv("NVIDIA_RERANK_URL")

EMBED_MODEL = os.getenv("NVIDIA_EMBEDDING_MODEL")
LLM_MODEL = os.getenv("NVIDIA_LLM_MODEL")
RERANK_MODEL = os.getenv("RERANK_MODEL")

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
TOP_P = float(os.getenv("TOP_P", "0.7"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
# Bump this whenever you change prompt.yaml semantics so the cache invalidates.
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v2")

# --- Milvus ---
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "docs")
CACHE_COLLECTION_NAME = os.getenv("MILVUS_CACHE_COLLECTION_NAME", "qa_cache")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))
MILVUS_NPROBE = int(os.getenv("MILVUS_NPROBE", "16"))
MILVUS_CONSISTENCY_LEVEL = os.getenv("MILVUS_CONSISTENCY_LEVEL", "Strong")

# --- RAG / chunking ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
# How many chunks Milvus pulls per query before reranking. Should be >= TOP_K.
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "25"))
# How many chunks survive reranking and end up in the LLM prompt context.
TOP_K = int(os.getenv("TOP_K", "5"))
INSERT_BATCH_SIZE = int(os.getenv("INSERT_BATCH_SIZE", "32"))

# --- Cache ---
TOGGLE_CACHE = _bool("TOGGLE_CACHE", False)
CACHE_MIN_SIMILARITY = float(os.getenv("CACHE_MIN_SIMILARITY", "0.9"))

# --- Conversation history ---
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "20"))
HISTORY_MAX_TURNS = int(os.getenv("HISTORY_MAX_TURNS", "6"))

# --- App / runtime ---
THREAD_POOL_MAX_WORKERS = int(os.getenv("THREAD_POOL_MAX_WORKERS", "4"))
DEFAULT_CONVERSATION_TITLE = os.getenv("DEFAULT_CONVERSATION_TITLE", "New conversation")
METRICS_DOMAIN = os.getenv("METRICS_DOMAIN", "default")
PDF_DIR = os.getenv("PDF_DIR", "pdfs")

# --- Redis lock ---
REDIS_LOCK_PREFIX = os.getenv("REDIS_LOCK_PREFIX", "lock:conversation:")
REDIS_LOCK_TTL_SECONDS = int(os.getenv("REDIS_LOCK_TTL_SECONDS", "60"))
