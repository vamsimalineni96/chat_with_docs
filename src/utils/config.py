# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# Database config
DATABASE_URL = os.getenv("DATABASE_URL")
# Redis config
REDIS_URL = os.getenv("REDIS_URL")

# NIM / NVIDIA config
NVIDIA_API_KEY = os.environ["NVIDIA_API_KEY"]
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL")

EMBED_MODEL = os.getenv("NVIDIA_EMBEDDING_MODEL", "nvidia/nv-embed-v1")
LLM_MODEL = os.getenv("NVIDIA_LLM_MODEL", "meta/llama-3.3-70b-instruct")
TEMPERATURE = float(os.getenv("TEMPERATURE",0.1))
TOP_P = float(os.getenv("TOP_P",0.7))
MAX_TOKENS = int(os.getenv("MAX_TOKENS",1024))

# Milvus config (standalone server in Docker)
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")  # empty string = no auth

COLLECTION_NAME = "rag_nim_milvus"
EMBED_DIM = 4096  # nv-embed-v1 dimension

# RAG / chunking config
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5
