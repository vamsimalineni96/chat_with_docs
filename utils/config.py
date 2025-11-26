# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# NIM / NVIDIA config
NVIDIA_API_KEY = os.environ["NVIDIA_API_KEY"]
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

EMBED_MODEL = os.getenv("NVIDIA_EMBEDDING_MODEL", "nvidia/nv-embed-v1")
LLM_MODEL = os.getenv("NVIDIA_LLM_MODEL", "meta/llama-3.3-70b-instruct")

# Milvus config (standalone server in Docker)
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")  # empty string = no auth

COLLECTION_NAME = "rag_nim_milvus"
EMBED_DIM = 4096  # nv-embed-v1 dimension

# RAG / chunking config
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5
