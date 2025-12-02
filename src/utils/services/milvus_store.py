import uuid
from typing import List, Dict, Any, Optional
from pymilvus import MilvusClient, DataType
from datetime import datetime

from src.utils import config
from src.utils.services.embedder import EmbeddingHandler
from src.utils.services.logger_config import logger


class MilvusStoreHandler:
    def __init__(
        self,
        uri: Optional[str] = None,
        token: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedder: Optional[EmbeddingHandler] = None,
    ) -> None:
        """
        Wrapper around MilvusClient plus embedding + PDF utilities.
        """
        self.uri = uri or config.MILVUS_URI
        self.token = token or getattr(config, "MILVUS_TOKEN", None)
        self.collection_name = collection_name or config.COLLECTION_NAME
        self.embed_dim = config.EMBED_DIM
        self.top_k = config.TOP_K

        self.client = self._get_milvus_client()
        self.embedder = embedder or EmbeddingHandler()

        # Ensure collection exists and is loaded
        self.ensure_collection()

    def _get_milvus_client(self) -> MilvusClient:
        """
        Return a MilvusClient pointing to the standalone server.
        """
        if self.token:
            return MilvusClient(uri=self.uri, token=self.token)
        return MilvusClient(uri=self.uri)

    def ensure_collection(self) -> None:
        """
        Create the collection and index if it doesn't exist, then load it.
        """
        if not self.client.has_collection(self.collection_name):
            schema = MilvusClient.create_schema(
                auto_id=False,
                enable_dynamic_field=True,  # store extra metadata in $meta
            )
            schema.add_field(
                field_name="id",
                datatype=DataType.VARCHAR,
                is_primary=True,
                max_length=64,
            )
            schema.add_field(
                field_name="embedding",
                datatype=DataType.FLOAT_VECTOR,
                dim=self.embed_dim,
            )

            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                consistency_level="Strong",
            )

            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                index_type="AUTOINDEX",
                metric_type="COSINE",
            )

            self.client.create_index(
                collection_name=self.collection_name,
                index_params=index_params,
            )

        # Always make sure collection is loaded before use
        self.client.load_collection(collection_name=self.collection_name)

    def store_in_milvus(
        self,
        text: str,
        doc_id: Optional[str] = None,
        source: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ) -> None:
        """
        Ingest a single long text into Milvus:
        text -> chunks -> embeddings -> Milvus.

        doc_id and source are optional metadata fields.
        """
        self.ensure_collection()

        if doc_id is None:
            doc_id = str(uuid.uuid4())
        if source is None:
            source = "inline_text"

        # Use your existing embedder to get vectors + chunk strings
        vectors, chunks = self.embedder.get_document_embeddings(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            long_text=text,
        )

        if len(vectors) != len(chunks):
            raise ValueError(
                f"Mismatch between vectors ({len(vectors)}) and chunks ({len(chunks)})"
            )

        BATCH = 32
        total = len(chunks)
        for i in range(0, total, BATCH):
            batch_vectors = vectors[i : i + BATCH]
            batch_chunks = chunks[i : i + BATCH]

            rows: List[Dict[str, Any]] = []
            for order, (vec, chunk_text) in enumerate(
                zip(batch_vectors, batch_chunks), start=i
            ):
                rows.append(
                    {
                        "id": str(uuid.uuid4()),
                        "embedding": vec,
                        "doc_id": doc_id,
                        "source": source,
                        "chunk_order": order,
                        "text": chunk_text,
                    }
                )

            self.client.insert(
                collection_name=self.collection_name,
                data=rows,
            )
            print(f"Indexed {min(i + len(batch_chunks), total)} / {total} chunks")

        self.client.flush(self.collection_name)
        print("Finished indexing.")

    def search_similar_chunks(
        self,
        query_vec: List[float],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Embed the query with NIM and search similar chunks in Milvus.
        Returns a list of dicts with id, score, doc_id, source, chunk_order, text.
        """
        if top_k is None:
            top_k = self.top_k

        self.client.load_collection(collection_name=self.collection_name)

        # Embed query with input_type="query"
        # query_vec = self.embedder.get_embedding(text=query)

        res = self.client.search(
            collection_name=self.collection_name,
            data=[query_vec],
            anns_field="embedding",
            limit=top_k,
            output_fields=["doc_id", "source", "chunk_order", "text"],
            search_params={
                "metric_type": "COSINE",
                "params": {"nprobe": 16},
            },
        )

        hits = res[0]  # single query
        results: List[Dict[str, Any]] = []
        for h in hits:
            row = {
                "id": h["id"],
                "score": h["distance"],
                "doc_id": h.get("doc_id"),
                "source": h.get("source"),
                "chunk_order": h.get("chunk_order"),
                "text": h.get("text"),
            }
            results.append(row)
        return results

    def delete_collection(self) -> None:
        """
        Drop the configured collection after releasing it from memory.
        """
        collection_name = self.collection_name

        state = self.client.get_load_state(collection_name=collection_name)
        if state.get("state") == "Loaded":
            print(f"Releasing collection '{collection_name}' from memory...")
            self.client.release_collection(collection_name=collection_name)
            print("Collection released.")

        print(f"Dropping collection '{collection_name}'...")
        self.client.drop_collection(collection_name=collection_name)
        print(f"Collection '{collection_name}' dropped successfully.")

    def view_collection(self, collection_name: Optional[str]) -> None:
        """
        Print a small sample of rows from the given (or default) collection.
        """
        collection_name = collection_name or self.collection_name
        collection_list = self.client.list_collections()
        print(f"List of collections: {collection_list}")

        if collection_name in collection_list:
            self.client.load_collection(collection_name=collection_name)
            rows = self.client.query(
                collection_name=collection_name,
                filter="",  # no filter -> everything
                output_fields=["id", "doc_id", "source", "chunk_order", "text"],
                limit=100,  # just peek at first 5
            )

            for r in rows:
                print("-" * 80)
                print("id:        ", r.get("id"))
                print("doc_id:    ", r.get("doc_id"))
                print("source:    ", r.get("source"))
                print("order:     ", r.get("chunk_order"))
                print("text[0:200]:")
                print((r.get("text") or "")[:200], "...")
        else:
            print("collection is empty")


class CacheStoreHandler:
    """
    Semantic QA cache in Milvus.

    - Each entry is a (question embedding, answer, metadata) row.
    - We search this collection first before doing full RAG.
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        token: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedder: Optional[EmbeddingHandler] = None,
    ) -> None:
        """
        Wrapper around MilvusClient + embedding for QA cache.
        """
        self.uri = uri or config.MILVUS_URI
        self.token = token or getattr(config, "MILVUS_TOKEN", None)

        # use a separate collection from your document chunks
        self.collection_name = collection_name or config.CACHE_COLLECTION_NAME

        self.embed_dim = config.EMBED_DIM
        self.top_k = getattr(config, "TOP_K", 5)

        self.client = self._get_milvus_client()
        self.embedder = embedder or EmbeddingHandler()

        # Make sure the cache collection exists and is loaded
        logger.info("Ensuring cache store has the collection")
        self.ensure_collection()

    def _get_milvus_client(self) -> MilvusClient:
        """
        Return a MilvusClient pointing to the standalone server.
        """
        if self.token:
            return MilvusClient(uri=self.uri, token=self.token)
        return MilvusClient(uri=self.uri)

    def ensure_collection(self) -> None:
        """
        Create the cache collection and index if it doesn't exist, then load it.
        """
        if not self.client.has_collection(self.collection_name):
            schema = MilvusClient.create_schema(
                auto_id=False,
                enable_dynamic_field=True,  # store extra metadata in $meta
            )
            schema.add_field(
                field_name="id",
                datatype=DataType.VARCHAR,
                is_primary=True,
                max_length=64,
            )
            schema.add_field(
                field_name="embedding",
                datatype=DataType.FLOAT_VECTOR,
                dim=self.embed_dim,
            )

            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                consistency_level="Strong",
            )

            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                index_type="AUTOINDEX",
                metric_type="COSINE",
            )

            self.client.create_index(
                collection_name=self.collection_name,
                index_params=index_params,
            )

        # Always make sure collection is loaded before use
        logger.info(f"Loading the cache collection: {self.collection_name}")
        self.client.load_collection(collection_name=self.collection_name)

    def put_entry(
        self,
        question_text: str,
        query_vec: List[float],
        answer_text: str,
        context_chunk_ids: List[str],
        model_name: str,
        prompt_version: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store a QA cache entry: (question embedding + answer + metadata).
        Returns the new entry id.
        """
        self.ensure_collection()

        # Embed the question for semantic lookup
        # q_vec = self.embedder.get_embedding(text=question_text)
        entry_id = str(uuid.uuid4())

        row: Dict[str, Any] = {
            "id": entry_id,
            "embedding": query_vec,
            # query side
            "question_text": question_text,
            "question_norm": question_text.strip().lower(),
            # answer side
            "answer_text": answer_text,
            # retrieval context
            "context_chunk_ids": context_chunk_ids,
            # generation config
            "model_name": model_name,
            "prompt_version": prompt_version,
            "temperature": temperature,
            "max_tokens": max_tokens,
            # bookkeeping
            "created_at": datetime.utcnow().isoformat(),
            "hit_count": 0,
            "last_hit_at": None,
        }

        if extra_metadata:
            # Allow caller to stuff anything else into dynamic meta
            row.update(extra_metadata)

        self.client.insert(
            collection_name=self.collection_name,
            data=[row],
        )
        self.client.flush(self.collection_name)
        return entry_id

    def search_similar(
        self,
        query: str,
        q_vec: List[float],
        model_name: str,
        prompt_version: str,
        top_k: Optional[int] = None,
        min_similarity: float = 0.9,
    ) -> Optional[Dict[str, Any]]:
        """
        Semantic QA cache lookup.

        - Embeds `query`
        - Vector search in cache collection
        - Filters by model_name, prompt_version, style
        - Enforces similarity threshold `min_similarity` (0-1)

        Returns the best matching entry dict or None.
        """
        if top_k is None:
            top_k = self.top_k

        self.client.load_collection(collection_name=self.collection_name)

        # q_vec = self.embedder.get_embedding(text=query)

        res = self.client.search(
            collection_name=self.collection_name,
            data=[q_vec],
            anns_field="embedding",
            limit=top_k,
            output_fields=[
                "question_text",
                "question_norm",
                "answer_text",
                "context_chunk_ids",
                "model_name",
                "prompt_version",
                "temperature",
                "max_tokens",
                "created_at",
                "hit_count",
                "last_hit_at",
            ],
            search_params={
                "metric_type": "COSINE",
                "params": {"nprobe": 16},
            },
        )

        hits = res[0]  # single query
        best: Optional[Dict[str, Any]] = None

        for h in hits:
            # Milvus with COSINE typically uses distance = 1 - cosine_similarity
            distance = h["distance"]
            similarity = 1.0 - float(distance)

            meta = {
                "id": h["id"],
                "distance": distance,
                "similarity": similarity,
                "question_text": h.get("question_text"),
                "question_norm": h.get("question_norm"),
                "answer_text": h.get("answer_text"),
                "context_chunk_ids": h.get("context_chunk_ids") or [],
                "model_name": h.get("model_name"),
                "prompt_version": h.get("prompt_version"),
                "temperature": h.get("temperature"),
                "max_tokens": h.get("max_tokens"),
                "created_at": h.get("created_at"),
                "hit_count": h.get("hit_count"),
                "last_hit_at": h.get("last_hit_at"),
            }
            logger.info(
                f"[CACHE CANDIDATE] sim={similarity:.4f}, "
                f"model={meta['model_name']}, prompt={meta['prompt_version']}, "
                f"q='{meta['question_text']}'"
            )

            if (
                similarity >= min_similarity
                and meta["model_name"] == model_name
                and meta["prompt_version"] == prompt_version
            ):
                best = meta
                break

        if best is None:
            logger.info(f"[CACHE MISS] no suitable entry for query='{query}'")

        return best

    def delete_collection(self) -> None:
        """
        Drop the cache collection (for resets / migrations).
        """
        collection_name = self.collection_name

        state = self.client.get_load_state(collection_name=collection_name)
        if state.get("state") == "Loaded":
            print(f"Releasing cache collection '{collection_name}' from memory...")
            self.client.release_collection(collection_name=collection_name)
            print("Collection released.")

        print(f"Dropping cache collection '{collection_name}'...")
        self.client.drop_collection(collection_name=collection_name)
        print(f"Cache collection '{collection_name}' dropped successfully.")


_cache_store_instance = None


def get_cache_store():
    global _cache_store_instance
    if _cache_store_instance is None:
        _cache_store_instance = CacheStoreHandler()
    return _cache_store_instance


_milvus_store_instance = None


def get_milvus_store():
    global _milvus_store_instance
    if _milvus_store_instance is None:
        _milvus_store_instance = MilvusStoreHandler()
    return _milvus_store_instance
