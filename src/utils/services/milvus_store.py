import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

from pymilvus import MilvusClient, DataType
from src.utils import config
from src.utils.errors import MilvusError, CacheError, EmbeddingError
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

        Raises:
            MilvusError: if the client or collection initialization fails.
        """
        self.uri = uri or config.MILVUS_URI
        self.token = token or getattr(config, "MILVUS_TOKEN", None)
        self.collection_name = collection_name or config.COLLECTION_NAME
        self.embed_dim = config.EMBED_DIM
        self.top_k = config.TOP_K

        try:
            self.client = self._get_milvus_client()
        except Exception as e:
            logger.exception("Failed to initialize Milvus client: %s", e)
            raise MilvusError("Failed to initialize Milvus client.") from e

        self.embedder = embedder or EmbeddingHandler()

        try:
            self.ensure_collection()
        except Exception as e:
            logger.exception(
                "Failed to ensure Milvus collection '%s': %s",
                self.collection_name,
                e,
            )
            raise MilvusError("Failed to initialize Milvus collection.") from e

    def _get_milvus_client(self) -> MilvusClient:
        """Return a MilvusClient pointing to the standalone server."""
        if self.token:
            return MilvusClient(uri=self.uri, token=self.token)
        return MilvusClient(uri=self.uri)

    def ensure_collection(self) -> None:
        """
        Create the collection and index if it doesn't exist, then load it.

        Raises:
            MilvusError: on Milvus failures.
        """
        try:
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

            self.client.load_collection(collection_name=self.collection_name)
        except Exception as e:
            logger.exception(
                "Error ensuring/creating Milvus collection '%s': %s",
                self.collection_name,
                e,
            )
            raise MilvusError("Failed to ensure Milvus collection.") from e

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

        Raises:
            EmbeddingError: on embedding failures.
            MilvusError: on Milvus insert/flush issues.
        """
        self.ensure_collection()

        if doc_id is None:
            doc_id = str(uuid.uuid4())
        if source is None:
            source = "inline_text"

        try:
            vectors, chunks = self.embedder.get_document_embeddings(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                long_text=text,
            )
        except EmbeddingError:
            # Let callers know embedding failed specifically
            raise
        except Exception as e:
            logger.exception("Unexpected error while embedding document: %s", e)
            raise EmbeddingError("Unexpected error while embedding document.") from e

        if len(vectors) != len(chunks):
            raise MilvusError(
                f"Mismatch between vectors ({len(vectors)}) and chunks ({len(chunks)})"
            )

        BATCH = 32
        total = len(chunks)
        try:
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
                logger.info(
                    "Indexed %d / %d chunks into Milvus collection '%s'",
                    min(i + len(batch_chunks), total),
                    total,
                    self.collection_name,
                )

            self.client.flush(self.collection_name)
            logger.info("Finished indexing document into Milvus.")
        except Exception as e:
            logger.exception("Milvus error during store_in_milvus: %s", e)
            raise MilvusError("Failed to store document in Milvus.") from e

    def search_similar_chunks(
        self,
        query_vec: List[float],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search similar chunks in Milvus using a precomputed query_vec.

        Raises:
            MilvusError: if search fails.
        """
        if top_k is None:
            top_k = self.top_k

        try:
            self.client.load_collection(collection_name=self.collection_name)

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
        except Exception as e:
            logger.exception("Milvus search error in search_similar_chunks: %s", e)
            raise MilvusError("Failed to search similar chunks in Milvus.") from e

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

        Raises:
            MilvusError: if drop fails.
        """
        collection_name = self.collection_name

        try:
            state = self.client.get_load_state(collection_name=collection_name)
            if state.get("state") == "Loaded":
                logger.info(
                    "Releasing collection '%s' from memory before dropping...",
                    collection_name,
                )
                self.client.release_collection(collection_name=collection_name)

            logger.info("Dropping collection '%s'...", collection_name)
            self.client.drop_collection(collection_name=collection_name)
            logger.info("Collection '%s' dropped successfully.", collection_name)
        except Exception as e:
            logger.exception("Error while dropping Milvus collection '%s': %s", collection_name, e)
            raise MilvusError("Failed to drop Milvus collection.") from e

    def view_collection(self, collection_name: Optional[str]) -> None:
        """
        Print a small sample of rows from the given (or default) collection.

        Raises:
            MilvusError: on errors querying Milvus.
        """
        collection_name = collection_name or self.collection_name

        try:
            collection_list = self.client.list_collections()
            print(f"List of collections: {collection_list}")

            if collection_name in collection_list:
                self.client.load_collection(collection_name=collection_name)
                rows = self.client.query(
                    collection_name=collection_name,
                    filter="",  # no filter -> everything
                    output_fields=["id", "doc_id", "source", "chunk_order", "text"],
                    limit=100,
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
        except Exception as e:
            logger.exception(
                "Error while viewing Milvus collection '%s': %s", collection_name, e
            )
            raise MilvusError("Failed to view Milvus collection.") from e


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

        Raises:
            MilvusError: if cache collection fails to initialize.
        """
        self.uri = uri or config.MILVUS_URI
        self.token = token or getattr(config, "MILVUS_TOKEN", None)
        self.collection_name = collection_name or config.CACHE_COLLECTION_NAME
        self.embed_dim = config.EMBED_DIM
        self.top_k = getattr(config, "TOP_K", 5)

        try:
            self.client = self._get_milvus_client()
        except Exception as e:
            logger.exception("Failed to initialize Milvus cache client: %s", e)
            raise MilvusError("Failed to initialize Milvus cache client.") from e

        self.embedder = embedder or EmbeddingHandler()

        logger.info("Ensuring cache store has the collection '%s'", self.collection_name)
        try:
            self.ensure_collection()
        except Exception as e:
            logger.exception(
                "Failed to ensure cache collection '%s': %s",
                self.collection_name,
                e,
            )
            raise MilvusError("Failed to initialize cache collection.") from e

    def _get_milvus_client(self) -> MilvusClient:
        if self.token:
            return MilvusClient(uri=self.uri, token=self.token)
        return MilvusClient(uri=self.uri)

    def ensure_collection(self) -> None:
        """Create the cache collection and index if it doesn't exist, then load it."""
        try:
            if not self.client.has_collection(self.collection_name):
                schema = MilvusClient.create_schema(
                    auto_id=False,
                    enable_dynamic_field=True,
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

            logger.info("Loading the cache collection: %s", self.collection_name)
            self.client.load_collection(collection_name=self.collection_name)
        except Exception as e:
            logger.exception("Error ensuring cache collection '%s': %s", self.collection_name, e)
            raise MilvusError("Failed to ensure cache collection.") from e

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

        Raises:
            CacheError: if insert fails.
        """
        self.ensure_collection()
        entry_id = str(uuid.uuid4())

        row: Dict[str, Any] = {
            "id": entry_id,
            "embedding": query_vec,
            "question_text": question_text,
            "question_norm": question_text.strip().lower(),
            "answer_text": answer_text,
            "context_chunk_ids": context_chunk_ids,
            "model_name": model_name,
            "prompt_version": prompt_version,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "created_at": datetime.utcnow().isoformat(),
            "hit_count": 0,
            "last_hit_at": None,
        }

        if extra_metadata:
            row.update(extra_metadata)

        try:
            self.client.insert(
                collection_name=self.collection_name,
                data=[row],
            )
            self.client.flush(self.collection_name)
            return entry_id
        except Exception as e:
            logger.exception("Error inserting cache entry into Milvus: %s", e)
            raise CacheError("Failed to store cache entry.") from e

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

        Raises:
            CacheError: if search fails.
        """
        if top_k is None:
            top_k = self.top_k

        try:
            self.client.load_collection(collection_name=self.collection_name)

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
            hits = res[0]
        except Exception as e:
            logger.exception("Milvus cache search error: %s", e)
            raise CacheError("Failed to search cache collection.") from e

        best: Optional[Dict[str, Any]] = None
        for h in hits:
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
                "[CACHE CANDIDATE] sim=%.4f, model=%s, prompt=%s, q='%s'",
                similarity,
                meta["model_name"],
                meta["prompt_version"],
                meta["question_text"],
            )

            if (
                similarity >= min_similarity
                and meta["model_name"] == model_name
                and meta["prompt_version"] == prompt_version
            ):
                best = meta
                break

        if best is None:
            logger.info("[CACHE MISS] no suitable entry for query='%s'", query)

        return best

    def delete_collection(self) -> None:
        """Drop the cache collection (for resets / migrations)."""
        collection_name = self.collection_name
        try:
            state = self.client.get_load_state(collection_name=collection_name)
            if state.get("state") == "Loaded":
                print(f"Releasing cache collection '{collection_name}' from memory...")
                self.client.release_collection(collection_name=collection_name)
                print("Collection released.")

            print(f"Dropping cache collection '{collection_name}'...")
            self.client.drop_collection(collection_name=collection_name)
            print(f"Cache collection '{collection_name}' dropped successfully.")
        except Exception as e:
            logger.exception(
                "Error while dropping cache collection '%s': %s", collection_name, e
            )
            raise CacheError("Failed to drop cache collection.") from e


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
