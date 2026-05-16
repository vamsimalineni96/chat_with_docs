import uuid
from datetime import datetime
from typing import Any

from langchain_milvus import BM25BuiltInFunction, Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import DataType, MilvusClient

from src.utils import config
from src.utils.errors import CacheError, EmbeddingError, MilvusError
from src.utils.services.embedder import EmbeddingHandler
from src.utils.services.logger_config import logger


class MilvusStoreHandler:
    """Hybrid (dense + BM25) Milvus store via langchain-milvus.

    Schema (auto-created by langchain-milvus on first use):
      - pk            : INT64 auto-generated primary key
      - text          : VARCHAR (the chunk text; BM25 reads from this)
      - dense         : FLOAT_VECTOR(EMBED_DIM) populated by NVIDIAEmbeddings
      - sparse        : SPARSE_FLOAT_VECTOR populated by BM25BuiltInFunction
      - + dynamic metadata fields (doc_id, source, chunk_order, ...)

    Search is hybrid: dense + sparse with reciprocal-rank fusion. BM25 catches
    exact term matches (proper nouns, IDs, jargon) that pure dense embeddings
    miss; this is critical for sensitive/unseen documents where the embedder
    has no domain knowledge to lean on.

    Existing single-vector collections from the previous schema are NOT
    compatible. You must drop the old collection and re-ingest. Use
    POST /clear_milvus?name=<collection> via the API.
    """

    def __init__(
        self,
        uri: str | None = None,
        token: str | None = None,
        collection_name: str | None = None,
        embedder: EmbeddingHandler | None = None,
    ) -> None:
        self.uri = uri or config.MILVUS_URI
        self.token = token or getattr(config, "MILVUS_TOKEN", None)
        self.collection_name = collection_name or config.COLLECTION_NAME
        self.embed_dim = config.EMBED_DIM
        self.top_k = config.TOP_K

        self.embedder = embedder or EmbeddingHandler()

        self._connection_args: dict[str, Any] = {"uri": self.uri}
        if self.token:
            self._connection_args["token"] = self.token

        try:
            self.client = (
                MilvusClient(uri=self.uri, token=self.token)
                if self.token
                else MilvusClient(uri=self.uri)
            )
        except Exception as e:
            logger.exception("Failed to initialize Milvus client: %s", e)
            raise MilvusError("Failed to initialize Milvus client.") from e

        # The langchain `Milvus` wrapper is lazily constructed on first use
        # (see `_get_vector_store`). Keeping it lazy means admin operations
        # like `delete_collection` and `view_collection` never trigger the
        # langchain init — important when an existing collection has a stale
        # schema (e.g. left over from a previous code version).
        self._vector_store: Milvus | None = None

    def ensure_collection(self) -> None:
        """
        No-op kept for backwards-compatible callers. Collection creation and
        loading is handled by langchain-milvus on first add_texts / search.
        """
        return None

    def _get_vector_store(self) -> Milvus:
        """
        Build the langchain `Milvus` wrapper on first call. If the collection
        already exists with an incompatible schema, this is where the failure
        will surface — but only when something actually needs to read/write
        vectors, not on routine admin paths.
        """
        if self._vector_store is not None:
            return self._vector_store

        try:
            self._vector_store = Milvus(
                embedding_function=self.embedder.embeddings,
                builtin_function=BM25BuiltInFunction(
                    input_field_names="text",
                    output_field_names="sparse",
                ),
                vector_field=["dense", "sparse"],
                text_field="text",
                collection_name=self.collection_name,
                connection_args=self._connection_args,
                auto_id=True,
                enable_dynamic_field=True,
                consistency_level=config.MILVUS_CONSISTENCY_LEVEL,
            )
        except Exception as e:
            logger.exception("Failed to initialize hybrid Milvus vectorstore: %s", e)
            raise MilvusError(
                "Failed to initialize hybrid Milvus vectorstore. "
                "If the collection was created with the old single-vector schema, "
                "drop it via POST /clear_milvus and re-ingest."
            ) from e
        return self._vector_store

    def store_in_milvus(
        self,
        text: str,
        doc_id: str | None = None,
        source: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        """
        Split a long text, embed each chunk, and insert into Milvus via the
        langchain vectorstore.
        """
        self.ensure_collection()

        if doc_id is None:
            doc_id = str(uuid.uuid4())
        if source is None:
            source = "inline_text"
        if chunk_size is None:
            chunk_size = config.CHUNK_SIZE
        if chunk_overlap is None:
            chunk_overlap = config.CHUNK_OVERLAP

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", ".", "!", "?", " "],
        )
        chunks = splitter.split_text(text)
        if not chunks:
            logger.warning("No chunks produced from text; skipping insert.")
            return

        metadatas = [
            {"doc_id": doc_id, "source": source, "chunk_order": order}
            for order, _ in enumerate(chunks)
        ]

        BATCH = config.INSERT_BATCH_SIZE
        try:
            for i in range(0, len(chunks), BATCH):
                batch_texts = chunks[i : i + BATCH]
                batch_meta = metadatas[i : i + BATCH]
                self._get_vector_store().add_texts(
                    texts=batch_texts,
                    metadatas=batch_meta,
                )
                logger.info(
                    "Indexed %d / %d chunks into Milvus collection '%s'",
                    min(i + len(batch_texts), len(chunks)),
                    len(chunks),
                    self.collection_name,
                )
            self.client.flush(self.collection_name)
            logger.info("Finished indexing document into Milvus.")
        except EmbeddingError:
            raise
        except Exception as e:
            logger.exception("Milvus error during store_in_milvus: %s", e)
            raise MilvusError("Failed to store document in Milvus.") from e

    def search_similar_chunks(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Hybrid search (dense + BM25) using the query *text*. The langchain-milvus
        store handles dense embedding internally via the configured
        embedding_function, and Milvus computes BM25 on the indexed `text` field
        via the BM25BuiltInFunction. Results are fused with reciprocal rank
        fusion (RRF).

        Returns a list of dicts with {id, score, doc_id, source, chunk_order, text}.
        """
        if top_k is None:
            top_k = self.top_k

        try:
            docs_and_scores = self._get_vector_store().similarity_search_with_score(
                query=query,
                k=top_k,
                ranker_type="rrf",
                ranker_params={"k": 60},
            )
        except Exception as e:
            logger.exception("Hybrid search error in search_similar_chunks: %s", e)
            raise MilvusError("Failed to run hybrid search in Milvus.") from e

        results: list[dict[str, Any]] = []
        for doc, score in docs_and_scores:
            meta = doc.metadata or {}
            results.append(
                {
                    "id": meta.get("pk") or meta.get("id"),
                    "score": score,
                    "doc_id": meta.get("doc_id"),
                    "source": meta.get("source"),
                    "chunk_order": meta.get("chunk_order"),
                    "text": doc.page_content,
                }
            )
        return results

    def _hits_to_dicts(self, hits) -> list[dict[str, Any]]:
        """Normalize raw pymilvus hits into the same dict shape as search_similar_chunks."""
        results: list[dict[str, Any]] = []
        for h in hits:
            entity = h.get("entity") or {}
            results.append(
                {
                    "id": entity.get("pk") or h.get("id"),
                    "score": h.get("distance"),
                    "doc_id": entity.get("doc_id"),
                    "source": entity.get("source"),
                    "chunk_order": entity.get("chunk_order"),
                    "text": entity.get("text"),
                }
            )
        return results

    def _get_dense_metric_type(self) -> str:
        """
        Discover the metric type of the `dense` field's index. langchain-milvus
        creates the index with whatever its default is (typically L2), so we
        query it rather than hardcoding to avoid metric-mismatch search errors.
        Falls back to L2 (langchain-milvus default) on any lookup failure.
        """
        try:
            idx_names = self.client.list_indexes(collection_name=self.collection_name)
            for idx_name in idx_names:
                info = self.client.describe_index(
                    collection_name=self.collection_name, index_name=idx_name
                )
                if info.get("field_name") == "dense":
                    return info.get("metric_type", "L2")
        except Exception as e:
            logger.warning("Could not introspect dense index metric: %s", e)
        return "L2"

    def search_dense_only(
        self, query: str, top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Pure dense vector search (no BM25). Used for hybrid-retrieval diagnostics —
        embeds the query, ANN-searches the `dense` field, returns top_k.

        The metric type is read from the index itself; for NVIDIA's L2-normalized
        embeddings, L2 and COSINE rank identically, only the score scale differs.
        """
        if top_k is None:
            top_k = self.top_k
        try:
            query_vec = self.embedder.get_embedding(text=query, input_type="query")
            metric = self._get_dense_metric_type()
            res = self.client.search(
                collection_name=self.collection_name,
                data=[query_vec],
                anns_field="dense",
                limit=top_k,
                output_fields=["pk", "text", "doc_id", "source", "chunk_order"],
                search_params={
                    "metric_type": metric,
                    "params": {"nprobe": config.MILVUS_NPROBE},
                },
            )
            return self._hits_to_dicts(res[0])
        except Exception as e:
            logger.exception("Dense-only search failed: %s", e)
            raise MilvusError("Dense-only search failed.") from e

    def search_sparse_only(
        self, query: str, top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Pure BM25 search (no dense vectors). Milvus 2.5+ tokenizes the query
        text via the BM25 function registered on the collection and ranks
        chunks by BM25 score. Used for hybrid-retrieval diagnostics.
        """
        if top_k is None:
            top_k = self.top_k
        try:
            res = self.client.search(
                collection_name=self.collection_name,
                data=[query],
                anns_field="sparse",
                limit=top_k,
                output_fields=["pk", "text", "doc_id", "source", "chunk_order"],
                search_params={"metric_type": "BM25"},
            )
            return self._hits_to_dicts(res[0])
        except Exception as e:
            logger.exception("Sparse-only (BM25) search failed: %s", e)
            raise MilvusError("Sparse-only (BM25) search failed.") from e

    def delete_collection(self) -> None:
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
            logger.exception(
                "Error while dropping Milvus collection '%s': %s", collection_name, e
            )
            raise MilvusError("Failed to drop Milvus collection.") from e

    def view_collection(self, collection_name: str | None) -> None:
        collection_name = collection_name or self.collection_name
        try:
            collection_list = self.client.list_collections()
            print(f"List of collections: {collection_list}")

            if collection_name in collection_list:
                self.client.load_collection(collection_name=collection_name)
                rows = self.client.query(
                    collection_name=collection_name,
                    filter="",
                    output_fields=["pk", "doc_id", "source", "chunk_order", "text"],
                    limit=100,
                )
                for r in rows:
                    print("-" * 80)
                    print("pk:        ", r.get("pk"))
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

    Schema is custom (question_text, answer_text, model_name, prompt_version,
    hit_count, ...) so this stays as a direct `MilvusClient` consumer rather
    than being wrapped by `langchain-milvus`. Only the embedding call goes
    through the new langchain-backed `EmbeddingHandler`.
    """

    def __init__(
        self,
        uri: str | None = None,
        token: str | None = None,
        collection_name: str | None = None,
        embedder: EmbeddingHandler | None = None,
    ) -> None:
        self.uri = uri or config.MILVUS_URI
        self.token = token or getattr(config, "MILVUS_TOKEN", None)
        self.collection_name = collection_name or config.CACHE_COLLECTION_NAME
        self.embed_dim = config.EMBED_DIM
        self.top_k = getattr(config, "TOP_K", 5)

        try:
            self.client = (
                MilvusClient(uri=self.uri, token=self.token)
                if self.token
                else MilvusClient(uri=self.uri)
            )
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

    def ensure_collection(self) -> None:
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
                    consistency_level=config.MILVUS_CONSISTENCY_LEVEL,
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
        query_vec: list[float],
        answer_text: str,
        context_chunk_ids: list[str],
        model_name: str,
        prompt_version: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> str:
        self.ensure_collection()
        entry_id = str(uuid.uuid4())

        row: dict[str, Any] = {
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
        q_vec: list[float],
        model_name: str,
        prompt_version: str,
        top_k: int | None = None,
        min_similarity: float = 0.9,
    ) -> dict[str, Any] | None:
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
                    "params": {"nprobe": config.MILVUS_NPROBE},
                },
            )
            hits = res[0]
        except Exception as e:
            logger.exception("Milvus cache search error: %s", e)
            raise CacheError("Failed to search cache collection.") from e

        best: dict[str, Any] | None = None
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
