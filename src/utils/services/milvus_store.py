# milvus_store.py
from typing import List, Dict, Any, Optional
from pymilvus import MilvusClient, DataType

import uuid
from src.utils import config
from src.utils.services.embedder import EmbeddingHandler


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
        query: str,
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
        query_vec = self.embedder.get_embedding(text=query)

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
