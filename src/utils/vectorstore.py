import chromadb
from chromadb.config import Settings
from src.utils.embedder import EmbeddingHandler
import uuid

from src.utils.logger_config import LoggerConfig


# Initialize the logger
logger_config = LoggerConfig()
logger = logger_config.logger


class VectorStoreHandler:
    def __init__(
        self,
        persist_directory: str = "./chroma_data",  # Local folder for DB files
        collection_name: str = "documents",
    ):
        logger.info(
            "Initializing VectorStoreHandler with local ChromaDB at %s, collection=%s",
            persist_directory,
            collection_name,
        )

        self.embedder = EmbeddingHandler()

        try:
            # Connect to local ChromaDB (PersistentClient stores data on disk)
            self.chroma_client = chromadb.PersistentClient(
                path=persist_directory, settings=Settings()
            )

            # Create or get a local collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=collection_name
            )

            logger.info("Connected to local ChromaDB successfully.")

        except Exception as e:
            logger.error("Failed to initialize local ChromaDB: %s", str(e))
            self.chroma_client = None
            self.collection = None

    def delete_by_case_id(self, collection_name: str, target_case_id: str) -> int:
        # Iterate over each collection and delete documents matching the case_id
        try:
            collection = self.chroma_client.get_collection(name=collection_name)
            collection.delete(where={"db_name": target_case_id})
            logger.info(
                f"Deleted documents with case_id '{target_case_id}' from collection '{collection_name}'."
            )
        except Exception as e:
            print(f"Error processing collection '{collection_name}': {e}")

    def add_summary(self, db_name: str, text: str, table_name: str):
        embeddings, text_chunks = self.embedder.get_summary_embeddings(text)
        try:
            for chunk, embedding in zip(text_chunks, embeddings):
                doc_id = str(uuid.uuid4())
                self.collection.add(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{"db_name": db_name, "table_name": table_name}],
                )
            logger.info("Successfully added the documents to chromadb")
        except Exception as e:
            logger.error(f"Failed to add documents to chromadb: {e}")

    def query_old(self, query: str, db_name: str):
        embeddings, _ = self.embedder.get_summary_embeddings(query)
        results = self.collection.query(
            query_embeddings=embeddings,
            n_results=5,
            where={"db_name": db_name},
        )
        table_name = results.get("metadatas")[0][0].get("table_name")
        table_doc = results.get("documents")[0]

        return {"results": results, "table_name": table_name, "table_docs": table_doc}

    def query(self, question: str, db_name: str, n_results: int = 50, top_k_tables: int = 6):
        """
        Query Chroma for a single DB, aggregate by table, and return pruned tables.
        Assumes each vector's metadata has: {"db_name": <str>, "table_name": <str>, ...}
        """
        # 1) Embed the question
        q_emb, _ = self.embedder.get_summary_embeddings(question)

        # 2) Query only inside this DB via metadata filter
        res = self.collection.query(
            query_embeddings=q_emb,
            n_results=n_results,
            where={"db_name": db_name},                  # <-- restrict to one DB
        )

        # Safety: empty results
        docs_list      = res.get("documents", [])
        metas_list     = res.get("metadatas", [])
        dists_list     = res.get("distances", [])
        ids_list       = res.get("ids", [])
        if not docs_list or not docs_list[0]:
            return {
                "pruned_tables": [],
                "top_table": None,
                "top_hits": [],
                "raw": res,
            }

        docs   = docs_list[0]
        metas  = metas_list[0]
        dists  = dists_list[0] if dists_list else [None] * len(docs)
        ids    = ids_list[0] if ids_list else [None] * len(docs)

        # 3) Aggregate by table using max-pool of a monotonic score
        #    Convert distance -> score; if distance missing, fall back to rank-based score
        by_table = {}
        table_hits = {}  # keep best doc per table for debugging/inspection

        for rank, (doc, meta, dist, _id) in enumerate(zip(docs, metas, dists, ids)):
            table = meta.get("table_name") or meta.get("table")  # be tolerant to key naming
            if not table:
                continue
            if dist is None:
                score = 1.0 / (1.0 + rank)  # degrade gracefully if distances unavailable
            else:
                score = 1.0 / (1e-6 + float(dist))
            # max-pool per table
            if (table not in by_table) or (score > by_table[table]):
                by_table[table] = score
                table_hits[table] = {
                    "doc": doc,
                    "meta": meta,
                    "distance": dist,
                    "id": _id,
                    "score": score,
                }

        # 4) Pick top-k tables
        pruned_tables = [t for t, _ in sorted(by_table.items(), key=lambda x: x[1], reverse=True)[:top_k_tables]]
        top_table = pruned_tables[0] if pruned_tables else None

        # 5) Prepare a compact debug payload of top hits per table
        top_hits = [
            {
                "table_name": t,
                "score": table_hits[t]["score"],
                "distance": table_hits[t]["distance"],
                "doc": table_hits[t]["doc"],
                "meta": table_hits[t]["meta"],
                "id": table_hits[t]["id"],
            }
            for t in pruned_tables
        ]

        return {
            "pruned_tables": pruned_tables,
            "top_table": top_table,
            "top_hits": top_hits,  # one best hit per selected table
            "raw": res,            # keep raw results for downstream debugging if needed
        }

_vectorstore_instance = None


def get_vectorstore_handler():
    global _vectorstore_instance
    if _vectorstore_instance is None:
        _vectorstore_instance = VectorStoreHandler()
    return _vectorstore_instance
