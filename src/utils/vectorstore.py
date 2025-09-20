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

    def add_summary(self, db_name: str, text: str):
        embeddings, text_chunks = self.embedder.get_summary_embeddings(text)
        try:    
            for chunk, embedding in zip(text_chunks, embeddings):
                doc_id = str(uuid.uuid4())
                self.collection.add(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{"db_name": db_name,}],
                )
            logger.info("Successfully added the documents to chromadb")
        except Exception as e:
            logger.error(f"Failed to add documents to chromadb: {e}")

    def query(self,query: str):
        embeddings,_ = self.embedder.get_summary_embeddings(query)
        results = self.collection.query(
                query_embeddings=embeddings,
                n_results=1
            )
        db_name = results.get("metadatas")[0][0].get("db_name")
        return db_name


_vectorstore_instance = None


def get_vectorstore_handler():
    global _vectorstore_instance
    if _vectorstore_instance is None:
        _vectorstore_instance = VectorStoreHandler()
    return _vectorstore_instance
