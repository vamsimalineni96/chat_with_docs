from dotenv import load_dotenv

load_dotenv()
from src.utils.vectorstore import get_vectorstore_handler
from src.utils.add_schema_docs import add_to_db
from src.utils.create_schema_docs import create_docs
from fastapi import FastAPI

app = FastAPI()
vector_store = get_vectorstore_handler()


@app.get("/prune_schema")
async def prune_schema(question: str, db_name: str):
    result = vector_store.query(question=question, db_name=db_name)
    return result

@app.post("/add_schema_docs")
async def add_schema_docs():
    add_to_db()

@app.post("/create_docs")
async def create():
    create_docs()

@app.get("/view_chunks")
async def get_case_data():
    """Fetch all chunks stored in ChromaDB for a given case_id from both 'documents' and 'chat_history' collections."""

    try:
        # Fetch documents from the 'documents' collection
        case_documents = vector_store.collection.get() or {}
        # Extracting document texts and metadata
        documents = case_documents.get("documents", [])
        doc_metadata = case_documents.get("metadatas", [])
        # Return combined results
        return {
            "total_documents": len(documents),
            "documents": documents,
            "document_metadata": doc_metadata,
        }

    except Exception as e:
        return {"error": str(e)}

