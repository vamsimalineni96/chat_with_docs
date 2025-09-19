from dotenv import load_dotenv

load_dotenv()

import os
import traceback
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("files:", os.listdir())

from src.utils.logger_config import LoggerConfig
from src.utils.models import RagCase
from src.utils.vectorstore import get_vectorstore_handler

from src.rag_flow import RagFlow


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = LoggerConfig().logger


@app.post("/chat")
async def chat(lead: RagCase):
    rag_pipeline = RagFlow()
    
    try:
        logger.info("Generating the response")

        try:
            rag_pipeline.state["user_query"] = lead.question
        except Exception as e:
            raise Exception(f"Failed to set pipeline state: {str(e)}")

        # Step 3: Kick off the pipeline
        try:
            result = await rag_pipeline.kickoff_async()
        except Exception as e:
            raise Exception(f"Pipeline execution failed: {str(e)}")

        return {"user_query": lead.question, "reply": result}

    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Error in chatbot_evidence: {str(e)}\n{error_trace}")
        return {"error": str(e)}

@app.get("/view_chunks")
async def get_case_data():
    """Fetch all chunks stored in ChromaDB for a given case_id from both 'documents' and 'chat_history' collections."""

    vector_store = get_vectorstore_handler()
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

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
