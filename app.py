from dotenv import load_dotenv

load_dotenv()

import os
import json
import httpx
import traceback
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.utils.logger_config import LoggerConfig
from src.utils.models import RagCase, DbSummarize, DeleteCaseRequest
from src.utils.vectorstore import get_vectorstore_handler
from src.utils.summarize_db_schema import run_summarizer
from src.utils.config import DATA,EVAL_OUTPUT
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

def save_to_json(data: List[Dict], file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


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


@app.post("/evaluate")
async def evaluate_prompt():

    # Load questions from a local JSON file
    with open(os.path.join(DATA, "final_train_data.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = {}
    for i in data:
        questions[i.get("id")] = i.get("question")

    results = []
    op_file="results.json"

    # Create an async HTTP client to hit the /chat endpoint
    async with httpx.AsyncClient() as client:
        for key, value in questions.items():
            try:
                payload = {"question": value}
                response = await client.post(
                    "http://localhost:8000/chat",  # or your deployed URL
                    json=payload,
                    timeout=60,
                )
                if response.status_code == 200:
                    results.append(
                        {"question": value, "reply": response.json().get("reply")}
                    )
                else:
                    results.append({"question": value, "error": response.text})
            except Exception as e:
                results.append({"question": value, "error": str(e)})
        
        save_to_json(data=results,file_path=os.path.join(EVAL_OUTPUT, op_file))

    return {"evaluations": results}


@app.post("/summarize_db")
async def db_summarize(lead: DbSummarize):
    try:
        await run_summarizer(db_name=lead.db_name)
        return {"message": f"Summarized the schema for {lead.db_name} database"}
    except Exception as e:
        logger.error("Error during summarizing db schema: {e}")
        return {"message": "Error occurred go through the logs for resolution"}


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


@app.post("/delete_summary_data")
def delete_summary_data(req: DeleteCaseRequest):
    # Initialize your vector store handler
    from src.utils.vectorstore import VectorStoreHandler

    vs = VectorStoreHandler()

    collections = [
        name.strip() for name in req.collection_names.split(",") if name.strip()
    ]
    deleted_collections = []

    for collection in collections:
        try:
            vs.delete_by_case_id(collection, req.db_name)
            deleted_collections.append(collection)
        except ValueError as ve:
            raise HTTPException(
                status_code=400, detail=f"Invalid collection: {collection} - {ve}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Deletion failed for {collection}: {e}"
            )

    return {
        "message": f"Deleted documents for case_id '{req.db_name}' from collections: {', '.join(deleted_collections)}"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
