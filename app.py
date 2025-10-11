from dotenv import load_dotenv

load_dotenv()

import os
import json
import httpx
import traceback
import numpy as np
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from collections import defaultdict

from src.utils.logger_config import LoggerConfig
from src.utils.models import (
    RagCase,
    RagEval,
    DbSummarize,
    DeleteCaseRequest,
    NimEvalType,
)
from src.utils.vectorstore import get_vectorstore_handler
from src.utils.summarize_db_schema import run_summarizer
from src.utils.config import (
    DATA,
    EVAL_OUTPUT,
    SEGREGATED_DATASET,
    FINETUNE_INPUTS,
    FINETUNE_OUTPUTS,
)
from src.utils.inference import Sql2Text
from src.utils.sql_handler import DatabaseHandler

from src.rag_flow import RagFlow
from src.eval_flow import EvalFlow

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

@app.post("/get_ground_truths")
async def get_ground_truth():
    with open(os.path.join(DATA, "final_train_data.json"), "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = defaultdict(list)
    nl_questions = defaultdict(list)
    ids = defaultdict(list)

    for i in data:
        questions[i.get("db_id")].append(i.get("query"))
        nl_questions[i.get("db_id")].append(i.get("question"))
        if i.get("id") is not None:
            ids[i.get("id")].append(i.get("id"))

    results = []
    op_file = "results_gt.json"
    sql2nl = Sql2Text()  # your model wrapper

    for item in data:
        key = item.get("db_id")
        query = item.get("query")
        question = item.get("question")
        key_id = item.get("id")

        db_hander = DatabaseHandler(db_name=key)

        try:
            sql_answer = db_hander.execute_command(query=query)

            # ✅ Catch LLM / JSON parsing errors
            try:
                response = sql2nl.run(
                    user_query=question, sql_answer=sql_answer, sql_query=query
                )
                results.append({"id": key_id, "question": question, "reply": response})
            except Exception as llm_err:
                # If model output is invalid
                results.append(
                    {
                        "id": key_id,
                        "question": question,
                        "error": f"LLM error: {str(llm_err)}",
                    }
                )

        except Exception as db_err:
            # If DB query itself fails
            results.append(
                {"id": key_id, "question": query, "error": f"DB error: {str(db_err)}"}
            )

        # ✅ Save after each iteration so partial progress is never lost
        save_to_json(data=results, file_path=os.path.join(EVAL_OUTPUT, op_file))

    return results

# Endpoint to communicate with the nim hosted model
@app.post("/chat_eval")
async def chat_eval(lead: RagEval):
    eval_pipeline = EvalFlow()

    try:
        logger.info("Generating the response")

        try:
            eval_pipeline.state["user_query"] = lead.question
            eval_pipeline.state["db_schema"] = lead.db_schema
            eval_pipeline.state["db_name"] = lead.db_name
            eval_pipeline.state["model"] = lead.model
            eval_pipeline.state["shot"] = lead.shot

        except Exception as e:
            raise Exception(f"Failed to set pipeline state: {str(e)}")

        # Step 3: Kick off the pipeline
        try:
            result = await eval_pipeline.kickoff_async()
        except Exception as e:
            raise Exception(f"Pipeline execution failed: {str(e)}")

        return {
            "user_query": lead.question,
            "reply": result.get("sql_answer"),
            "generated_query": result.get("sql_query"),
            "latency": result.get("latency"),
        }

    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Error in chatbot_evidence: {str(e)}\n{error_trace}")
        return {"error": str(e)}

# Batch endpoint to run and execute the commands generated by nims hosted models for different prompting techniques 0 3 8  shot prompts
@app.post("/nim_inference")
async def run_nim_inference():
    comp_types = ["easy", "medium", "hard"]
    shots = [3, 8]
    models = ["gemma", "llama"]

    for comp_type in comp_types:
        for shot in shots:
            for model in models:
                if comp_type == "easy":
                    path = os.path.join(SEGREGATED_DATASET, "easy.jsonl")
                    print(path)
                if comp_type == "medium":
                    path = os.path.join(SEGREGATED_DATASET, "medium.jsonl")
                    print(path)
                if comp_type == "hard":
                    path = os.path.join(SEGREGATED_DATASET, "hard.jsonl")
                    print(path)

                # Load questions from a local JSONL file
                data = []
                with open(path, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f, start=1):
                        line = line.strip()
                        if not line:  # skip empty lines
                            continue
                        try:
                            obj = json.loads(line)
                            data.append(obj)
                        except json.JSONDecodeError as e:
                            print(f"Error in line {i}: {e}")

                op_file = os.path.join(
                    EVAL_OUTPUT, f"{model}_{shot}_shot_test_results_{comp_type}.jsonl"
                )
                # Keeping track of the latencies
                latencies = []
                # Create an async HTTP client to hit the /chat_eval endpoint
                async with httpx.AsyncClient() as client:
                    with open(op_file, "w", encoding="utf-8") as outfile:
                        for item in data:
                            try:
                                payload = {
                                    "question": str(item.get("question")),
                                    "db_schema": str(item.get("db_schema")),
                                    "db_name": str(item.get("db_id")),
                                    "model": model,
                                    "shot": shot,
                                }
                                response = await client.post(
                                    "http://localhost:8000/chat_eval",
                                    json=payload,
                                    timeout=60,
                                )
                                if response.status_code == 200:
                                    latencies.append(response.json().get("latency"))
                                    result = {
                                        "id": item.get("id"),
                                        "question": item.get("question"),
                                        "reply": response.json().get("reply"),
                                        "generated_query": response.json().get(
                                            "generated_query"
                                        ),
                                    }
                                else:
                                    result = {
                                        "question": item.get("question"),
                                        "error": response.text,
                                    }
                            except Exception as e:
                                result = {
                                    "question": item.get("question"),
                                    "error": str(e),
                                }
                                latencies.append(None)

                            # Write each result immediately as JSONL
                            outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
                            outfile.flush()

                latencies = [l for l in latencies if l is not None]

                logger.info(f"Avg latency: {np.mean(latencies):.2f} ms")
                logger.info(f"P95 latency: {np.percentile(latencies, 95):.2f} ms")
                logger.info(f"P99 latency: {np.percentile(latencies, 99):.2f} ms")

    return {"evaluations_saved_to": op_file}


# Endpoint to execute the generated queries from finetuned gemma
@app.post("/finetune_execute_queries")
async def execute_finetune_generated_queries(comp_type:str):
    data = []
    path = os.path.join(FINETUNE_INPUTS, f"fgemma_results_{comp_type}.jsonl")
    print(path)
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:  # skip empty lines
                continue
            try:
                obj = json.loads(line)
                data.append(obj)
            except json.JSONDecodeError as e:
                print(f"Error in line {i}: {e}")

    op_file = os.path.join(FINETUNE_OUTPUTS, f"fgemma_results_{comp_type}.jsonl")
    with open(op_file, "w", encoding="utf-8") as outfile:
        for datapoint in data:
            test_database_handler = DatabaseHandler(
                db_name=datapoint.get("db_id"), test=True
            )
            generated_query = datapoint.get("generated_query")
            reply = test_database_handler.execute_command(query=generated_query)

            result = {
                "id": datapoint.get("id"),
                "question": datapoint.get("question"),
                "reply": reply,
                "generated_query": datapoint.get("generated_query"),
            }
            outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
            outfile.flush()
    return{"message":"Completed"}

# @app.post("/evaluate_prompt")
# async def evaluate_prompt():

#     # Load questions from a local JSON file
#     with open(os.path.join(DATA, "final_train_data.json"), "r", encoding="utf-8") as f:
#         data = json.load(f)

#     questions = {}
#     for i in data:
#         questions[i.get("id")] = i.get("question")

#     results = []
#     op_file = "results.json"

#     # Create an async HTTP client to hit the /chat endpoint
#     async with httpx.AsyncClient() as client:
#         for key, value in questions.items():
#             try:
#                 payload = {"question": value}
#                 response = await client.post(
#                     "http://localhost:8000/chat",  # or your deployed URL
#                     json=payload,
#                     timeout=60,
#                 )
#                 if response.status_code == 200:
#                     results.append(
#                         {
#                             "id": key,
#                             "question": value,
#                             "reply": response.json().get("reply"),
#                         }
#                     )
#                 else:
#                     results.append({"question": value, "error": response.text})
#             except Exception as e:
#                 results.append({"question": value, "error": str(e)})

#         save_to_json(data=results, file_path=os.path.join(EVAL_OUTPUT, op_file))

#     return {"evaluations": results}


# @app.post("/summarize_db")
# async def db_summarize(lead: DbSummarize):
#     try:
#         await run_summarizer(db_name=lead.db_name)
#         return {"message": f"Summarized the schema for {lead.db_name} database"}
#     except Exception as e:
#         logger.error("Error during summarizing db schema: {e}")
#         return {"message": "Error occurred go through the logs for resolution"}


# @app.get("/view_chunks")
# async def get_case_data():
#     """Fetch all chunks stored in ChromaDB for a given case_id from both 'documents' and 'chat_history' collections."""

#     vector_store = get_vectorstore_handler()
#     try:
#         # Fetch documents from the 'documents' collection
#         case_documents = vector_store.collection.get() or {}
#         # Extracting document texts and metadata
#         documents = case_documents.get("documents", [])
#         doc_metadata = case_documents.get("metadatas", [])
#         # Return combined results
#         return {
#             "total_documents": len(documents),
#             "documents": documents,
#             "document_metadata": doc_metadata,
#         }

#     except Exception as e:
#         return {"error": str(e)}


# @app.post("/delete_summary_data")
# def delete_summary_data(req: DeleteCaseRequest):
#     # Initialize your vector store handler
#     from src.utils.vectorstore import VectorStoreHandler

#     vs = VectorStoreHandler()

#     collections = [
#         name.strip() for name in req.collection_names.split(",") if name.strip()
#     ]
#     deleted_collections = []

#     for collection in collections:
#         try:
#             vs.delete_by_case_id(collection, req.db_name)
#             deleted_collections.append(collection)
#         except ValueError as ve:
#             raise HTTPException(
#                 status_code=400, detail=f"Invalid collection: {collection} - {ve}"
#             )
#         except Exception as e:
#             raise HTTPException(
#                 status_code=500, detail=f"Deletion failed for {collection}: {e}"
#             )

#     return {
#         "message": f"Deleted documents for case_id '{req.db_name}' from collections: {', '.join(deleted_collections)}"
#     }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
