from dotenv import load_dotenv

load_dotenv()
import os
import json
import httpx
import traceback
import numpy as np 

from prune_src.utils.inference import SchemaPrune
from prune_src.utils.config import SEGREGATED_DATASET

from chat_src.eval_flow import EvalFlow
from chat_src.utils.logger_config import LoggerConfig
from chat_src.utils.models import RagEval
from chat_src.utils.config import EVAL_OUTPUT

from fastapi import FastAPI

app = FastAPI()
logger = LoggerConfig().logger

def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] JSON error on line {i}: {e}")


@app.post("/prune_schema")
async def prune_schema(model: str, comp: str):
    schema_pruner = SchemaPrune(model=model)

    input_path = os.path.join(SEGREGATED_DATASET, f"{comp}.jsonl")
    output_path = os.path.join(SEGREGATED_DATASET, f"{comp}_pruned_{model}_test.jsonl")

    # Open output file in write mode (overwrite). Use "a" if you want to append.
    num_processed = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for obj in read_jsonl(path=input_path):
            id = obj.get("id")
            db_id = obj.get("db_id")
            question = obj.get("question")
            db_schema = obj.get("db_schema")
            query = obj.get("query")

            pruned_schema, _ = schema_pruner.run(
                user_query=question,
                schema_info=db_schema,
            )

            # Whatever structure you want to save per line
            out_obj = {
                "id": id,
                "db_id": db_id,
                "question": question,
                "db_schema": pruned_schema,
                "query": query,
            }
            # --- write to JSONL *every iteration* ---
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                # Optional: force flush to disk for safety
                f.flush()
                os.fsync(f.fileno())
            # ----------------------------------------

            num_processed += 1

    return {
        "message": "Pruning completed",
        "num_examples": num_processed,
        "output_path": output_path,
    }


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


@app.post("/nim_inference")
async def run_nim_inference(comp_type:str, model: str):
    shot = 0
    os.makedirs(EVAL_OUTPUT, exist_ok=True)

    if comp_type == "easy":
        path = os.path.join(SEGREGATED_DATASET, "easy_pruned_llama.jsonl")
        print(path)
    if comp_type == "medium":
        path = os.path.join(SEGREGATED_DATASET, "medium_pruned_llama.jsonl")
        print(path)
    if comp_type == "hard":
        path = os.path.join(SEGREGATED_DATASET, "hard_pruned_llama.jsonl")
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

    logger.info(f"{comp_type} Avg latency: {np.mean(latencies):.2f} ms")
    logger.info(f"{comp_type} P95 latency: {np.percentile(latencies, 95):.2f} ms")
    logger.info(f"{comp_type} P99 latency: {np.percentile(latencies, 99):.2f} ms")

    return {"evaluations_saved_to": op_file}

