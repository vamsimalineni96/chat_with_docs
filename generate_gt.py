import json
import os
from src.utils.config import EVAL_OUTPUT, DATA

from src.utils.sql_handler import DatabaseHandler

data = []
path = os.path.join(DATA, "test_dataset_id.jsonl")
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

op_file = os.path.join(EVAL_OUTPUT, "test_ground_truth.jsonl")
with open(op_file, "w", encoding="utf-8") as outfile:
    for item in data:
        db_name = item.get("db_id")
        sql_query = item.get("query")
        sql_handler = DatabaseHandler(db_name, test=True)
        
        print("Executing the generated sql query")
        sql_answer = sql_handler.execute_command(query=sql_query)
        result = {
            "id": item.get("id"),
            "question": item.get("question"),
            "reply": sql_answer,
            "sql_query": sql_query,
        }
        outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
        outfile.flush()
