from prune_src.utils.config import TEST_DATASET, SEGREGATED_DATASET
import json
import os

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

hard_path = os.path.join(SEGREGATED_DATASET, "hard.jsonl")

for obj in read_jsonl(path=hard_path):
    question=obj.get("question")
    db_schema = obj.get("db_schema")
    print(question, type(question))
    print(db_schema, type(db_schema))
    break