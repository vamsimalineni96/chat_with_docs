import json
import os
from src.config import PROJECT_ROOT
from src.utils.config import EVAL_OUTPUT
input_file = os.path.join(PROJECT_ROOT,EVAL_OUTPUT,"test_results.json")
output_file = os.path.join(PROJECT_ROOT,EVAL_OUTPUT,"test_results.jsonl")

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)   # load full JSON (expects a list at the top level)

with open(output_file, "w", encoding="utf-8") as f:
    for obj in data:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
