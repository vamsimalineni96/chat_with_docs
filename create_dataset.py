import json
import os
from typing import List, Dict
from collections import Counter
from src.utils.config import DATA
from src.utils.sql_handler import DatabaseHandler


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_to_json(data: List[Dict], file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def save_to_jsonl(data: List[Dict], file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        for entry in data:
            json_line = json.dumps(entry, ensure_ascii=False)
            f.write(json_line + "\n")


def create_dataset(inp_file, op_file, num_ques):
    file_path = os.path.join(DATA, inp_file)

    print("Loading the complete training dataset")
    complete_train_data = load_json(file_path)
    train_data = []

    i = 1
    for c_instance in complete_train_data:

        db_handler = DatabaseHandler(c_instance.get("db_id"))
        train_data.append(
            {
                "db_id": c_instance.get("db_id"),
                "query": c_instance.get("query"),
                "question": c_instance.get("question"),
                "db_schema": json.loads(db_handler.get_db_schema_json()),
            }
        )

    db_ids = [item["db_id"] for item in train_data if "db_id" in item]
    count = dict(Counter(db_ids))
    print(f"Creating the dataset where the number of questions are > {num_ques}")
    final_data = []

    for key, value in count.items():
        if value > num_ques:
            print(key, value)
            for instance in train_data:
                if instance.get("db_id") == key:
                    final_data.append(
                        {
                            "id": i,
                            "db_id": instance.get("db_id"),
                            "output": instance.get("query"),
                            "input": instance.get("question"),
                            "context": instance.get("db_schema"),
                        }
                    )
                    i += 1

    jsonl_data = []
    for element in final_data:
        prompt = f"DATABASE SCHEMA: {element.get("context")} Question: {element.get("input")}"
        jsonl_data.append({"input": prompt, "output": element.get("output")})

    name, ext = os.path.splitext(op_file)
    if ext.lower() == ".json":
        print("Saving the dataset to a json file")
        save_to_json(data=final_data, file_path=os.path.join(DATA, op_file))
    elif ext.lower() == ".jsonl":
        print("Saving the dataset to a jsonl file")
        save_to_jsonl(data=jsonl_data, file_path=os.path.join(DATA, op_file))


create_dataset(
    inp_file="train_spider.json", op_file="fine_tune_dataset.jsonl", num_ques=1
)
