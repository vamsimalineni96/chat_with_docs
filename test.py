import json
import os
from src.utils.sql_handler import DatabaseHandler
from typing import List, Dict
from collections import Counter
from src.utils.config import DATA
import pprint

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_to_json(data: List[Dict], file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def create_dataset(inp_file, op_file, num_ques):
    file_path = os.path.join(DATA, inp_file)

    print("Loading the complete training dataset")
    complete_train_data = load_json(file_path)
    train_data = []
    
    for instance in complete_train_data:
        train_data.append(
            {
                "db_id": instance.get("db_id"),
                "query": instance.get("query"),
                "question": instance.get("question"),
            }
        )

    db_ids = [item["db_id"] for item in train_data if "db_id" in item]
    count = dict(Counter(db_ids))
    # pprint.pprint(count)
    print(f"Creating the dataset where the number of questions are > {num_ques}")
    final_data = []
    for key, value in count.items():
        if value > num_ques:
            print(key,value)
            for instance in train_data:
                if instance.get("db_id") == key:
                    final_data.append(
                        {
                            "db_id": instance.get("db_id"),
                            "query": instance.get("query"),
                            "question": instance.get("question"),
                        }
                    )
    print("Saving the dataset to a json file")
    save_to_json(data=final_data, file_path=os.path.join(DATA, op_file))


# create_dataset(
#     inp_file="train_spider.json", op_file="final_train_data.json", num_ques=100
# )

# Using this training dataset to perform the prompt finetuning 
# training_dataset=load_json(file_path=os.path.join(DATA, "final_train_data.json"))

db_name="bike_1"
db_handler=DatabaseHandler(db_name)
db_schema=db_handler.get_db_schema_json()

import asyncio
from src.utils.summary import summarize_schema
from src.utils.vectorstore import get_vectorstore_handler

result=asyncio.run(summarize_schema(db_schema))

vector_store=get_vectorstore_handler()
vector_store.add_summary(db_name=db_name,text=result)

