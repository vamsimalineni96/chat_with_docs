import json
import os
from src.utils.sql_handler import DatabaseHandler
from typing import List, Dict
from collections import Counter
from src.utils.config import DATA,EVAL_OUTPUT


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
    i = 1
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
            print(key, value)
            for instance in train_data:
                if instance.get("db_id") == key:
                    final_data.append(
                        {
                            "id": i,
                            "db_id": instance.get("db_id"),
                            "query": instance.get("query"),
                            "question": instance.get("question"),
                        }
                    )
                    i += 1
    print("Saving the dataset to a json file")
    save_to_json(data=final_data, file_path=os.path.join(DATA, op_file))


from src_finetune.create_dataset import DatasetCreator

creator = DatasetCreator()

creator.create_dataset(
    inp_file="train_spider.json", op_file="training_dataset.jsonl", num_ques=1
)

creator.create_validation_dataset(
    inp_file="train_spider.json",
    op_file="validation_dataset.jsonl",
    num_ques=1,
    validation_ratio=0.1,
    seed=42,
)

# create_dataset(
#     inp_file="train_spider.json", op_file="final_train_data.json", num_ques=100
# )

# Using this training dataset to perform the prompt finetuning
# training_dataset = load_json(file_path=os.path.join(DATA, "final_train_data.json"))

# with open(os.path.join(DATA, "final_train_data.json"), "r",encoding="utf-8") as f:
    
#     data = json.load(f)          
# questions={}
# for i in data:
#     questions[i.get("id")] = i.get("question")

# db_name="bike_1"
# db_handler=DatabaseHandler(db_name)
# db_schema=db_handler.get_db_schema_json()
# query="SELECT T1.lat ,  T1.long ,  T1.city FROM station AS T1 JOIN trip AS T2 ON T1.id  =  T2.start_station_id ORDER BY T2.duration LIMIT 1"
# print(db_handler.execute_command(query))


# gt_data=load_json(file_path=os.path.join(EVAL_OUTPUT,"results_gt.json"))
# ml_data=load_json(file_path=os.path.join(EVAL_OUTPUT,"results.json"))
# gt_errors=[]
# ml_errors=[]
# for i in gt_data:
#     if i.get("error") is not None:
#         gt_errors.append(i.get("id"))
# for i in ml_data:
#     if i.get("reply") is None:
#         ml_errors.append(i.get("id"))

# ml_missing=[]
# gt_missing=[]
# print(len(gt_errors),len(ml_errors))
# for i in gt_errors:
#     if i not in ml_errors:
#         gt_missing.append(i)

# for i in ml_errors:
#     if i not in gt_errors:
#         ml_missing.append(i)

# print(gt_missing)
# print(ml_missing)