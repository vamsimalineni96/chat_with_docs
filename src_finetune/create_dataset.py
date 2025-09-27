import json
import os
import random
from collections import defaultdict, Counter
from typing import List, Dict
from src.utils.config import DATA
from src.utils.sql_handler import DatabaseHandler


class DatasetCreator:
    def __init__(self, data_dir: str = DATA):
        """
        Handles creation of fine-tuning and validation datasets.
        :param data_dir: Base directory where data files are stored.
        """
        self.data_dir = data_dir

    # ---------- Utility Methods ---------- #
    @staticmethod
    def load_json(file_path: str) -> List[Dict]:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def save_to_json(data: List[Dict], file_path: str) -> None:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    @staticmethod
    def save_to_jsonl(data: List[Dict], file_path: str) -> None:
        with open(file_path, "w", encoding="utf-8") as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ---------- Dataset Creation ---------- #
    def create_dataset(self, inp_file: str, op_file: str, num_ques: int) -> None:
        """
        Create a fine-tuning dataset with DB schema and query-question pairs.
        Saves to JSON or JSONL depending on output extension.
        """
        file_path = os.path.join(self.data_dir, inp_file)
        print("Loading the complete training dataset")
        complete_train_data = self.load_json(file_path)
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

        # Count number of questions per db_id
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

        # Prepare JSONL-style data
        jsonl_data = []
        for element in final_data:
            prompt = f"Instruction: Generate an SQL query to answer the given question based on the database schema.\nDATABASE SCHEMA: {element.get('context')} Question: {element.get('input')}"

            jsonl_data.append({"input": prompt, "output": element.get("output")})

        name, ext = os.path.splitext(op_file)
        output_path = os.path.join(self.data_dir, op_file)
        if ext.lower() == ".json":
            print("Saving the dataset to a json file")
            self.save_to_json(data=final_data, file_path=output_path)
        elif ext.lower() == ".jsonl":
            print("Saving the dataset to a jsonl file")
            self.save_to_jsonl(data=jsonl_data, file_path=output_path)

    def create_validation_dataset(
        self,
        inp_file: str,
        op_file: str,
        num_ques: int,
        validation_ratio: float = 0.1,
        seed: int = 42,
    ) -> None:
        """
        Create a validation dataset sampled from databases having > num_ques records.
        Always saved as JSONL.
        """
        random.seed(seed)
        file_path = os.path.join(self.data_dir, inp_file)
        print("Loading the complete dataset")
        complete_data = self.load_json(file_path)

        # Group data by database id
        data_by_db = defaultdict(list)
        for c_instance in complete_data:
            db_id = c_instance.get("db_id")
            data_by_db[db_id].append(c_instance)

        validation_data = []
        count = 0

        for db_id, items in data_by_db.items():
            if len(items) > num_ques:
                sample_size = max(1, int(len(items) * validation_ratio))
                sampled_items = random.sample(items, sample_size)
                print(f"DB {db_id}: Sampling {sample_size}/{len(items)} for validation")

                for item in sampled_items:
                    db_handler = DatabaseHandler(db_id)
                    prompt = (
                        f"Instruction: Generate an SQL query to answer the given question based on the database schema.\nDATABASE SCHEMA: {json.loads(db_handler.get_db_schema_json())} "
                        f"Question: {item.get('question')}"
                    )
                    validation_data.append(
                        {"input": prompt, "output": item.get("query")}
                    )
                    count += 1

        print(f"Saving {count} validation samples to {op_file}")
        self.save_to_jsonl(
            data=validation_data, file_path=os.path.join(self.data_dir, op_file)
        )


# creator = DatasetCreator()

# creator.create_dataset(
#     inp_file="train_spider.json", op_file="training_dataset.jsonl", num_ques=1
# )

# creator.create_validation_dataset(
#     inp_file="train_spider.json",
#     op_file="validation_dataset.jsonl",
#     num_ques=1,
#     validation_ratio=0.1,
#     seed=42,
# )
