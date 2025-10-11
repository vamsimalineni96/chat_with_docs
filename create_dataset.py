import json
import os
import random
from faker import Faker
from typing import List, Dict
from src.utils.config import DATA
from src.utils.sql_handler import DatabaseHandler
from src.utils.segregate_dataset import run as segregate_test_datasets

fake = Faker()


class DatasetCreator:
    def __init__(
        self, data_dir: str = DATA, fake: bool = False, segregate_test: bool = False
    ):
        self.data_dir = data_dir
        self.fake = fake
        self.segregate_test = segregate_test

    @staticmethod
    def load_json(file_path: str) -> List[Dict]:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def save_to_jsonl(data: List[Dict], file_path: str) -> None:
        with open(file_path, "w", encoding="utf-8") as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def normalize_dtype(self, dtype: str) -> str:
        dtype = dtype.strip().upper()
        if any(
            x in dtype
            for x in ["INT", "NUMBER", "SMALLINT", "BIGINT", "MEDIUMINT", "TINYINT"]
        ):
            return "INT"
        elif any(x in dtype for x in ["REAL", "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC"]):
            return "REAL"
        elif any(x in dtype for x in ["CHAR", "VARCHAR", "TEXT", "CLOB"]):
            return "TEXT"
        elif "DATE" in dtype and "TIME" not in dtype:
            return "DATE"
        elif "TIME" in dtype:
            return "DATETIME"
        elif any(x in dtype for x in ["BOOL", "BIT"]):
            return "BOOLEAN"
        elif "BLOB" in dtype:
            return "BLOB"
        else:
            return "TEXT"

    def generate_sample_value(self, dtype: str):
        dtype = self.normalize_dtype(dtype)
        if dtype == "INT":
            return str(random.randint(1, 1000))
        elif dtype == "REAL":
            return str(round(random.uniform(1.0, 5000.0), 2))
        elif dtype == "TEXT":
            return f"'{fake.word()}'"
        elif dtype == "DATE":
            return f"'{fake.date_between(start_date='-5y', end_date='today')}'"
        elif dtype == "DATETIME":
            return f"'{fake.date_time_this_decade().strftime('%Y-%m-%d %H:%M:%S')}'"
        elif dtype == "BOOLEAN":
            return random.choice(["TRUE", "FALSE"])
        elif dtype == "BLOB":
            return f"X'{fake.sha1()[:8]}'"
        else:
            return "NULL"

    # --- Convert schema to SQL (CREATE + INSERT) ---
    def schema_to_sql(self, schema: dict, rows_per_table: int = 1):
        sql_statements = []

        for table_name, columns in schema.items():
            # CREATE TABLE
            cols_def = ", ".join(
                [
                    f"{col} {self.normalize_dtype(dtype)}"
                    for col, dtype in columns.items()
                ]
            )
            create_stmt = f"CREATE TABLE {table_name} ({cols_def});"
            sql_statements.append(create_stmt)

            # INSERT INTO
            if self.fake:
                col_names = ", ".join(columns.keys())
                values_list = []
                for _ in range(rows_per_table):
                    vals = [
                        self.generate_sample_value(dtype) for dtype in columns.values()
                    ]
                    values_list.append(f"({', '.join(vals)})")
                insert_stmt = f"INSERT INTO {table_name} ({col_names}) VALUES {', '.join(values_list)};"
                sql_statements.append(insert_stmt)

        return " ".join(sql_statements)

    def create_dataset(self, inp_file: str, op_file: str, test: bool = False):
        file_path = os.path.join(self.data_dir, inp_file)
        print("Loading the complete training dataset")
        complete_data = self.load_json(file_path)

        train_data = []

        # Prepare dataset with SQL schema
        for i, c_instance in enumerate(complete_data):
            db_handler = DatabaseHandler(c_instance.get("db_id"), test=test)
            train_data.append(
                {
                    "id": i,
                    "db_id": c_instance.get("db_id"),
                    "query": c_instance.get("query"),
                    "question": c_instance.get("question"),
                    "db_schema": self.schema_to_sql(
                        json.loads(db_handler.get_db_schema_json())
                    ),
                }
            )

        # Save directly to final JSONL
        output_path = os.path.join(self.data_dir, op_file)
        print(f"Saving the final dataset to {output_path}")
        self.save_to_jsonl(data=train_data, file_path=output_path)
        print(f"Dataset saved to {output_path}")

        if self.segregate_test:
            print("Segregating the test dataset based on complexity")
            segregate_test_datasets()


# Example usage:
# creator=DatasetCreator(fake=True)
# creator.create_dataset(
#     inp_file="test.json",
#     op_file="test_dataset_db_id.jsonl",
#     test=True
# )
