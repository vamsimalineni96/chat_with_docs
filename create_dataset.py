#!/usr/bin/env python3
import json
import os
import random
from typing import List, Dict

from faker import Faker

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
        # Robust in case dtype is None or not a plain string
        if dtype is None:
            return "TEXT"
        if not isinstance(dtype, str):
            dtype = str(dtype)

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

    def schema_to_sql(self, schema: dict, rows_per_table: int = 1) -> str:
        """
        Convert schema dict to SQL DDL (and optional fake INSERTs).
        """

        sql_statements = []

        for table_name, info in schema.items():
            # Detect schema shape: extended vs simple
            if isinstance(info, dict) and any(
                key in info for key in ("columns", "primary_keys", "foreign_keys")
            ):
                # EXTENDED
                columns = info.get("columns", {}) or {}
                primary_keys = info.get("primary_keys", []) or []
                foreign_keys = info.get("foreign_keys", []) or []
            else:
                # SIMPLE (backward compatible)
                columns = info
                primary_keys = []
                foreign_keys = []

            # --- CREATE TABLE ---
            col_defs = []

            # Column definitions
            for col, dtype in columns.items():
                col_defs.append(f"{col} {self.normalize_dtype(dtype)}")

            # Primary key constraint
            if primary_keys:
                pk_cols = ", ".join(primary_keys)
                col_defs.append(f"PRIMARY KEY ({pk_cols})")

            # Foreign key constraints
            for fk in foreign_keys:
                from_col = fk.get("from_column")
                to_table = fk.get("to_table")
                to_col = fk.get("to_column")

                if not (from_col and to_table and to_col):
                    continue

                constraint = f"FOREIGN KEY ({from_col}) REFERENCES {to_table}({to_col})"

                on_update = fk.get("on_update")
                on_delete = fk.get("on_delete")

                # Avoid cluttering with default NO ACTION
                if on_update and on_update.upper() != "NO ACTION":
                    constraint += f" ON UPDATE {on_update}"
                if on_delete and on_delete.upper() != "NO ACTION":
                    constraint += f" ON DELETE {on_delete}"

                col_defs.append(constraint)

            cols_def_str = ", ".join(col_defs)
            create_stmt = f"CREATE TABLE {table_name} ({cols_def_str});"
            sql_statements.append(create_stmt)

            # --- INSERT INTO (optional fake rows) ---
            if self.fake and columns:
                col_names = ", ".join(columns.keys())
                values_list = []
                for _ in range(rows_per_table):
                    vals = [
                        self.generate_sample_value(dtype)
                        for dtype in columns.values()
                    ]
                    values_list.append(f"({', '.join(vals)})")
                insert_stmt = (
                    f"INSERT INTO {table_name} ({col_names}) "
                    f"VALUES {', '.join(values_list)};"
                )
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

            # Extended schema with PK + FK for richer prompts
            schema_json = db_handler.get_db_schema_with_relations_json()
            schema_dict = json.loads(schema_json)

            train_data.append(
                {
                    "id": i,
                    "db_id": c_instance.get("db_id"),
                    "query": c_instance.get("query"),
                    "question": c_instance.get("question"),
                    "db_schema": self.schema_to_sql(schema_dict),
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


# Example usage (you can remove or adapt this part as needed)
if __name__ == "__main__":
    creator = DatasetCreator(fake=False)
    creator.create_dataset(
        inp_file="test.json",
        op_file="test_dataset_db_id_test.jsonl",
        test=True,
    )
