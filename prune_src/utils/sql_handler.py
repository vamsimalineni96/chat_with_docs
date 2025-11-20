import os
import json
import sqlite3
from typing import Any, Dict, List, Optional

from prune_src.utils.logger_config import LoggerConfig
from prune_src.utils.config import TRAIN_ROOT, TEST_ROOT

# Initialize logger
logger_config = LoggerConfig()
logger = logger_config.logger


class SqlHandler:
    def __init__(self, db_path: str = "sakila_master.db"):
        self.db_path = db_path

    def execute_command(self, query: str, fetch: bool = True) -> Optional[List[Any]]:
        logger.info(f"Executing query: {query}")
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                if fetch:
                    results = cursor.fetchall()
                    logger.info(f"Query returned {len(results)} rows.")
                    return results
        except sqlite3.Error as e:
            logger.error(f"SQLite error: {e}")
            return None

    def get_db_schema(self) -> Dict[str, List[str]]:
        schema_info: Dict[str, List[str]] = {}
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                for table in tables:
                    table_name = table[0]
                    cursor.execute(f"PRAGMA table_info({table_name});")
                    columns = [col[1] for col in cursor.fetchall()]
                    schema_info[table_name] = columns
            logger.info(f"Retrieved schema for {len(schema_info)} tables.")
        except sqlite3.Error as e:
            logger.error(f"SQLite error while fetching schema: {e}")
        return schema_info

    def get_db_schema_json(self) -> str:
        """Return schema as a JSON-formatted string."""
        return json.dumps(self.get_db_schema(), indent=2)


class DatabaseHandler_old:
    def __init__(self, db_name: str = None, test: bool = False):
        self.db_name = db_name
        self.file_name = f"{self.db_name}.sqlite"
        if test:
            self.db_path = os.path.join(TEST_ROOT, self.db_name, self.file_name)
        else:
            self.db_path = os.path.join(TRAIN_ROOT, self.db_name, self.file_name)

        logger.info(f"Accessing: {self.db_path} ")

    def get_db_schema(self) -> Dict[str, Dict[str, str]]:
        schema_info: Dict[str, Dict[str, str]] = {}
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                for table in tables:
                    table_name = table[0]
                    cursor.execute(f"PRAGMA table_info({table_name});")
                    columns = cursor.fetchall()
                    schema_info[table_name] = {
                        col[1]: col[2]
                        for col in columns  # col[1]=column name, col[2]=type
                    }
            logger.info(f"Retrieved schema (with types) for {len(schema_info)} tables.")
        except sqlite3.Error as e:
            logger.error(f"SQLite error while fetching schema: {e}")
        return schema_info

    def get_db_schema_json(self) -> str:
        """Return schema as a JSON-formatted string."""
        return json.dumps(self.get_db_schema(), indent=2)

    def execute_command(self, query: str, fetch: bool = True) -> Optional[List[Any]]:
        logger.info(f"Executing query: {query}")
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                if fetch:
                    results = cursor.fetchall()
                    logger.info(f"Query returned {len(results)} rows.")
                    return results
        except sqlite3.Error as e:
            logger.error(f"SQLite error: {e}")
            return None


class DatabaseHandler:
    def __init__(self, db_name: str = None, test: bool = False):
        self.db_name = db_name
        self.file_name = f"{self.db_name}.sqlite"
        if test:
            self.db_path = os.path.join(TEST_ROOT, self.db_name, self.file_name)
        else:
            self.db_path = os.path.join(TRAIN_ROOT, self.db_name, self.file_name)

        logger.info(f"Accessing: {self.db_path} ")

    def get_db_schema(self) -> Dict[str, Dict[str, str]]:
        schema_info: Dict[str, Dict[str, str]] = {}

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [row[0] for row in cursor.fetchall()]

                for table_name in tables:
                    cursor.execute(f"PRAGMA table_info({table_name});")
                    columns = cursor.fetchall()
                    # columns rows: (cid, name, type, notnull, dflt_value, pk)
                    schema_info[table_name] = {
                        col[1]: col[2]
                        for col in columns  # col[1]=column name, col[2]=type
                    }

            logger.info(f"Retrieved schema (with types) for {len(schema_info)} tables.")
        except sqlite3.Error as e:
            logger.error(f"SQLite error while fetching schema: {e}")

        return schema_info

    def get_db_schema_with_relations(self) -> Dict[str, Dict[str, Any]]:
        schema_info: Dict[str, Dict[str, Any]] = {}

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [row[0] for row in cursor.fetchall()]

                for table_name in tables:
                    # Columns + PK
                    cursor.execute(f"PRAGMA table_info({table_name});")
                    columns = cursor.fetchall()
                    columns_dict = {col[1]: col[2] for col in columns}
                    primary_keys = [col[1] for col in columns if col[5] > 0]

                    # FKs
                    cursor.execute(f"PRAGMA foreign_key_list({table_name});")
                    fk_rows = cursor.fetchall()
                    foreign_keys: List[Dict[str, Any]] = []
                    for row in fk_rows:
                        # row: (id, seq, table, from, to, on_update, on_delete, match)
                        foreign_keys.append(
                            {
                                "from_column": row[3],
                                "to_table": row[2],
                                "to_column": row[4],
                                "on_update": row[5],
                                "on_delete": row[6],
                                "match": row[7],
                            }
                        )

                    schema_info[table_name] = {
                        "columns": columns_dict,
                        "primary_keys": primary_keys,
                        "foreign_keys": foreign_keys,
                    }

            logger.info(
                f"Retrieved schema (types + PK + FK) for {len(schema_info)} tables."
            )

        except sqlite3.Error as e:
            logger.error(f"SQLite error while fetching extended schema: {e}")

        return schema_info

    def get_db_schema_json(self) -> str:
        """Return the SIMPLE schema (no relations) as a JSON-formatted string."""
        return json.dumps(self.get_db_schema(), indent=2)

    def get_db_schema_with_relations_json(self) -> str:
        """Return the extended schema (with PK + FK) as a JSON-formatted string."""
        return json.dumps(self.get_db_schema_with_relations(), indent=2)

    def execute_command(self, query: str, fetch: bool = True) -> Optional[List[Any]]:
        logger.info(f"Executing query: {query}")
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                if fetch:
                    results = cursor.fetchall()
                    logger.info(f"Query returned {len(results)} rows.")
                    return results
        except sqlite3.Error as e:
            logger.error(f"SQLite error: {e}")
            return None
