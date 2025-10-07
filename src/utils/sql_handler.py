import os
import json
import sqlite3
from typing import Any, Dict, List, Optional

from src.utils.logger_config import LoggerConfig
from src.utils.config import SPIDER_ROOT, TEST_ROOT

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


class DatabaseHandler:
    def __init__(self, db_name: str = None, test:bool=False):
        self.db_name = db_name
        self.file_name = f"{self.db_name}.sqlite"
        if test:
            self.db_path = os.path.join(TEST_ROOT, self.db_name, self.file_name)
        else:
            self.db_path = os.path.join(SPIDER_ROOT, self.db_name, self.file_name)


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
                        col[1]: col[2] for col in columns  # col[1]=column name, col[2]=type
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
