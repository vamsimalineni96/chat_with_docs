import re, json
from src.utils.config import SCHEMA_DOCS
from src.utils.vectorstore import get_vectorstore_handler
from src.utils.logger_config import LoggerConfig

# Initialize the logger
logger_config = LoggerConfig()
logger = logger_config.logger


vector_store = get_vectorstore_handler()


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


def add_to_db():
    for obj in read_jsonl(path=SCHEMA_DOCS):
        db_name = list(obj.keys())[0]
        tables = obj.get(db_name).get("tables")

        for table in tables:
            m = re.match(r"^\s*([A-Za-z0-9_]+)\s*\.",table)
            table_name = m.group(1) if m else None
            vector_store.add_summary(db_name=db_name, text=table, table_name=table_name)
            logger.info(f"Added schema for db: {db_name}")
