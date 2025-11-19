import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

SPIDER_ROOT = os.path.join(PROJECT_ROOT, "spider_data", "database")
TEST_ROOT = os.path.join(PROJECT_ROOT, "spider_data", "test_database")

DATA = os.path.join(PROJECT_ROOT, "spider_data")
DB_DOCS = os.path.join(DATA, "db_docs")
SCHEMA_DOCS = os.path.join(DB_DOCS, "tables_with_types_and_keys_by_db.jsonl")

TRAIN_DATASET = os.path.join(DATA, "train_dataset_nodb_id.jsonl")
TEST_DATASET = os.path.join(DATA, "test_dataset_db_id_test.jsonl")

SEGREGATED_DATASET = "test_datasets"

