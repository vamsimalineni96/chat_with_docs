import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA = os.path.join(PROJECT_ROOT, "spider_data")

TRAIN_ROOT = os.path.join(DATA, "database")
TEST_ROOT = os.path.join(DATA, "test_database")

# Place where the training and pruned training dataset jsonl files are present
TRAIN_DATASET = os.path.join(DATA, "train_dataset_nodb.jsonl")
PRUNE_TRAIN_DATASET = os.path.join(DATA, "train_dataset_nodb_pruned.jsonl")

# Place where the test dataset and the segregated test datasets are present
TEST_DATASET = os.path.join(DATA, "test_dataset_nodb_id.jsonl")
SEGREGATED_DATASET = "test_datasets"
