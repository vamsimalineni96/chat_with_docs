import os
from src.utils.config import SPIDER_ROOT, TEST_ROOT

db_name = "soccer_3"
file_name = f"{db_name}.sqlite"
db_path = os.path.join(TEST_ROOT, db_name, file_name)

print("SPIDER_ROOT:", SPIDER_ROOT)
print("TEST_ROOT:", TEST_ROOT)
print("Final db_path:", db_path)
print("Dir exists? ", os.path.isdir(os.path.dirname(db_path)))
print("File exists?", os.path.isfile(db_path))

path=r"C:\Work\bpd_github\tsql_schema_prune\spider_data\test_database\soccer_3\soccer_3.sqlite"
print(os.path.isfile(path))