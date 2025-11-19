import json
import os
from collections import Counter
from prune_src.utils.config import SEGREGATED_DATASET, DATA, TEST_DATASET
from prune_src.utils.sql_handler import DatabaseHandler

AGG_FUNCS = {"count", "sum", "avg", "min", "max"}
SET_OPS = {"union", "intersect", "except"}
LOGICAL_OPS = {"and", "or", "not"}


def normalize_sql(sql: str) -> str:
    return " ".join((sql or "").strip().split())


def heuristic_score(sql: str) -> int:
    """
    Simple, dialect-agnostic scoring:
      +2 per JOIN
      +2 per subquery signal
      +2 per set-op (UNION/INTERSECT/EXCEPT)
      +2 for window functions
      +2 for GROUP BY with aggregate(s)
      +2 for HAVING
      +1 for aggregate without GROUP BY
      +1 each: ORDER BY, DISTINCT, LIMIT, OFFSET
      +1 for boolean complexity (>=2 logical ops in WHERE)
      +1 for CASE WHEN usage
      +1 for wide projection (>=3 cols before FROM)
      +table-count bonus: max(0, tables-1) estimated via FROM + JOIN occurrences
    """
    s = normalize_sql(sql).lower()
    score = 0

    # Joins
    score += 2 * s.count(" join ")

    # Subqueries (heuristic)
    if "select" in s and (" in (" in s or " exists (" in s or " from (" in s):
        score += 2
    score += 2 * s.count("(select ")

    # Set ops
    for op in SET_OPS:
        if f" {op} " in s:
            score += 2

    # Window
    if " over (" in s:
        score += 2

    # Group/Aggregate
    has_group = " group by " in s
    has_agg = any(f"{fn}(" in s for fn in AGG_FUNCS)
    if has_group and has_agg:
        score += 2
    elif has_agg:
        score += 1

    # Having
    if " having " in s:
        score += 2

    # Order/Distinct/Limit/Offset
    if " order by " in s:
        score += 1
    if " distinct " in s:
        score += 1
    if " limit " in s:
        score += 1
    if " offset " in s:
        score += 1

    # Boolean complexity (rough)
    bool_ops = sum(s.count(op) for op in LOGICAL_OPS)
    if bool_ops >= 2:
        score += 1

    # CASE WHEN
    if " case " in s and " when " in s:
        score += 1

    # Projection width (rough): count commas in SELECT before FROM
    if "select" in s and " from " in s:
        seg = s.split(" from ", 1)[0]
        cols = seg.count(",") + (0 if "*" in seg else 1)
        if cols >= 3:
            score += 1

    # Table count bonus – approximate via FROM + JOINs
    table_refs = s.count(" from ") + s.count(" join ")
    T = max(1, table_refs)
    score += max(0, T - 1)

    return score


def bucket(score: int) -> str:
    if score <= 2:
        return "easy"
    if score <= 6:
        return "medium"
    return "hard"


def split_file(in_path: str, outdir: str) -> dict:
    os.makedirs(outdir, exist_ok=True)
    paths = {
        "easy": os.path.join(outdir, "easy.jsonl"),
        "medium": os.path.join(outdir, "medium.jsonl"),
        "hard": os.path.join(outdir, "hard.jsonl"),
    }
    writers = {k: open(p, "w", encoding="utf-8") for k, p in paths.items()}

    counts = Counter()
    total = 0

    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            item = json.loads(line)
            sql = item.get("query", "") or ""
            s = heuristic_score(sql)
            diff = bucket(s)
            item["difficulty_score"] = s
            item["difficulty"] = diff
            writers[diff].write(json.dumps(item, ensure_ascii=False) + "\n")
            counts[diff] += 1

    for w in writers.values():
        w.close()

    return {"total": total, "counts": dict(counts), "paths": paths}


def generate_dataset():
    in_path = os.path.join(DATA, TEST_DATASET)
    outdir = SEGREGATED_DATASET

    stats = split_file(in_path, outdir)

    print("\n=== Split complete ===")
    print(f"Total items: {stats['total']}")
    for b in ["easy", "medium", "hard"]:
        print(f"{b:6}: {stats['counts'].get(b,0)} -> {stats['paths'][b]}")
    print()


def generate_ground_truths():
    comp_types = ["easy", "medium", "hard"]

    for comp_type in comp_types:
        print(f"Generating Ground truth for : {comp_type} ")

        path = os.path.join(SEGREGATED_DATASET, f"{comp_type}.jsonl")
        data = []

        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:  # skip empty lines
                    continue
                try:
                    obj = json.loads(line)
                    data.append(obj)
                except json.JSONDecodeError as e:
                    print(f"Error in line {i}: {e}")

        op_file = os.path.join(
            SEGREGATED_DATASET, f"test_ground_truth_{comp_type}.jsonl"
        )
        with open(op_file, "w", encoding="utf-8") as outfile:
            for item in data:
                try:
                    gt_generator = DatabaseHandler(db_name=item.get("db_id"), test=True)
                    gt_generator.execute_command
                    result = {
                        "id": item.get("id"),
                        "question": item.get("question"),
                        "reply": gt_generator.execute_command(query=item.get("query")),
                        "sql_query": item.get("query"),
                    }
                except Exception as e:
                    result = {"question": item.get("question"), "error": str(e)}

                # Write each result immediately as JSONL
                outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
                outfile.flush()


def run():
    generate_dataset()
    generate_ground_truths()
    