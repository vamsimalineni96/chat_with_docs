import json
import os
from chat_src.utils.config import EVAL_OUTPUT, SEGREGATED_DATASET, FINETUNE_OUTPUTS


import json
from typing import List, Dict
import sqlglot


def load_jsonl(path: str) -> Dict[int, Dict]:
    """Load a JSONL file into a dict keyed by id."""
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            data[item["id"]] = item
    return data


def compare_replies(reply1, reply2) -> bool:
    """Compare DB replies (execution results)."""
    return reply1 == reply2


def compare_sql_strings(sql1: str, sql2: str) -> bool:
    """Compare SQL strings directly (case + formatting sensitive)."""
    if sql1 is None or sql2 is None:
        return False
    return sql1.strip().lower() == sql2.strip().lower()


def compare_sql_structures(sql1: str, sql2: str) -> bool:
    try:
        return sqlglot.parse_one(sql1).dump() == sqlglot.parse_one(sql2).dump()
    except Exception:
        return False


def evaluate(ml_path: str, gt_path: str, comp_type: str, model: str):
    ml_data = load_jsonl(ml_path)
    gt_data = load_jsonl(gt_path)

    total = len(gt_data)
    ex_correct = sm_correct = struct_correct = 0

    for id_, gt in gt_data.items():
        if id_ not in ml_data:
            continue

        ml = ml_data[id_]

        if ml.get("generated_query") is None or gt.get("sql_query") is None:
            print(f"NoneType SQL query in id={id_}")
        # Execution Match
        if compare_replies(ml["reply"], gt["reply"]):
            ex_correct += 1

        # String Match
        if compare_sql_strings(ml["generated_query"], gt["sql_query"]):
            sm_correct += 1

        # Structural Match
        if compare_sql_structures(ml["generated_query"], gt["sql_query"]):
            struct_correct += 1

    print(f"Model: {model}")
    print(f"Complexity: {comp_type}")
    print(f"Total samples: {total}")
    print(f"Execution Accuracy (EX): {ex_correct / total:.2%}")
    print(f"String Match (SM): {sm_correct / total:.2%}")
    print(f"Structural Match: {struct_correct / total:.2%}")


if __name__ == "__main__":
    gmodel = "gemma"
    qmodel = "qwen"
    lmodel = "llama"

    shot = 0
    comp_type = "hard"

    lora_r = 8
    lora_alpha = 16
    lr = "3e-4"
    Model = f"qwen-{lora_r}-{lora_alpha}-{lr}"

    gt_file = os.path.join(SEGREGATED_DATASET, f"test_ground_truth_{comp_type}.jsonl")

    llama = os.path.join(
        EVAL_OUTPUT, f"llama_{shot}_shot_test_results_{comp_type}_pruned.jsonl"
    )
    gemma = os.path.join(
        EVAL_OUTPUT, f"gemma_{shot}_shot_test_results_{comp_type}.jsonl"
    )
    qwen = os.path.join(EVAL_OUTPUT, f"qwen_{shot}_shot_test_results_{comp_type}.jsonl")

    fgemma = os.path.join(FINETUNE_OUTPUTS, f"f{gmodel}_results_{comp_type}.jsonl")

    fqwen = os.path.join(
        FINETUNE_OUTPUTS,
        f"f{qmodel}_results_{comp_type}_{lora_r}_{lora_alpha}_{lr}.jsonl",
    )

    agentic = os.path.join(FINETUNE_OUTPUTS, "test_results_hard.jsonl")

    evaluate(ml_path=qwen, model=qmodel, comp_type=comp_type, gt_path=gt_file)
