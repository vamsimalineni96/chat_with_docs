"""Evaluation harness for the agentic RAG pipeline.

Reports on a JSONL dataset of `{id, question, ideal_answer, expected_substrings,
history, expected_to_refuse}` test cases:

    - Accuracy by LLM-judge grade (CORRECT / PARTIAL / WRONG / REFUSED).
    - Retrieval recall (fraction of expected_substrings found in any retrieved
      passage from the agent's tool calls).
    - Verifier catch rate (% of WRONG/PARTIAL answers the verifier flagged) and
      precision (% of flagged answers that were actually wrong).
    - Latency p50 / p95.
    - Average agent iterations + verifier corrections per question.
    - Breakdown by tag.

Usage (from project root):
    python scripts/eval.py
    python scripts/eval.py --collection docs --output eval_results.json
    python scripts/eval.py --limit 5                # smoke-test on first 5 cases
    python scripts/eval.py --tags factual followup  # filter to specific categories
"""

import argparse
import json
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Make `src` importable regardless of how this script is invoked
# (`python scripts/eval.py` puts the script's dir at sys.path[0], not the
# project root, which would break `from src.utils import …`).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from src.utils import config  # noqa: E402
from src.utils.rag_pipeline import answer_question  # noqa: E402
from src.utils.services.embedder import EmbeddingHandler  # noqa: E402
from src.utils.services.inference import NIMClient  # noqa: E402


JUDGE_SYSTEM_PROMPT = """You are an impartial grader for a question-answering system.

You will see:
1. A question.
2. An ideal answer (the reference).
3. The system's actual answer.
4. Whether the question was expected to be refused (the document does not
   contain the answer).

Grade the system's answer using EXACTLY one of these labels:

- CORRECT  : The actual answer covers the same key facts as the ideal answer.
             Phrasing may differ; minor missing details are okay.
             Or: the question was expected to be refused AND the system
             refused (said it can't find the info, declined to guess).
- PARTIAL  : Some of the key facts are right but others are missing or wrong.
             Or: the system partially answered when refusal was expected.
- WRONG    : The actual answer contradicts the ideal, OR fabricates
             specifics that are not in the ideal answer.
             Or: refusal was expected but the system invented an answer.
- REFUSED  : The system declined to answer when an answer WAS expected
             (the ideal answer contains real facts but the system said
             it couldn't find them).

Respond in EXACTLY this format, no preamble, no markdown:

GRADE: <one of CORRECT|PARTIAL|WRONG|REFUSED>
REASON: <one short sentence>
"""


def llm_judge(judge_llm, question: str, ideal: str, actual: str, expected_to_refuse: bool) -> Dict[str, str]:
    """Grade `actual` against `ideal`. Returns {'grade': ..., 'reason': ...}."""
    payload = (
        f"QUESTION:\n{question}\n\n"
        f"IDEAL ANSWER:\n{ideal}\n\n"
        f"SYSTEM'S ACTUAL ANSWER:\n{actual}\n\n"
        f"EXPECTED TO REFUSE: {'yes' if expected_to_refuse else 'no'}"
    )
    try:
        resp = judge_llm.invoke([
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=payload),
        ])
        text = (resp.content or "").strip()
    except Exception as e:
        return {"grade": "ERROR", "reason": f"judge call failed: {e}"}

    grade = "ERROR"
    reason = ""
    for line in text.splitlines():
        upper = line.upper()
        if upper.startswith("GRADE:"):
            value = line.split(":", 1)[1].strip().upper()
            if value in ("CORRECT", "PARTIAL", "WRONG", "REFUSED"):
                grade = value
        elif upper.startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
    return {"grade": grade, "reason": reason}


def retrieval_recall(retrieved_passages: List[str], expected_substrings: List[str]) -> float:
    """Fraction of expected_substrings found in any retrieved passage.

    Case-insensitive substring search. If no expected substrings are declared
    (e.g., for off-topic questions), recall is reported as 1.0 — there's
    nothing to verify.
    """
    if not expected_substrings:
        return 1.0
    haystack = " ".join(retrieved_passages).lower()
    hits = sum(1 for needle in expected_substrings if needle.lower() in haystack)
    return hits / len(expected_substrings)


def evaluate_one(
    test_case: Dict[str, Any],
    embedder: EmbeddingHandler,
    judge_llm,
    collection_name: str,
) -> Dict[str, Any]:
    question = test_case["question"]
    history = test_case.get("history", [])
    ideal = test_case["ideal_answer"]
    expected_to_refuse = bool(test_case.get("expected_to_refuse", False))

    q_embed = embedder.get_embedding(question, input_type="query")

    t0 = time.perf_counter()
    try:
        result = answer_question(
            question=question,
            query_vec=q_embed,
            collection_name=collection_name,
            history=history,
            debug=True,  # always debug so we can read intermediate state
        )
        error = None
    except Exception as e:
        result = {"answer": "", "debug": {}}
        error = str(e)
    elapsed = time.perf_counter() - t0

    answer = result.get("answer", "")
    debug = result.get("debug") or {}

    # Pull tool result texts (the retrieved passages the agent actually saw).
    tool_passages = [
        m.get("content", "") for m in debug.get("final_messages", [])
        if m.get("type") == "tool"
    ]
    recall = retrieval_recall(tool_passages, test_case.get("expected_substrings", []))

    # Tool calls actually made by the agent (across all iterations).
    tool_call_count = sum(
        len(m.get("tool_calls", []) or [])
        for m in debug.get("final_messages", [])
        if m.get("type") in (None, "ai", "AIMessage")
    )

    judge = llm_judge(judge_llm, question, ideal, answer, expected_to_refuse) if not error else {
        "grade": "ERROR", "reason": "pipeline error"
    }

    return {
        "id": test_case["id"],
        "tags": test_case.get("tags", []),
        "question": question,
        "ideal_answer": ideal,
        "actual_answer": answer,
        "expected_to_refuse": expected_to_refuse,
        "grade": judge["grade"],
        "judge_reason": judge["reason"],
        "recall": recall,
        "verifier_verdict": debug.get("verifier_verdict"),
        "corrections_used": debug.get("corrections_used", 0),
        "agent_iterations": debug.get("total_iterations", 0),
        "tool_call_count": tool_call_count,
        "rewritten_query": debug.get("rewritten_query"),
        "latency_s": elapsed,
        "error": error,
    }


def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    successful = [r for r in results if r.get("grade") != "ERROR" and r.get("error") is None]
    n = len(successful)
    if n == 0:
        return {"n": 0, "note": "no successful runs"}

    grade_counts = Counter(r["grade"] for r in successful)
    accuracy = grade_counts.get("CORRECT", 0) / n

    latencies = sorted(r["latency_s"] for r in successful)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[max(0, int(len(latencies) * 0.95) - 1)]

    # Verifier metrics — only meaningful for runs where verifier actually fired
    # (which it does on every successful answer in this pipeline).
    wrong_set = [r for r in successful if r["grade"] in ("WRONG", "PARTIAL")]
    flagged_set = [r for r in successful if r["verifier_verdict"] in ("retry", "exhausted")]

    catch_rate = (
        sum(1 for r in wrong_set if r["verifier_verdict"] in ("retry", "exhausted")) / len(wrong_set)
        if wrong_set else None
    )
    precision = (
        sum(1 for r in flagged_set if r["grade"] in ("WRONG", "PARTIAL")) / len(flagged_set)
        if flagged_set else None
    )

    # Per-tag breakdown
    by_tag = defaultdict(lambda: {"n": 0, "correct": 0, "recall_sum": 0.0})
    for r in successful:
        for tag in r["tags"]:
            by_tag[tag]["n"] += 1
            by_tag[tag]["correct"] += int(r["grade"] == "CORRECT")
            by_tag[tag]["recall_sum"] += r["recall"]
    tag_breakdown = {
        tag: {
            "n": v["n"],
            "accuracy": v["correct"] / v["n"],
            "avg_recall": v["recall_sum"] / v["n"],
        }
        for tag, v in by_tag.items()
    }

    return {
        "n": n,
        "n_errors": len(results) - n,
        "accuracy": round(accuracy, 3),
        "grade_breakdown": dict(grade_counts),
        "avg_recall": round(statistics.mean(r["recall"] for r in successful), 3),
        "p50_latency_s": round(p50, 2),
        "p95_latency_s": round(p95, 2),
        "avg_agent_iterations": round(statistics.mean(r["agent_iterations"] for r in successful), 2),
        "avg_corrections": round(statistics.mean(r["corrections_used"] for r in successful), 2),
        "avg_tool_calls": round(statistics.mean(r["tool_call_count"] for r in successful), 2),
        "verifier_catch_rate": (
            None if catch_rate is None else round(catch_rate, 3)
        ),
        "verifier_precision": (
            None if precision is None else round(precision, 3)
        ),
        "by_tag": tag_breakdown,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(_PROJECT_ROOT / "eval_dataset.jsonl"))
    parser.add_argument("--collection", default=config.COLLECTION_NAME)
    parser.add_argument("--output", default=None,
                        help="Optional path to dump full results as JSON.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N cases.")
    parser.add_argument("--tags", nargs="+", default=None,
                        help="Only run cases tagged with any of these.")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    test_cases = [json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip()]
    if args.tags:
        wanted = set(args.tags)
        test_cases = [tc for tc in test_cases if wanted & set(tc.get("tags", []))]
    if args.limit:
        test_cases = test_cases[: args.limit]

    if not test_cases:
        print("No test cases after filtering.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(test_cases)} test cases from {dataset_path}")
    print(f"Collection: {args.collection}")
    print()

    embedder = EmbeddingHandler()
    judge_llm = NIMClient().llm  # raw ChatNVIDIA (no tools bound) for grading

    results = []
    start = time.perf_counter()
    for i, tc in enumerate(test_cases, 1):
        q_short = re.sub(r"\s+", " ", tc["question"])[:60]
        print(f"[{i:2d}/{len(test_cases)}] {tc['id']:6s} {q_short!r:<65s}", end="", flush=True)
        r = evaluate_one(tc, embedder, judge_llm, args.collection)
        results.append(r)
        if r["error"]:
            print(f" ✗ {r['error'][:50]}")
        else:
            print(f" {r['grade']:8s} recall={r['recall']:.2f} verdict={str(r['verifier_verdict'])[:9]:9s} {r['latency_s']:5.1f}s")

    total = time.perf_counter() - start
    print(f"\nDone in {total:.1f}s ({total/len(test_cases):.1f}s avg per case)\n")

    summary = aggregate(results)
    print("─── Summary ───")
    print(json.dumps(summary, indent=2))

    if args.output:
        out = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "dataset": str(dataset_path),
            "collection": args.collection,
            "n_cases": len(test_cases),
            "summary": summary,
            "results": results,
        }
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
