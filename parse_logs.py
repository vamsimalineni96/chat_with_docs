import re
import json
from typing import Dict, Any

LOG_METRIC_TAG = "RAG_PIPELINE_METRICS"

# Example:
# 2025-12-03 14:31:31,303 | INFO | chat_service | RAG_PIPELINE_METRICS | conv_id=... | db_load_ms=16.01 | ...
LINE_REGEX = re.compile(
    r'^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\s+\|\s+.*?'
    + re.escape(LOG_METRIC_TAG)
    + r'\s+\|\s+(?P<kvpairs>.+)$'
)


def parse_metrics_from_log(log_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse RAG pipeline metrics from a log file.

    Returns:
        {
          "2025-12-03 14:31:31,303": {
              "conv_id": "c5a21ce9-...",
              "domain": "harry_potter",
              "db_load_ms": 16.01,
              "milvus_ms": 221.54,
              "llm_ms": 952.93,
              "db_save_ms": 25.03,
              "total_ms": 1243.54,
          },
          ...
        }
    """
    metrics_by_ts: Dict[str, Dict[str, Any]] = {}

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if LOG_METRIC_TAG not in line:
                continue  # fast skip

            m = LINE_REGEX.match(line)
            if not m:
                continue

            ts = m.group("ts")
            kv_str = m.group("kvpairs")

            # Each piece is like "conv_id=...", "domain=...", "db_load_ms=16.01"
            parts = [p.strip() for p in kv_str.split("|")]
            kv_dict: Dict[str, Any] = {}

            for part in parts:
                if "=" not in part:
                    continue
                key, value = part.split("=", 1)
                key = key.strip()
                value = value.strip()

                # Try to cast numeric metrics to float (e.g., *_ms or numeric strings)
                if key.endswith("_ms"):
                    try:
                        kv_dict[key] = float(value)
                    except ValueError:
                        kv_dict[key] = value  # fallback to raw string
                else:
                    # Generic heuristic: cast to float if it looks numeric
                    try:
                        num_val = float(value)
                        kv_dict[key] = num_val
                    except ValueError:
                        kv_dict[key] = value

            metrics_by_ts[ts] = kv_dict

    return metrics_by_ts


def dump_metrics_to_json(metrics: Dict[str, Dict[str, Any]], json_path: str) -> None:
    """
    Dump parsed metrics to a JSON file.
    """
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    log_path = r"src\logs\central_log.log"
    json_out_path = "rag_metrics.json"

    metrics = parse_metrics_from_log(log_path)
    dump_metrics_to_json(metrics, json_out_path)

    print(f"Wrote {len(metrics)} metric entries to {json_out_path}")
