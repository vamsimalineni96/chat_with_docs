import json
from typing import Dict, Any, List

import pandas as pd
import matplotlib.pyplot as plt


METRIC_KEYS: List[str] = [
    "db_load_ms",
    "milvus_ms",
    "llm_ms",
    "db_save_ms",
    "total_ms",
]


def load_metrics_json(json_path: str) -> pd.DataFrame:
    """
    Load metrics JSON (from previous script) into a pandas DataFrame.

    JSON format:
    {
      "2025-12-03 14:31:31,303": {
          "conv_id": "...",
          "db_load_ms": 16.01,
          "milvus_ms": 221.54,
          "llm_ms": 952.93,
          "db_save_ms": 25.03,
          "total_ms": 1243.54,
          ...
      },
      ...
    }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data: Dict[str, Dict[str, Any]] = json.load(f)

    # Convert to DataFrame (index = timestamp)
    df = pd.DataFrame.from_dict(data, orient="index")

    # Normalize / clean timestamp index
    df.index.name = "ts"
    df.reset_index(inplace=True)

    # Convert ts to datetime
    # strip the comma in milliseconds (",303" -> ".303")
    df["ts"] = (
        df["ts"]
        .str.replace(",", ".", regex=False)
        .pipe(pd.to_datetime, format="%Y-%m-%d %H:%M:%S.%f")
    )

    # Sort by time just in case
    df = df.sort_values("ts").reset_index(drop=True)

    return df


def plot_metrics_time_series(df: pd.DataFrame, save_path: str = None) -> None:
    """
    Plot all metric stages on a single line chart vs time.
    """
    plt.figure(figsize=(12, 6))

    for key in METRIC_KEYS:
        if key in df.columns:
            plt.plot(df["ts"], df[key], marker="o", label=key)

    plt.xlabel("Timestamp")
    plt.ylabel("Latency (ms)")
    plt.title("RAG Pipeline Stage Latencies Over Time")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[INFO] Saved time-series plot to {save_path}")
    else:
        plt.show()


def print_basic_stats(df: pd.DataFrame) -> None:
    """
    Print some basic stats to see bottlenecks.
    """
    print("\n=== Basic Latency Stats (ms) ===")
    print(df[METRIC_KEYS].describe())

    # Average contribution to total_ms (where total_ms exists and >0)
    if "total_ms" in df.columns:
        valid = df[df["total_ms"] > 0]
        if not valid.empty:
            contrib = {}
            for key in METRIC_KEYS:
                if key == "total_ms" or key not in valid.columns:
                    continue
                contrib[key] = (valid[key] / valid["total_ms"]).mean() * 100.0

            print("\n=== Average Stage Contribution to total_ms (%) ===")
            for k, v in contrib.items():
                print(f"{k:12s}: {v:6.2f} %")


def plot_average_breakdown(df: pd.DataFrame, save_path: str = None) -> None:
    """
    Plot a bar chart of average latency per stage (excluding total_ms),
    so you can visually see the bottleneck.
    """
    means = {}
    for key in METRIC_KEYS:
        if key == "total_ms":
            continue
        if key in df.columns:
            means[key] = df[key].mean()

    if not means:
        print("[WARN] No metrics found for breakdown plot")
        return

    plt.figure(figsize=(8, 5))
    stages = list(means.keys())
    values = [means[k] for k in stages]

    plt.bar(stages, values)
    plt.ylabel("Average Latency (ms)")
    plt.title("Average Latency per Stage")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[INFO] Saved breakdown plot to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    json_path = "st_rag_metrics.json"  # output from previous script

    df = load_metrics_json(json_path)    
    # 2) Print some quick stats
    print_basic_stats(df)


    # 1) Show/Save time series plot with all metrics
    # plot_metrics_time_series(df, save_path="mt_rag_metrics_timeseries.png")


    # 3) Show/Save average breakdown to visually see bottleneck
    # plot_average_breakdown(df, save_path="mt_rag_metrics_breakdown.png")
