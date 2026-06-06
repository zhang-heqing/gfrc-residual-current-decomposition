import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common import write_csv  # noqa: E402


METRICS = [
    "mse",
    "mae",
    "rmse",
    "ntae",
    "clea",
    "med_rce",
    "p90_rce",
    "acc_at_100",
    "crps",
    "picp_90",
    "mpiw_90",
    "alarm_precision",
    "alarm_recall",
    "alarm_f1",
    "alarm_accuracy",
    "alarm_far",
    "time_per_call_seconds",
]

MAXIMIZE_METRICS = {
    "clea",
    "acc_at_100",
    "picp_90",
    "alarm_precision",
    "alarm_recall",
    "alarm_f1",
    "alarm_accuracy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize repeated evaluation results.")
    parser.add_argument("--input-csv", default="experiments/output_repeat_eval5/per_repeat_metrics.csv")
    parser.add_argument("--output-csv", default="experiments/output_repeat_eval5/repeat_eval_summary.csv")
    return parser.parse_args()


def to_float(row: Dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in {"", None} else float("nan")


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(Path(args.input_csv).open()))
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)

    summary_rows: List[Dict[str, object]] = []
    for method, items in grouped.items():
        payload: Dict[str, object] = {
            "method": method,
            "repeats": len(items),
            "stochastic_eval": items[0].get("stochastic_eval", "False"),
        }
        for metric in METRICS:
            values = [to_float(row, metric) for row in items]
            values = [value for value in values if not math.isnan(value)]
            if not values:
                continue
            payload[f"{metric}_mean"] = sum(values) / len(values)
            payload[f"{metric}_std"] = 0.0 if len(values) == 1 else (sum((value - payload[f"{metric}_mean"]) ** 2 for value in values) / len(values)) ** 0.5
            payload[f"{metric}_best"] = max(values) if metric in MAXIMIZE_METRICS else min(values)
        summary_rows.append(payload)

    fieldnames = ["method", "repeats", "stochastic_eval"]
    for metric in METRICS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_best"])
    write_csv(Path(args.output_csv), summary_rows, fieldnames=fieldnames)


if __name__ == "__main__":
    main()
