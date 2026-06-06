import argparse
import json
from pathlib import Path
from typing import Dict, List
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common import write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize physical consistency JSON outputs into one CSV.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    rows: List[Dict[str, object]] = []

    for json_path in sorted(input_dir.glob("*.json")):
        payload = json.loads(json_path.read_text())
        pred = payload["predicted_consistency_mae"]
        oracle = payload["oracle_consistency_mae"]
        rows.append(
            {
                "method": payload["method"],
                "pred_mae_mean": pred["mean"],
                "pred_mae_median": pred["median"],
                "pred_mae_p90": pred["p90"],
                "pred_mae_p95": pred["p95"],
                "oracle_mae_mean": oracle["mean"],
                "oracle_mae_median": oracle["median"],
                "oracle_mae_p90": oracle["p90"],
                "oracle_mae_p95": oracle["p95"],
                "pred_normalized_gap_mean": payload["predicted_normalized_gap_mean"],
                "oracle_normalized_gap_mean": payload["oracle_normalized_gap_mean"],
                "num_windows": payload["num_windows"],
                "run_dir": payload["run_dir"],
            }
        )

    write_csv(
        Path(args.output_csv).resolve(),
        rows,
        fieldnames=[
            "method",
            "pred_mae_mean",
            "pred_mae_median",
            "pred_mae_p90",
            "pred_mae_p95",
            "oracle_mae_mean",
            "oracle_mae_median",
            "oracle_mae_p90",
            "oracle_mae_p95",
            "pred_normalized_gap_mean",
            "oracle_normalized_gap_mean",
            "num_windows",
            "run_dir",
        ],
    )


if __name__ == "__main__":
    main()
