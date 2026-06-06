import argparse
import csv
from pathlib import Path
from typing import Dict, List
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common import write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine per-method robustness CSV files into one summary CSV.")
    parser.add_argument("--input-dir", default="experiments/output/robustness")
    parser.add_argument("--output-csv", default="experiments/output/robustness_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    rows: List[Dict[str, object]] = []
    for path in sorted(input_dir.glob("*.csv")):
        with path.open() as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(dict(row))

    write_csv(
        Path(args.output_csv).resolve(),
        rows,
        fieldnames=[
            "method",
            "corruption",
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
            "time_per_call_seconds",
        ],
    )


if __name__ == "__main__":
    main()
