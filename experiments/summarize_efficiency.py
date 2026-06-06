import argparse
from pathlib import Path
from typing import Dict, List
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common import infer_param_count_from_checkpoint, read_json, write_csv


RUNS: Dict[str, str] = {
    "GFRC": "",
    "TimeXer": "timexer",
    "TFT": "tft",
    "Deterministic-Physics": "deterministic_physics",
    "CVAE": "cvae",
    "Diffusion": "diffusion",
    "cINN": "cinn",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize efficiency-related metrics into a CSV table.")
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--gfrc-run-dir", default="training_runs/shanse001_rebuilt_entity_split")
    parser.add_argument("--baseline-root", default="training_runs/baselines")
    parser.add_argument("--output-csv", default="experiments/output/efficiency_comparison.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()
    baseline_root = root_dir / args.baseline_root
    gfrc_run_dir = root_dir / args.gfrc_run_dir
    rows: List[Dict[str, object]] = []

    for method, rel_run_dir in RUNS.items():
        run_dir = gfrc_run_dir if method == "GFRC" else baseline_root / rel_run_dir
        if not run_dir.exists():
            continue
        metrics = read_json(run_dir / "evaluation_metrics.json")
        run_config = read_json(run_dir / "run_config.json")
        training_summary_path = run_dir / "training_summary.json"
        training_summary = read_json(training_summary_path) if training_summary_path.exists() else {}
        checkpoint_name = "best_gfrc_model.pt" if (run_dir / "best_gfrc_model.pt").exists() else "best_model.pt"
        params = infer_param_count_from_checkpoint(run_dir / checkpoint_name, "model_state_dict")
        rows.append(
            {
                "method": method,
                "params": params,
                "avg_epoch_seconds": training_summary.get("avg_epoch_seconds"),
                "time_per_call_seconds": metrics.get("time_per_call_seconds"),
                "rmse": metrics.get("rmse"),
                "ntae": metrics.get("ntae"),
                "crps": metrics.get("crps"),
                "batch_size": run_config.get("batch_size"),
                "epochs": run_config.get("epochs"),
                "run_dir": str(run_dir),
            }
        )

    write_csv(
        Path(args.output_csv).resolve(),
        rows,
        fieldnames=[
            "method",
            "params",
            "avg_epoch_seconds",
            "time_per_call_seconds",
            "rmse",
            "ntae",
            "crps",
            "batch_size",
            "epochs",
            "run_dir",
        ],
    )


if __name__ == "__main__":
    main()
