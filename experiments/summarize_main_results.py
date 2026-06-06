import argparse
from pathlib import Path
from typing import Dict, List
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common import infer_param_count_from_checkpoint, read_json, write_csv


METHOD_META: Dict[str, Dict[str, str]] = {
    "GFRC": {"type": "Flow-based", "subdir": ""},
    "CNN-BiLSTM": {"type": "Deterministic", "subdir": "cnn_bilstm"},
    "Prophet": {"type": "Deterministic", "subdir": "prophet"},
    "TFT": {"type": "Deterministic", "subdir": "tft"},
    "TimeXer": {"type": "Deterministic", "subdir": "timexer"},
    "Deterministic-Physics": {"type": "Deterministic", "subdir": "deterministic_physics"},
    "CVAE": {"type": "Probabilistic", "subdir": "cvae"},
    "Diffusion": {"type": "Diffusion", "subdir": "diffusion"},
    "cINN": {"type": "Invertible", "subdir": "cinn"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize main comparison results into a CSV table.")
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--gfrc-run-dir", default="training_runs/shanse001_rebuilt_entity_split")
    parser.add_argument("--baseline-root", default="training_runs/baselines")
    parser.add_argument("--output-csv", default="experiments/output/main_comparison.csv")
    return parser.parse_args()


def summarize_run(run_dir: Path, method: str, method_type: str) -> Dict[str, object]:
    metrics = read_json(run_dir / "evaluation_metrics.json")
    run_config = read_json(run_dir / "run_config.json")
    training_summary_path = run_dir / "training_summary.json"
    training_summary = read_json(training_summary_path) if training_summary_path.exists() else {}

    if (run_dir / "best_gfrc_model.pt").exists():
        params = infer_param_count_from_checkpoint(run_dir / "best_gfrc_model.pt", "model_state_dict")
    else:
        params = infer_param_count_from_checkpoint(run_dir / "best_model.pt", "model_state_dict")

    return {
        "method": method,
        "type": method_type,
        "mae": metrics.get("mae"),
        "rmse": metrics.get("rmse"),
        "ntae": metrics.get("ntae"),
        "clea": metrics.get("clea"),
        "med_rce": metrics.get("med_rce"),
        "p90_rce": metrics.get("p90_rce"),
        "acc_at_100": metrics.get("acc_at_100"),
        "crps": metrics.get("crps"),
        "picp_90": metrics.get("picp_90"),
        "mpiw_90": metrics.get("mpiw_90"),
        "time_per_call_seconds": metrics.get("time_per_call_seconds"),
        "avg_epoch_seconds": training_summary.get("avg_epoch_seconds"),
        "params": params,
        "epochs": run_config.get("epochs"),
        "run_dir": str(run_dir),
    }


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()
    baseline_root = root_dir / args.baseline_root
    gfrc_run_dir = root_dir / args.gfrc_run_dir
    rows: List[Dict[str, object]] = []
    for method, meta in METHOD_META.items():
        run_dir = gfrc_run_dir if method == "GFRC" else baseline_root / meta["subdir"]
        if not run_dir.exists():
            continue
        rows.append(summarize_run(run_dir, method=method, method_type=meta["type"]))

    write_csv(
        Path(args.output_csv).resolve(),
        rows,
        fieldnames=[
            "method",
            "type",
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
            "avg_epoch_seconds",
            "params",
            "epochs",
            "run_dir",
        ],
    )


if __name__ == "__main__":
    main()
