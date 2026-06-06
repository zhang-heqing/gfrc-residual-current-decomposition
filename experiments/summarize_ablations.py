import argparse
from pathlib import Path
from typing import Dict, List
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common import infer_param_count_from_checkpoint, read_json, write_csv


ABLATION_META: Dict[str, str] = {
    "GFRC": "gfrc_full",
    "w/o cue": "gfrc_no_cue",
    "w/o physics": "gfrc_no_physics",
    "Deterministic backbone": "deterministic_backbone",
    "Deterministic + physics": "deterministic_physics",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize ablation results into a CSV table.")
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--ablation-root", default="training_runs/ablations")
    parser.add_argument("--output-csv", default="experiments/output/ablation_comparison.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()
    ablation_root = root_dir / args.ablation_root
    rows: List[Dict[str, object]] = []
    for variant, rel_run_dir in ABLATION_META.items():
        run_dir = ablation_root / rel_run_dir
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
                "variant": variant,
                "use_cue": run_config.get("use_cue", not run_config.get("disable_cue", False)),
                "lambda_physical": run_config.get("lambda_physical"),
                "rmse": metrics.get("rmse"),
                "ntae": metrics.get("ntae"),
                "clea": metrics.get("clea"),
                "med_rce": metrics.get("med_rce"),
                "p90_rce": metrics.get("p90_rce"),
                "acc_at_100": metrics.get("acc_at_100"),
                "crps": metrics.get("crps"),
                "avg_epoch_seconds": training_summary.get("avg_epoch_seconds"),
                "params": params,
                "run_dir": str(run_dir),
            }
        )

    write_csv(
        Path(args.output_csv).resolve(),
        rows,
        fieldnames=[
            "variant",
            "use_cue",
            "lambda_physical",
            "rmse",
            "ntae",
            "clea",
            "med_rce",
            "p90_rce",
            "acc_at_100",
            "crps",
            "avg_epoch_seconds",
            "params",
            "run_dir",
        ],
    )


if __name__ == "__main__":
    main()
