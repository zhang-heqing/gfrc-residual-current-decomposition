import argparse
import csv
from pathlib import Path
from typing import Dict, List
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common import ensure_parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CSV summaries into simple LaTeX table snippets.")
    parser.add_argument("--main-csv", default="experiments/output/main_comparison.csv")
    parser.add_argument("--ablation-csv", default="experiments/output/ablation_comparison.csv")
    parser.add_argument("--efficiency-csv", default="experiments/output/efficiency_comparison.csv")
    parser.add_argument("--robustness-csv", default="experiments/output/robustness_summary.csv")
    parser.add_argument("--physical-csv", default="experiments/output_revision/physical_consistency/summary.csv")
    parser.add_argument("--lambda-csv", default="training_runs/lambda_physical_sweep/lambda_physical_summary.csv")
    parser.add_argument("--output-dir", default="experiments/output/latex")
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def format_float(value: str | None, digits: int = 4) -> str:
    if value is None or value == "":
        return "--"
    try:
        return f"{float(value):.{digits}f}"
    except ValueError:
        return value


def display_method(value: str | None) -> str:
    if not value:
        return "--"
    mapping = {
        "gfrc": "GFRC",
        "timexer": "TimeXer",
        "deterministic_physics": "Deterministic-Physics",
        "cvae": "CVAE",
        "diffusion": "Diffusion",
        "cinn": "cINN",
    }
    return mapping.get(value, value)


def write_table(path: Path, caption: str, label: str, columns: List[str], rows: List[List[str]]) -> None:
    ensure_parent(path)
    align = "l" * len(columns)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join(columns) + " \\\\",
        "\\midrule",
    ]
    lines.extend(" & ".join(row) + " \\\\" for row in rows)
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    main_rows = load_rows(Path(args.main_csv).resolve())
    if main_rows:
        write_table(
            output_dir / "main_comparison.tex",
            "Main comparison on the rebuilt residual current benchmark.",
            "tab:main_comparison",
            ["Method", "Type", "MAE", "RMSE", "NTAE", "CRPS", "Time/sample (s)"],
            [
                [
                    row["method"],
                    row["type"],
                    format_float(row.get("mae")),
                    format_float(row.get("rmse")),
                    format_float(row.get("ntae")),
                    format_float(row.get("crps")),
                    format_float(row.get("time_per_call_seconds")),
                ]
                for row in main_rows
            ],
        )
        cumulative_rows = [row for row in main_rows if row.get("method") in {"GFRC", "Diffusion", "CVAE", "Deterministic-Physics"}]
        write_table(
            output_dir / "cumulative_distribution.tex",
            "Direct sequence-level cumulative error summaries for representative methods.",
            "tab:cumulative_distribution",
            ["Method", "MedRCE", "P90-RCE", "Acc@100\\%"],
            [
                [
                    display_method(row["method"]),
                    format_float(row.get("med_rce")),
                    format_float(row.get("p90_rce")),
                    format_float(row.get("acc_at_100")),
                ]
                for row in cumulative_rows
            ],
        )
        uncertainty_rows = [row for row in main_rows if row.get("type") in {"Flow-based", "Probabilistic", "Invertible", "Diffusion"}]
        write_table(
            output_dir / "uncertainty_comparison.tex",
            "Predictive uncertainty quality of probabilistic models.",
            "tab:uncertainty_comparison",
            ["Method", "CRPS", "PICP@90", "MPIW@90"],
            [
                [
                    display_method(row["method"]),
                    format_float(row.get("crps")),
                    format_float(row.get("picp_90")),
                    format_float(row.get("mpiw_90")),
                ]
                for row in uncertainty_rows
            ],
        )

    ablation_rows = load_rows(Path(args.ablation_csv).resolve())
    if ablation_rows:
        write_table(
            output_dir / "ablation_comparison.tex",
            "Ablation study of cue usage and physical regularization.",
            "tab:ablation_comparison",
            ["Variant", "Use cue", "$\\lambda_{phys}$", "RMSE", "NTAE", "CRPS"],
            [
                [
                    row["variant"],
                    str(row["use_cue"]),
                    format_float(row.get("lambda_physical")),
                    format_float(row.get("rmse")),
                    format_float(row.get("ntae")),
                    format_float(row.get("crps")),
                ]
                for row in ablation_rows
            ],
        )

    efficiency_rows = load_rows(Path(args.efficiency_csv).resolve())
    if efficiency_rows:
        write_table(
            output_dir / "efficiency_comparison.tex",
            "Efficiency comparison across representative methods.",
            "tab:efficiency_comparison",
            ["Method", "Params", "Epoch time (s)", "Time/sample (s)", "RMSE", "NTAE"],
            [
                [
                    row["method"],
                    row["params"],
                    format_float(row.get("avg_epoch_seconds"), 1),
                    format_float(row.get("time_per_call_seconds")),
                    format_float(row.get("rmse")),
                    format_float(row.get("ntae")),
                ]
                for row in efficiency_rows
            ],
        )

    robustness_rows = load_rows(Path(args.robustness_csv).resolve())
    if robustness_rows:
        filtered = [row for row in robustness_rows if row.get("corruption") in {"clean", "noise_0.10", "missing_0.40", "wrong_cue"}]
        write_table(
            output_dir / "robustness_comparison.tex",
            "Robustness under imperfect target-branch power cues.",
            "tab:robustness_comparison",
            ["Method", "Corruption", "RMSE", "NTAE", "CRPS"],
            [
                [
                    display_method(row["method"]),
                    row["corruption"],
                    format_float(row.get("rmse")),
                    format_float(row.get("ntae")),
                    format_float(row.get("crps")),
                ]
                for row in filtered
            ],
        )

    physical_rows = load_rows(Path(args.physical_csv).resolve())
    if physical_rows:
        write_table(
            output_dir / "physical_consistency.tex",
            "Kirchhoff-style physical consistency measured by the residual between aggregate current and the sum of predicted branch currents.",
            "tab:physical_consistency",
            ["Method", "Pred. MAE", "Pred. P90", "Oracle MAE", "Norm. gap"],
            [
                [
                    display_method(row.get("method")),
                    format_float(row.get("pred_mae_mean")),
                    format_float(row.get("pred_mae_p90")),
                    format_float(row.get("oracle_mae_mean")),
                    format_float(row.get("pred_normalized_gap_mean")),
                ]
                for row in physical_rows
            ],
        )

    lambda_rows = load_rows(Path(args.lambda_csv).resolve())
    if lambda_rows:
        write_table(
            output_dir / "lambda_sweep.tex",
            "Sensitivity of GFRC to the physical-regularization weight $\\lambda_{phys}$.",
            "tab:lambda_sweep",
            ["$\\lambda_{phys}$", "RMSE", "NTAE", "CRPS"],
            [
                [
                    format_float(row.get("lambda_physical"), 2),
                    format_float(row.get("rmse")),
                    format_float(row.get("ntae")),
                    format_float(row.get("crps")),
                ]
                for row in lambda_rows
            ],
        )


if __name__ == "__main__":
    main()
