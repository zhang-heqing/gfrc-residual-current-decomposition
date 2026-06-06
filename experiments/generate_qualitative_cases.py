import argparse
import json
from pathlib import Path
from typing import Dict, List
import sys

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common import build_test_loader, ensure_parent, load_model_from_run_dir, read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate qualitative case bundles and optional plots.")
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--compare-run", action="append", default=[])
    parser.add_argument("--split-config", required=True)
    parser.add_argument("--output-dir", default="experiments/output/qualitative")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--max-batches", type=int, default=12)
    parser.add_argument("--num-cases", type=int, default=3)
    return parser.parse_args()


def maybe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except Exception:
        return None


def model_label(run_dir: Path) -> str:
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        payload = read_json(config_path)
        return str(payload.get("method_name", payload.get("dataset_name", run_dir.name)))
    return run_dir.name


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_run = Path(args.reference_run).resolve()
    reference_cfg = read_json(reference_run / "run_config.json")
    seq_len = args.seq_len or int(reference_cfg.get("seq_len", 100))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reference_model = load_model_from_run_dir(reference_run, device)
    compare_models = [load_model_from_run_dir(Path(item).resolve(), device) for item in args.compare_run]
    all_models = [reference_model, *compare_models]

    test_loader, num_branches = build_test_loader(args.split_config, seq_len=seq_len, batch_size=args.batch_size)
    for model in all_models:
        model.num_branches = num_branches

    candidates: List[Dict[str, object]] = []
    batch_counter = 0
    for batch in tqdm(test_loader, desc="collect-qualitative", leave=False):
        x_r = batch["x_r"].to(device)
        x_p = batch["x_p"].to(device)
        branch_powers = batch["branch_powers"].to(device)
        branch_currents = batch["branch_currents"].to(device)

        for branch_index in range(num_branches):
            target_p = branch_powers[:, branch_index, :]
            target_y = branch_currents[:, branch_index, :]

            ref_mean, ref_samples = reference_model.sample(x_r, x_p, target_p)
            ref_lower = torch.quantile(ref_samples, 0.05, dim=0)
            ref_upper = torch.quantile(ref_samples, 0.95, dim=0)

            compare_predictions = {}
            for model in compare_models:
                mean, _ = model.sample(x_r, x_p, target_p)
                compare_predictions[model.name] = mean.detach().cpu()

            for item_index in range(x_r.size(0)):
                target_denorm = reference_model.denormalize_branch_current(branch_index, target_y[item_index].detach().cpu())
                ref_mean_denorm = reference_model.denormalize_branch_current(branch_index, ref_mean[item_index].detach().cpu())
                rmse = float(torch.sqrt(torch.mean((ref_mean_denorm - target_denorm) ** 2)).item())
                target_abs = float(torch.mean(torch.abs(target_denorm)).item())
                candidates.append(
                    {
                        "rmse": rmse,
                        "target_abs": target_abs,
                        "branch_index": branch_index,
                        "total_current": reference_model.denormalize_total_current(x_r[item_index].detach().cpu()).tolist(),
                        "total_power": reference_model.denormalize_total_power(x_p[item_index].detach().cpu()).tolist(),
                        "cue_power": reference_model.denormalize_branch_power(branch_index, target_p[item_index].detach().cpu()).tolist(),
                        "target": target_denorm.tolist(),
                        "reference_mean": ref_mean_denorm.tolist(),
                        "reference_p05": reference_model.denormalize_branch_current(branch_index, ref_lower[item_index].detach().cpu()).tolist(),
                        "reference_p95": reference_model.denormalize_branch_current(branch_index, ref_upper[item_index].detach().cpu()).tolist(),
                        "compare_predictions": {
                            model_name: model.denormalize_branch_current(branch_index, tensor[item_index]).tolist()
                            for model_name, tensor, model in [
                                (model.name, compare_predictions[model.name], model) for model in compare_models
                            ]
                        },
                    }
                )

        batch_counter += 1
        if batch_counter >= args.max_batches:
            break

    if not candidates:
        raise RuntimeError("No qualitative candidates were collected.")

    ordered = sorted(candidates, key=lambda item: float(item["rmse"]))
    positions = [0, len(ordered) // 2, len(ordered) - 1]
    labels = ["easy", "ambiguous", "hard"]
    chosen = [ordered[min(position, len(ordered) - 1)] for position in positions[: args.num_cases]]
    chosen_labels = labels[: len(chosen)]

    plt = maybe_import_matplotlib()
    manifest_rows = []
    for label, case in zip(chosen_labels, chosen):
        case_dir = output_dir / label
        case_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "label": label,
            "branch_index": int(case["branch_index"]) + 1,
            "rmse": case["rmse"],
            "target_abs_mean": case["target_abs"],
            "series": {
                "total_current": case["total_current"],
                "total_power": case["total_power"],
                "cue_power": case["cue_power"],
                "target": case["target"],
                "reference_mean": case["reference_mean"],
                "reference_p05": case["reference_p05"],
                "reference_p95": case["reference_p95"],
                **case["compare_predictions"],
            },
        }
        json_path = case_dir / "case.json"
        json_path.write_text(json.dumps(payload, indent=2))
        manifest_rows.append(
            {
                "label": label,
                "branch_index": int(case["branch_index"]) + 1,
                "rmse": case["rmse"],
                "json_path": str(json_path),
            }
        )

        if plt is not None:
            x_axis = list(range(len(case["target"])))
            fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            axes[0].plot(x_axis, case["target"], label="Ground truth", linewidth=2.0, color="#111111")
            axes[0].plot(x_axis, case["reference_mean"], label=reference_model.name, linewidth=1.8, color="#1b6ef3")
            axes[0].fill_between(x_axis, case["reference_p05"], case["reference_p95"], color="#1b6ef3", alpha=0.18)
            for model_name, series in case["compare_predictions"].items():
                axes[0].plot(x_axis, series, label=model_name, linewidth=1.4)
            axes[0].set_ylabel("Branch current")
            axes[0].legend(loc="upper right", fontsize=8)
            axes[0].set_title(f"{label.title()} case | branch {int(case['branch_index']) + 1} | RMSE={float(case['rmse']):.4f}")

            axes[1].plot(x_axis, case["total_current"], label="Total residual current", color="#d95f02")
            axes[1].plot(x_axis, case["cue_power"], label="Target branch power cue", color="#0f9d58")
            axes[1].set_xlabel("Time step")
            axes[1].set_ylabel("Auxiliary signals")
            axes[1].legend(loc="upper right", fontsize=8)
            fig.tight_layout()
            fig.savefig(case_dir / "case.png", dpi=180)
            plt.close(fig)

    (output_dir / "manifest.json").write_text(json.dumps(manifest_rows, indent=2))


if __name__ == "__main__":
    main()

