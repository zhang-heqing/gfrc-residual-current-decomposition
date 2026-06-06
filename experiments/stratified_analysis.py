import argparse
import math
from pathlib import Path
from typing import Dict, List
import sys

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.common import build_test_loader, load_model_from_run_dir, read_json, write_csv


class MetricAccumulator:
    def __init__(self) -> None:
        self.sum_sq_err = 0.0
        self.num_values = 0
        self.sum_abs_total_err = 0.0
        self.sum_abs_total_target = 0.0
        self.sum_clea = 0.0
        self.rce_values: List[float] = []
        self.num_sequences = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        diff = pred - target
        self.sum_sq_err += float((diff ** 2).sum().item())
        self.num_values += int(diff.numel())

        pred_total = pred.sum(dim=-1)
        target_total = target.sum(dim=-1)
        total_err = (pred_total - target_total).abs()
        target_abs = target_total.abs().clamp_min(1e-8)
        rce = total_err / target_abs
        self.sum_abs_total_err += float(total_err.sum().item())
        self.sum_abs_total_target += float(target_abs.sum().item())
        self.sum_clea += float((1.0 - total_err / target_abs).sum().item())
        self.rce_values.extend(rce.reshape(-1).tolist())
        self.num_sequences += int(target_total.numel())

    def finalize(self) -> Dict[str, float]:
        rce_array = np.asarray(self.rce_values, dtype=np.float64) if self.rce_values else np.asarray([float("nan")])
        return {
            "rmse": math.sqrt(self.sum_sq_err / max(self.num_values, 1)),
            "ntae": self.sum_abs_total_err / max(self.sum_abs_total_target, 1e-8),
            "clea": self.sum_clea / max(self.num_sequences, 1),
            "med_rce": float(np.nanmedian(rce_array)),
            "p90_rce": float(np.nanpercentile(rce_array, 90)),
            "acc_at_100": float(np.nanmean((rce_array <= 1.0).astype(np.float64))),
            "count": self.num_sequences,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stratified performance analysis over the test split.")
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--split-config", required=True)
    parser.add_argument("--output-csv", default="experiments/output/stratified_analysis.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--limit-batches", type=int)
    return parser.parse_args()


def collect_thresholds(model, test_loader, limit_batches: int | None) -> Dict[str, float]:
    residual_strengths: List[float] = []
    cue_strengths: List[float] = []
    batch_counter = 0

    for batch in tqdm(test_loader, desc="strata-thresholds", leave=False):
        x_r = batch["x_r"]
        branch_powers = batch["branch_powers"]
        branch_currents = batch["branch_currents"]

        total_current = model.denormalize_total_current(x_r)
        branch_currents_denorm = model.denormalize_branch_currents(branch_currents)

        for branch_index in range(branch_currents.size(1)):
            cue_power = model.denormalize_branch_power(branch_index, branch_powers[:, branch_index, :])
            residual_strengths.extend(torch.mean(torch.abs(total_current), dim=-1).tolist())
            cue_strengths.extend(torch.mean(torch.abs(cue_power), dim=-1).tolist())

        batch_counter += 1
        if limit_batches is not None and batch_counter >= limit_batches:
            break

    residual_q1, residual_q2 = np.quantile(np.asarray(residual_strengths), [0.33, 0.67]).tolist()
    cue_q1, cue_q2 = np.quantile(np.asarray(cue_strengths), [0.33, 0.67]).tolist()
    return {
        "residual_q1": residual_q1,
        "residual_q2": residual_q2,
        "cue_q1": cue_q1,
        "cue_q2": cue_q2,
    }


def label_by_quantile(value: float, low: float, high: float) -> str:
    if value <= low:
        return "low"
    if value <= high:
        return "medium"
    return "high"


def label_active_branch_count(branch_currents: torch.Tensor) -> List[str]:
    mean_abs = torch.mean(torch.abs(branch_currents), dim=-1)
    relative_threshold = torch.max(mean_abs, dim=-1, keepdim=True).values * 0.1
    counts = torch.sum(mean_abs > relative_threshold, dim=-1)
    labels = []
    for count in counts.tolist():
        if count <= 2:
            labels.append("sparse")
        elif count <= 5:
            labels.append("medium")
        else:
            labels.append("dense")
    return labels


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    first_run = Path(args.run_dir[0]).resolve()
    first_cfg = read_json(first_run / "run_config.json")
    seq_len = args.seq_len or int(first_cfg.get("seq_len", 100))
    test_loader, num_branches = build_test_loader(args.split_config, seq_len=seq_len, batch_size=args.batch_size)

    reference_model = load_model_from_run_dir(first_run, device)
    reference_model.num_branches = num_branches
    thresholds = collect_thresholds(reference_model, test_loader, args.limit_batches)

    rows: List[Dict[str, object]] = []
    for run_dir_value in args.run_dir:
        run_dir = Path(run_dir_value).resolve()
        model = load_model_from_run_dir(run_dir, device)
        model.num_branches = num_branches

        buckets: Dict[tuple[str, str], MetricAccumulator] = {}
        batch_counter = 0
        for batch in tqdm(test_loader, desc=f"stratified-{model.name}", leave=False):
            x_r = batch["x_r"].to(device)
            x_p = batch["x_p"].to(device)
            branch_powers = batch["branch_powers"].to(device)
            branch_currents = batch["branch_currents"].to(device)

            total_current_denorm = model.denormalize_total_current(x_r.detach().cpu())
            branch_currents_denorm_full = model.denormalize_branch_currents(branch_currents.detach().cpu())
            active_labels = label_active_branch_count(branch_currents_denorm_full)

            for branch_index in range(num_branches):
                target_p = branch_powers[:, branch_index, :]
                target_y = branch_currents[:, branch_index, :]
                mean, _ = model.sample(x_r, x_p, target_p)
                pred = model.denormalize_branch_current(branch_index, mean.detach().cpu())
                target = model.denormalize_branch_current(branch_index, target_y.detach().cpu())
                cue_power_denorm = model.denormalize_branch_power(branch_index, target_p.detach().cpu())

                residual_values = torch.mean(torch.abs(total_current_denorm), dim=-1).tolist()
                cue_values = torch.mean(torch.abs(cue_power_denorm), dim=-1).tolist()

                for item_index in range(pred.size(0)):
                    residual_label = label_by_quantile(
                        float(residual_values[item_index]),
                        thresholds["residual_q1"],
                        thresholds["residual_q2"],
                    )
                    cue_label = label_by_quantile(
                        float(cue_values[item_index]),
                        thresholds["cue_q1"],
                        thresholds["cue_q2"],
                    )
                    activity_label = active_labels[item_index]

                    for stratifier, label in [
                        ("residual_strength", residual_label),
                        ("cue_power_strength", cue_label),
                        ("active_branch_count", activity_label),
                    ]:
                        key = (stratifier, label)
                        if key not in buckets:
                            buckets[key] = MetricAccumulator()
                        buckets[key].update(pred[item_index : item_index + 1], target[item_index : item_index + 1])

            batch_counter += 1
            if args.limit_batches is not None and batch_counter >= args.limit_batches:
                break

        for (stratifier, label), accumulator in buckets.items():
            rows.append(
                {
                    "method": model.name,
                    "stratifier": stratifier,
                    "stratum": label,
                    **accumulator.finalize(),
                    "run_dir": str(run_dir),
                }
            )

    write_csv(
        Path(args.output_csv).resolve(),
        rows,
        fieldnames=["method", "stratifier", "stratum", "rmse", "ntae", "clea", "med_rce", "p90_rce", "acc_at_100", "count", "run_dir"],
    )


if __name__ == "__main__":
    main()
