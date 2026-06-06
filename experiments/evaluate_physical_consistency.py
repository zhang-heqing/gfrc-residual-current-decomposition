import argparse
import json
from pathlib import Path
from typing import Dict, List
import sys

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
GFRC_ROOT = REPO_ROOT / "gfrc_full_impl"
if str(GFRC_ROOT) not in sys.path:
    sys.path.insert(0, str(GFRC_ROOT))

from experiments.common import (
    LoadedModel,
    build_test_loader,
    ensure_parent,
    load_model_from_run_dir,
    project_branch_predictions_to_total,
    write_csv,
)
from gfrc_full_impl.config import TrainConfig
from gfrc_full_impl.dataset import create_explicit_split_data_loaders
from gfrc_full_impl.trainer import build_trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Kirchhoff-style physical consistency for a trained run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split-config", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--method-label", help="Optional label written into the summary payload.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--limit-batches", type=int)
    parser.add_argument("--num-samples-override", type=int)
    parser.add_argument("--projection-mode", choices=["none", "uniform", "magnitude", "power_weighted"], default="none")
    parser.add_argument("--projection-alpha", type=float, default=1.0)
    parser.add_argument("--disable-calibration", action="store_true")
    return parser.parse_args()


def summarize(values: List[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()) if len(array) else 0.0,
        "median": float(np.median(array)) if len(array) else 0.0,
        "p90": float(np.quantile(array, 0.9)) if len(array) else 0.0,
        "p95": float(np.quantile(array, 0.95)) if len(array) else 0.0,
        "max": float(array.max()) if len(array) else 0.0,
    }


def evaluate(
    model: LoadedModel,
    test_loader,
    limit_batches: int | None,
    projection_mode: str,
    projection_alpha: float,
    calibrators=None,
    calibration_branch_power_stats: tuple[np.ndarray, np.ndarray] | None = None,
) -> Dict[str, object]:
    pred_residual_values: List[float] = []
    oracle_residual_values: List[float] = []
    pred_total_abs_values: List[float] = []
    total_current_abs_values: List[float] = []
    batch_rows: List[Dict[str, object]] = []

    batch_counter = 0
    for batch in tqdm(test_loader, desc="physical-consistency", leave=False):
        x_r = batch["x_r"].to(model.device)
        x_p = batch["x_p"].to(model.device)
        branch_powers = batch["branch_powers"].to(model.device)
        branch_currents = batch["branch_currents"].to(model.device)

        predicted_branches = []
        for branch_index in range(model.num_branches):
            target_p = branch_powers[:, branch_index, :]
            if not model.use_cue:
                target_p = torch.zeros_like(target_p)
            point_pred = model.predict_point(x_r, x_p, target_p)
            pred_branch = model.denormalize_branch_current(branch_index, point_pred.detach().cpu())
            if calibrators is not None and calibration_branch_power_stats is not None:
                power_std, power_mean = calibration_branch_power_stats
                cue_sequence = target_p.detach().cpu() * float(power_std[branch_index]) + float(power_mean[branch_index])
                pred_branch, _ = model.calibration_bridge._apply_sequence_total_calibration(
                    branch_index=branch_index,
                    mean=pred_branch,
                    samples=pred_branch.unsqueeze(0),
                    cue_sequence=cue_sequence,
                    calibrators=calibrators,
                )
            predicted_branches.append(pred_branch)

        predicted_branches_tensor = torch.stack(predicted_branches, dim=1)
        target_sum = model.denormalize_branch_currents(branch_currents.detach().cpu()).sum(dim=1)
        total_current = model.denormalize_total_current(x_r.detach().cpu())
        branch_powers_denorm = model.denormalize_branch_currents(branch_powers.detach().cpu())

        if projection_mode != "none":
            predicted_branches_tensor = project_branch_predictions_to_total(
                predicted_branches=predicted_branches_tensor,
                total_current=total_current,
                branch_powers=branch_powers_denorm,
                mode=projection_mode,
                alpha=projection_alpha,
            )

        pred_sum = predicted_branches_tensor.sum(dim=1)

        pred_residual = torch.mean(torch.abs(total_current - pred_sum), dim=-1)
        oracle_residual = torch.mean(torch.abs(total_current - target_sum), dim=-1)
        pred_total_abs = torch.mean(torch.abs(pred_sum), dim=-1)
        total_current_abs = torch.mean(torch.abs(total_current), dim=-1)

        pred_residual_values.extend(pred_residual.tolist())
        oracle_residual_values.extend(oracle_residual.tolist())
        pred_total_abs_values.extend(pred_total_abs.tolist())
        total_current_abs_values.extend(total_current_abs.tolist())

        for item_index in range(pred_residual.size(0)):
            batch_rows.append(
                {
                    "batch_index": batch_counter,
                    "sample_index_in_batch": item_index,
                    "pred_residual_mae": float(pred_residual[item_index]),
                    "oracle_residual_mae": float(oracle_residual[item_index]),
                    "pred_total_abs_mean": float(pred_total_abs[item_index]),
                    "total_current_abs_mean": float(total_current_abs[item_index]),
                    "projection_mode": projection_mode,
                    "projection_alpha": projection_alpha,
                }
            )

        batch_counter += 1
        if limit_batches is not None and batch_counter >= limit_batches:
            break

    pred_summary = summarize(pred_residual_values)
    oracle_summary = summarize(oracle_residual_values)
    normalized_gap = float(np.mean(np.asarray(pred_residual_values) / np.maximum(np.asarray(total_current_abs_values), 1e-8))) if pred_residual_values else 0.0
    oracle_normalized_gap = float(np.mean(np.asarray(oracle_residual_values) / np.maximum(np.asarray(total_current_abs_values), 1e-8))) if oracle_residual_values else 0.0

    return {
        "predicted_consistency_mae": pred_summary,
        "oracle_consistency_mae": oracle_summary,
        "predicted_normalized_gap_mean": normalized_gap,
        "oracle_normalized_gap_mean": oracle_normalized_gap,
        "num_windows": len(pred_residual_values),
        "per_window_rows": batch_rows,
    }


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    run_config = json.loads((run_dir / "run_config.json").read_text())
    seq_len = args.seq_len or int(run_config.get("seq_len", 100))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_run_dir(run_dir, device)
    if args.num_samples_override is not None:
        model.num_samples = args.num_samples_override
    test_loader, num_branches = build_test_loader(args.split_config, seq_len=seq_len, batch_size=args.batch_size)
    model.num_branches = num_branches
    calibrators = None
    calibration_branch_power_stats = None
    if model.name == "GFRC" and not args.disable_calibration:
        checkpoint = torch.load(run_dir / "best_gfrc_model.pt", map_location="cpu", weights_only=False)
        config = TrainConfig(**checkpoint["config"])
        config.output_dir = run_dir
        split_config_path_obj = Path(args.split_config).resolve()
        split_config = json.loads(split_config_path_obj.read_text())
        train_path = split_config_path_obj.parent / str(split_config["train_csv_path"])
        val_path = split_config_path_obj.parent / str(split_config["val_csv_path"])
        test_path = split_config_path_obj.parent / str(split_config["test_csv_path"])
        _, val_loader, _, stats, num_branches = create_explicit_split_data_loaders(
            train_data_path=str(train_path.resolve()),
            val_data_path=str(val_path.resolve()),
            test_data_path=str(test_path.resolve()),
            seq_len=config.seq_len,
            batch_size=args.batch_size,
            num_workers=config.num_workers,
            train_sampling=config.train_sampling,
            augment_noise_std=config.augment_noise_std,
            augment_scale_range=(config.augment_scale_min, config.augment_scale_max),
        )
        trainer = build_trainer(config=config, num_branches=num_branches, stats=stats)
        trainer.load_best_checkpoint()
        calibrators = trainer._fit_sequence_total_calibrators(val_loader)
        calibration_branch_power_stats = (stats.branch_power_std, stats.branch_power_mean)
        model.calibration_bridge = trainer

    results = evaluate(
        model,
        test_loader,
        args.limit_batches,
        args.projection_mode,
        args.projection_alpha,
        calibrators=calibrators,
        calibration_branch_power_stats=calibration_branch_power_stats,
    )

    output_json = Path(args.output_json).resolve()
    ensure_parent(output_json)
    payload = {
        "run_dir": str(run_dir),
        "method": args.method_label or model.name,
        "base_method": model.name,
        "projection_mode": args.projection_mode,
        "projection_alpha": args.projection_alpha,
        **{key: value for key, value in results.items() if key != "per_window_rows"},
    }
    output_json.write_text(json.dumps(payload, indent=2))

    write_csv(
        Path(args.output_csv).resolve(),
        results["per_window_rows"],
        fieldnames=[
            "batch_index",
            "sample_index_in_batch",
            "pred_residual_mae",
            "oracle_residual_mae",
            "pred_total_abs_mean",
            "total_current_abs_mean",
            "projection_mode",
            "projection_alpha",
        ],
    )


if __name__ == "__main__":
    main()
