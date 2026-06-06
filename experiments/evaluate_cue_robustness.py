import argparse
import json
import time
from pathlib import Path
from typing import Dict, List
import sys

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
GFRC_ROOT = REPO_ROOT / "gfrc_full_impl"
if str(GFRC_ROOT) not in sys.path:
    sys.path.insert(0, str(GFRC_ROOT))

from experiments.common import LoadedModel, build_test_loader, ensure_parent, load_model_from_run_dir, read_json, write_csv
from baselines.shared.common import StreamingEvaluator
from gfrc_full_impl.config import TrainConfig
from gfrc_full_impl.dataset import create_explicit_split_data_loaders
from gfrc_full_impl.trainer import build_trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate cue robustness for a trained run directory.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split-config", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--limit-batches", type=int)
    parser.add_argument("--limit-branches", type=int)
    parser.add_argument(
        "--corruptions",
        help="Comma-separated subset of corruption names to evaluate. Defaults to the full suite.",
    )
    return parser.parse_args()


def apply_corruption(
    target_p: torch.Tensor,
    branch_powers: torch.Tensor,
    branch_index: int,
    corruption_name: str,
) -> torch.Tensor:
    if corruption_name == "clean":
        return target_p

    if corruption_name.startswith("noise_"):
        level = float(corruption_name.split("_", 1)[1])
        scale = target_p.std(dim=-1, keepdim=True).clamp_min(1e-6)
        return target_p + torch.randn_like(target_p) * scale * level

    if corruption_name.startswith("missing_"):
        level = float(corruption_name.split("_", 1)[1])
        missing_len = max(1, int(target_p.size(-1) * level))
        start = max((target_p.size(-1) - missing_len) // 2, 0)
        corrupted = target_p.clone()
        corrupted[:, start : start + missing_len] = 0.0
        return corrupted

    if corruption_name.startswith("scale_"):
        factor = float(corruption_name.split("_", 1)[1])
        return target_p * factor

    if corruption_name == "wrong_cue":
        wrong_index = (branch_index + 1) % branch_powers.size(1)
        return branch_powers[:, wrong_index, :]

    raise ValueError(f"Unsupported corruption: {corruption_name}")


def evaluate_corruption(
    model: LoadedModel,
    test_loader,
    corruption_name: str,
    limit_batches: int | None,
    limit_branches: int | None,
    calibrators=None,
    calibration_branch_power_stats=None,
) -> Dict[str, float]:
    evaluator = StreamingEvaluator(alpha=0.1)
    branch_limit = model.num_branches if limit_branches is None else min(limit_branches, model.num_branches)

    batch_count = 0
    for batch in tqdm(test_loader, desc=corruption_name, leave=False):
        x_r = batch["x_r"].to(model.device)
        x_p = batch["x_p"].to(model.device)
        branch_powers = batch["branch_powers"].to(model.device)
        branch_currents = batch["branch_currents"].to(model.device)

        for branch_index in range(branch_limit):
            target_p = branch_powers[:, branch_index, :]
            target_y = branch_currents[:, branch_index, :]
            corrupted_p = apply_corruption(target_p, branch_powers, branch_index, corruption_name)
            start = time.perf_counter()
            mean = model.predict_point(x_r, x_p, corrupted_p)
            _, samples = model.sample(x_r, x_p, corrupted_p)
            mean = model.denormalize_branch_current(branch_index, mean.detach().cpu())
            target = model.denormalize_branch_current(branch_index, target_y.detach().cpu())
            samples = model.denormalize_branch_current(branch_index, samples.detach().cpu())
            if calibrators is not None and calibration_branch_power_stats is not None:
                power_std, power_mean = calibration_branch_power_stats
                cue_sequence = corrupted_p.detach().cpu() * float(power_std[branch_index]) + float(power_mean[branch_index])
                mean, _ = model.calibration_bridge._apply_sequence_total_calibration(
                    branch_index=branch_index,
                    mean=mean,
                    samples=mean.unsqueeze(0),
                    cue_sequence=cue_sequence,
                    calibrators=calibrators,
                )
                _, samples = model.calibration_bridge._apply_sequence_total_calibration(
                    branch_index=branch_index,
                    mean=mean,
                    samples=samples,
                    cue_sequence=cue_sequence,
                    calibrators=calibrators,
                )
            evaluator.update(
                mean,
                target,
                samples,
                elapsed_seconds=time.perf_counter() - start,
            )

        batch_count += 1
        if limit_batches is not None and batch_count >= limit_batches:
            break

    return evaluator.finalize()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    run_config = read_json(run_dir / "run_config.json")
    seq_len = args.seq_len or int(run_config.get("seq_len", 100))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_run_dir(run_dir, device)
    test_loader, num_branches = build_test_loader(args.split_config, seq_len=seq_len, batch_size=args.batch_size)
    model.num_branches = num_branches
    calibrators = None
    calibration_branch_power_stats = None
    if model.name == "GFRC":
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

    corruption_names = [
        "clean",
        "noise_0.05",
        "noise_0.10",
        "noise_0.20",
        "missing_0.20",
        "missing_0.40",
        "missing_0.60",
        "scale_0.80",
        "scale_1.20",
        "wrong_cue",
    ]
    if args.corruptions:
        allowed = {name.strip() for name in args.corruptions.split(",") if name.strip()}
        corruption_names = [name for name in corruption_names if name in allowed]

    results: Dict[str, Dict[str, float]] = {}
    csv_rows: List[Dict[str, object]] = []
    for corruption_name in corruption_names:
        metrics = evaluate_corruption(
            model=model,
            test_loader=test_loader,
            corruption_name=corruption_name,
            limit_batches=args.limit_batches,
            limit_branches=args.limit_branches,
            calibrators=calibrators,
            calibration_branch_power_stats=calibration_branch_power_stats,
        )
        results[corruption_name] = metrics
        print(
            f"[{corruption_name}] rmse={metrics['rmse']:.4f} "
            f"ntae={metrics['ntae']:.4f} crps={metrics['crps']:.4f}"
        )
        csv_rows.append(
            {
                "method": model.name,
                "corruption": corruption_name,
                **metrics,
            }
        )

    output_json = Path(args.output_json).resolve()
    ensure_parent(output_json)
    output_json.write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "method": model.name,
                "results": results,
            },
            indent=2,
        )
    )

    if args.output_csv:
        write_csv(
            Path(args.output_csv).resolve(),
            csv_rows,
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
