import argparse
import json
from pathlib import Path

import numpy as np
import torch

from config import TrainConfig
from dataset import create_explicit_split_data_loaders
from trainer import build_trainer
from train import load_split_config, resolve_split_csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover missing GFRC evaluation artifacts from an existing checkpoint without retraining."
    )
    parser.add_argument("--split-config", required=True, help="Path to the benchmark split_config.json.")
    parser.add_argument("--output-dir", required=True, help="Directory containing best_gfrc_model.pt.")
    parser.add_argument("--sample-steps", type=int, default=None, help="Override evaluation ODE sampling steps.")
    parser.add_argument(
        "--num-eval-samples",
        type=int,
        default=None,
        help="Override the number of stochastic samples used during evaluation.",
    )
    return parser.parse_args()


def build_run_config(
    *,
    config: TrainConfig,
    benchmark_name: str,
    num_branches: int,
    split_paths: dict[str, Path],
) -> dict[str, object]:
    return {
        **config.__dict__,
        "dataset_name": benchmark_name,
        "num_branches": num_branches,
        "train_data_path": str(split_paths["train"]),
        "val_data_path": str(split_paths["val"]),
        "test_data_path": str(split_paths["test"]),
    }


def main() -> None:
    args = parse_args()
    split_config_path = Path(args.split_config).resolve()
    output_dir = Path(args.output_dir).resolve()
    checkpoint_path = output_dir / "best_gfrc_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    split_config = load_split_config(split_config_path)
    split_paths = {
        "train": resolve_split_csv_path(split_config_path, str(split_config["train_csv_path"])),
        "val": resolve_split_csv_path(split_config_path, str(split_config["val_csv_path"])),
        "test": resolve_split_csv_path(split_config_path, str(split_config["test_csv_path"])),
    }

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = TrainConfig(**checkpoint["config"])
    config.output_dir = output_dir
    if args.sample_steps is not None:
        config.sample_steps = args.sample_steps
    if args.num_eval_samples is not None:
        config.num_eval_samples = args.num_eval_samples

    train_loader, val_loader, test_loader, stats, num_branches = create_explicit_split_data_loaders(
        train_data_path=str(split_paths["train"]),
        val_data_path=str(split_paths["val"]),
        test_data_path=str(split_paths["test"]),
        seq_len=config.seq_len,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_sampling=config.train_sampling,
        augment_noise_std=config.augment_noise_std,
        augment_scale_range=(config.augment_scale_min, config.augment_scale_max),
    )
    del train_loader

    trainer = build_trainer(config=config, num_branches=num_branches, stats=stats)
    metrics = trainer.evaluate(test_loader, calibration_loader=val_loader)

    run_config = build_run_config(
        config=config,
        benchmark_name=str(split_config["benchmark_name"]),
        num_branches=num_branches,
        split_paths=split_paths,
    )
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str))
    (output_dir / "normalization_stats.json").write_text(json.dumps(stats.to_dict(), indent=2))

    history_path = output_dir / "training_history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text())
        epoch_seconds = history.get("epoch_seconds", [])
        summary_payload = {
            "best_checkpoint_metric": checkpoint.get("best_checkpoint_metric", checkpoint.get("best_val_loss")),
            "checkpoint_metric_name": checkpoint.get("checkpoint_metric_name", "loss"),
            "epoch_seconds": epoch_seconds,
            "avg_epoch_seconds": float(np.mean(epoch_seconds)) if epoch_seconds else 0.0,
            "final_test_metrics": metrics,
        }
        (output_dir / "training_summary.json").write_text(json.dumps(summary_payload, indent=2))

    print(f"Recovered artifacts in {output_dir}")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
