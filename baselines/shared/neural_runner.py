import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from baselines.shared.common import (
    DEFAULT_SPLIT_CONFIG,
    StreamingEvaluator,
    count_parameters,
    create_loaders_from_split_config,
    sample_target_branch,
    save_json,
    set_seed,
)
from baselines.shared.models import build_model


def default_output_dir(method_name: str) -> Path:
    return Path("training_runs") / "baselines" / method_name


def build_arg_parser(method_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Train baseline method: {method_name}")
    parser.add_argument("--split-config", default=str(DEFAULT_SPLIT_CONFIG))
    parser.add_argument("--output-dir", default=str(default_output_dir(method_name)))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda-physical", type=float, default=0.0)
    parser.add_argument("--disable-cue", action="store_true")
    parser.add_argument("--patch-size", type=int, default=10)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--flow-layers", type=int, default=4)
    parser.add_argument("--diffusion-steps", type=int, default=24)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument("--limit-train-batches", type=int)
    parser.add_argument("--limit-val-batches", type=int)
    parser.add_argument("--limit-test-batches", type=int)
    parser.add_argument("--limit-test-branches", type=int)
    return parser


def run_epoch(
    *,
    model,
    loader,
    optimizer: Optional[optim.Optimizer],
    device: torch.device,
    num_branches: int,
    lambda_physical: float,
    limit_batches: Optional[int],
    grad_clip_norm: float,
) -> Dict[str, float]:
    train = optimizer is not None
    model.train(mode=train)
    running: Dict[str, float] = {}
    progress = tqdm(loader, leave=False, desc="train" if train else "val")
    batch_count = 0
    for batch in progress:
        x_r = batch["x_r"].to(device)
        x_p = batch["x_p"].to(device)
        branch_powers = batch["branch_powers"].to(device)
        branch_currents = batch["branch_currents"].to(device)
        _, target_p, target_y = sample_target_branch(branch_powers, branch_currents)
        if getattr(model, "use_cue", True) is False:
            target_p = torch.zeros_like(target_p)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            loss_terms = model.loss_terms(x_r, x_p, target_p, target_y)
            total_loss = loss_terms["loss"]
            if lambda_physical > 0.0:
                branch_power_input = branch_powers if getattr(model, "use_cue", True) else torch.zeros_like(branch_powers)
                predicted_branches = model.predict_all_branches(x_r, x_p, branch_power_input)
                physical_loss = F.mse_loss(predicted_branches.sum(dim=1), x_r)
                total_loss = total_loss + lambda_physical * physical_loss
                loss_terms["physical_loss"] = physical_loss
                loss_terms["loss"] = total_loss

            if train:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()

        detached = {key: float(value.item()) for key, value in loss_terms.items()}
        for key, value in detached.items():
            running[key] = running.get(key, 0.0) + value
        progress.set_postfix({key.replace("_loss", ""): f"{value:.4f}" for key, value in detached.items()})
        batch_count += 1
        if limit_batches is not None and batch_count >= limit_batches:
            break

    divisor = max(batch_count, 1)
    return {key: value / divisor for key, value in running.items()}


def evaluate(
    *,
    model,
    loader,
    device: torch.device,
    num_branches: int,
    num_samples: int,
    limit_batches: Optional[int],
    limit_branches: Optional[int],
    stats,
) -> Dict[str, float]:
    model.eval()
    evaluator = StreamingEvaluator(alpha=0.1)
    branch_limit = num_branches if limit_branches is None else min(limit_branches, num_branches)
    batch_count = 0

    for batch in tqdm(loader, leave=False, desc="test"):
        x_r = batch["x_r"].to(device)
        x_p = batch["x_p"].to(device)
        branch_powers = batch["branch_powers"].to(device)
        branch_currents = batch["branch_currents"].to(device)

        for branch_index in range(branch_limit):
            target_p = branch_powers[:, branch_index, :]
            if getattr(model, "use_cue", True) is False:
                target_p = torch.zeros_like(target_p)
            target_y = branch_currents[:, branch_index, :]
            start_time = time.perf_counter()
            mean, samples = model.sample(x_r, x_p, target_p, num_samples=num_samples)
            elapsed = time.perf_counter() - start_time
            std = float(stats.branch_current_std[branch_index])
            offset = float(stats.branch_current_mean[branch_index])
            evaluator.update(
                mean * std + offset,
                target_y * std + offset,
                samples=samples * std + offset,
                elapsed_seconds=elapsed,
            )

        batch_count += 1
        if limit_batches is not None and batch_count >= limit_batches:
            break

    return evaluator.finalize()


def main(method_name: str) -> None:
    parser = build_arg_parser(method_name)
    args = parser.parse_args()

    if method_name == "deterministic_physics" and args.lambda_physical == 0.0:
        args.lambda_physical = 0.1

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    split_config, loaders = create_loaders_from_split_config(
        args.split_config,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_sampling="signal_weighted",
    )
    train_loader, val_loader, test_loader, stats, num_branches = loaders

    model = build_model(method_name, args).to(device)
    model.use_cue = not args.disable_cue
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    history = {"train": [], "val": [], "epoch_seconds": []}
    checkpoint_path = output_dir / "best_model.pt"

    print(f"Training {method_name} on {device}")
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Split config: {args.split_config}")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            num_branches=num_branches,
            lambda_physical=args.lambda_physical,
            limit_batches=args.limit_train_batches,
            grad_clip_norm=args.grad_clip_norm,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            num_branches=num_branches,
            lambda_physical=args.lambda_physical,
            limit_batches=args.limit_val_batches,
            grad_clip_norm=args.grad_clip_norm,
        )
        scheduler.step()
        epoch_seconds = time.perf_counter() - epoch_start

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)
        history["epoch_seconds"].append(epoch_seconds)
        val_loss = val_metrics["loss"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "method_name": method_name,
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "num_branches": num_branches,
                    "best_val_loss": best_val_loss,
                },
                checkpoint_path,
            )

        print(
            f"Epoch {epoch:03d} | "
            f"train={train_metrics['loss']:.4f} | "
            f"val={val_metrics['loss']:.4f} | "
            f"time={epoch_seconds:.1f}s"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        num_branches=num_branches,
        num_samples=args.num_samples,
        limit_batches=args.limit_test_batches,
        limit_branches=args.limit_test_branches,
        stats=stats,
    )

    save_json(output_dir / "training_history.json", history)
    save_json(output_dir / "evaluation_metrics.json", test_metrics)
    save_json(output_dir / "normalization_stats.json", stats.to_dict())
    summary_payload = {
        "best_val_loss": best_val_loss,
        "final_test_metrics": test_metrics,
        "epoch_seconds": history["epoch_seconds"],
        "avg_epoch_seconds": float(sum(history["epoch_seconds"]) / max(len(history["epoch_seconds"]), 1)),
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary_payload, indent=2))
    save_json(
        output_dir / "run_config.json",
        {
            **vars(args),
            "method_name": method_name,
            "device": str(device),
            "num_branches": num_branches,
            "benchmark_name": split_config["benchmark_name"],
            "use_cue": not args.disable_cue,
        },
    )

    print("Final test metrics:")
    for key, value in test_metrics.items():
        print(f"{key}: {value:.6f}")
