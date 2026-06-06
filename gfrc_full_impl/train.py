import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Optional, Sequence

GFRC_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(GFRC_ROOT) in sys.path:
    sys.path.remove(str(GFRC_ROOT))
sys.path.insert(0, str(GFRC_ROOT))

from config import TrainConfig
from dataset import create_data_loaders, create_explicit_split_data_loaders
from trainer import build_trainer

try:
    from data_preprocessing.path_utils import resolve_manifest_artifact_path, resolve_split_artifact_path
except ModuleNotFoundError:
    def _dedupe(paths):
        seen = set()
        ordered = []
        for path in paths:
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            ordered.append(path)
        return ordered

    def _resolve_first_existing(candidates):
        ordered = _dedupe(path.expanduser() for path in candidates)
        for candidate in ordered:
            if candidate.exists():
                return candidate.resolve()
        return ordered[-1].resolve()

    def resolve_manifest_artifact_path(manifest_path: Path, path_value: str) -> Path:
        raw_path = Path(path_value).expanduser()
        base_dir = manifest_path.parent
        candidates = []
        if raw_path.is_absolute():
            candidates.append(raw_path)
            candidates.append(base_dir / raw_path.name)
            if raw_path.parent.name:
                candidates.append(base_dir / raw_path.parent.name / raw_path.name)
        else:
            candidates.append((Path.cwd() / raw_path).resolve())
            candidates.append((base_dir / raw_path).resolve())
            candidates.append((base_dir / raw_path.name).resolve())
            if len(raw_path.parts) >= 2:
                candidates.append((base_dir / raw_path.parts[-2] / raw_path.name).resolve())
        return _resolve_first_existing(candidates)

    def resolve_split_artifact_path(split_config_path: Path, path_value: str) -> Path:
        raw_path = Path(path_value).expanduser()
        base_dir = split_config_path.parent
        candidates = []
        if raw_path.is_absolute():
            candidates.append(raw_path)
            candidates.append(base_dir / raw_path.name)
        else:
            candidates.append((Path.cwd() / raw_path).resolve())
            candidates.append((base_dir / raw_path).resolve())
            candidates.append((base_dir / raw_path.name).resolve())
        return _resolve_first_existing(candidates)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the full GFRC implementation.")
    parser.add_argument("--data", help="Path to a single residual-current CSV file.")
    parser.add_argument("--train-data", help="Path to the training CSV for an explicit split benchmark.")
    parser.add_argument("--val-data", help="Path to the validation CSV for an explicit split benchmark.")
    parser.add_argument("--test-data", help="Path to the test CSV for an explicit split benchmark.")
    parser.add_argument("--split-config", help="Path to a split_config.json file produced by the benchmark builder.")
    parser.add_argument("--manifest", help="Path to a preprocessing manifest.json file.")
    parser.add_argument(
        "--dataset-name",
        action="append",
        help="Dataset name from the manifest to train. Can be passed multiple times. Defaults to all datasets in the manifest.",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List datasets available in the manifest and exit.",
    )
    parser.add_argument("--output-dir", default="outputs", help="Directory for checkpoints and metrics.")
    parser.add_argument("--init-checkpoint", help="Optional checkpoint whose model weights will be loaded before training.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--flow-dim", type=int, default=256)
    parser.add_argument("--num-encoder-layers", type=int, default=2)
    parser.add_argument("--num-flow-blocks", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=4, help="Number of stochastic samples during evaluation.")
    parser.add_argument("--lambda-physical", type=float, default=0.1)
    parser.add_argument("--lambda-endpoint", type=float, default=0.5)
    parser.add_argument("--lambda-sequence-total", type=float, default=0.5)
    parser.add_argument("--lambda-multiscale-total", type=float, default=0.0)
    parser.add_argument("--lambda-prefix-cumsum", type=float, default=0.0)
    parser.add_argument("--lambda-delta", type=float, default=0.0)
    parser.add_argument("--lambda-activity", type=float, default=0.0)
    parser.add_argument("--lambda-alarm", type=float, default=0.0)
    parser.add_argument("--warmup-epochs-physical", type=int, default=0)
    parser.add_argument("--warmup-epochs-prefix", type=int, default=0)
    parser.add_argument("--warmup-epochs-delta", type=int, default=0)
    parser.add_argument("--warmup-epochs-sequence-total", type=int, default=0)
    parser.add_argument("--cue-dropout-prob", type=float, default=0.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--activity-threshold", type=float, default=0.01)
    parser.add_argument("--activity-sharpness", type=float, default=40.0)
    parser.add_argument("--alarm-threshold", type=float, default=0.05)
    parser.add_argument("--alarm-sharpness", type=float, default=10.0)
    parser.add_argument("--sample-steps", type=int, default=16)
    parser.add_argument("--physical-sample-steps", type=int, default=8)
    parser.add_argument("--ode-solver", choices=["euler", "heun"], default="euler")
    parser.add_argument("--physical-ode-solver", choices=["euler", "heun"], default="euler")
    parser.add_argument(
        "--eval-point-estimate-mode",
        choices=["stochastic_mean", "deterministic"],
        default="stochastic_mean",
        help="Use deterministic decoding for point-estimate metrics while keeping stochastic samples for uncertainty metrics.",
    )
    parser.add_argument("--checkpoint-eval-max-batches", type=int, default=32)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        help="Stop training early when the checkpoint metric has not improved for this many epochs.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum decrease required to reset early stopping patience.",
    )
    parser.add_argument("--max-train-batches", type=int, default=1000)
    parser.add_argument("--max-val-batches", type=int, default=200)
    parser.add_argument(
        "--train-sampling",
        choices=["uniform", "signal_weighted"],
        default="signal_weighted",
        help="Training window sampling policy.",
    )
    parser.add_argument(
        "--branch-sampling",
        choices=["uniform", "signal_weighted"],
        default="signal_weighted",
        help="Target-branch sampling policy inside each batch.",
    )
    parser.add_argument(
        "--branch-sampling-temperature",
        type=float,
        default=0.8,
        help="Lower values bias branch sampling more strongly toward active branches.",
    )
    parser.add_argument("--augment-noise-std", type=float, default=0.01)
    parser.add_argument("--augment-scale-min", type=float, default=0.97)
    parser.add_argument("--augment-scale-max", type=float, default=1.03)
    parser.add_argument(
        "--train-synthetic-mode",
        choices=["all", "base_only", "synthetic_only"],
        default="all",
        help="Which synthetic variants are visible in the training split.",
    )
    parser.add_argument(
        "--val-synthetic-mode",
        choices=["all", "base_only", "synthetic_only"],
        default="all",
        help="Which synthetic variants are visible in the validation split.",
    )
    parser.add_argument(
        "--test-synthetic-mode",
        choices=["all", "base_only", "synthetic_only"],
        default="all",
        help="Which synthetic variants are visible in the test split.",
    )
    parser.add_argument(
        "--checkpoint-metric",
        choices=[
            "loss",
            "flow_loss",
            "endpoint_loss",
            "sequence_total_loss",
            "physical_loss",
            "eval_rmse",
            "eval_ntae",
            "eval_med_rce",
            "eval_p90_rce",
            "eval_acc_at_100_neg",
            "eval_clea_neg",
            "eval_alarm_f1_neg",
        ],
        default="eval_ntae",
        help="Validation metric used to select the saved checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-eval-use-calibration",
        action="store_true",
        help="Use the validation split calibrator during eval-based checkpoint selection.",
    )
    parser.add_argument("--self-condition", action="store_true", help="Enable self-conditioning in the flow network.")
    parser.add_argument(
        "--self-condition-prob",
        type=float,
        default=0.5,
        help="Probability of using a detached self-conditioning signal during training.",
    )
    parser.add_argument(
        "--flow-path",
        choices=["linear", "cosine", "exp", "sigmoid"],
        default="linear",
        help="Probability path used to construct the conditional interpolation.",
    )
    parser.add_argument(
        "--flow-sigmoid-bias",
        type=float,
        default=0.0,
        help="Bias term for the sigmoid probability path.",
    )
    parser.add_argument(
        "--flow-sigmoid-scale",
        type=float,
        default=2.0,
        help="Scale term for the sigmoid probability path.",
    )
    parser.add_argument(
        "--multiscale-windows",
        default="5,10,20,50",
        help="Comma-separated window sizes for multi-scale cumulative supervision.",
    )
    parser.add_argument(
        "--endpoint-loss-type",
        choices=["mse", "huber"],
        default="mse",
        help="Endpoint reconstruction loss used on the decoded branch sequence.",
    )
    parser.add_argument(
        "--endpoint-huber-delta",
        type=float,
        default=1.0,
        help="Delta used when --endpoint-loss-type huber is selected.",
    )
    parser.add_argument(
        "--use-activity-gate",
        action="store_true",
        help="Enable an auxiliary activity gate that suppresses predictions on near-zero target segments.",
    )
    parser.add_argument("--disable-cue", action="store_true", help="Disable target-branch power cue for ablations.")
    args = parser.parse_args()
    has_single_csv = bool(args.data)
    has_manifest = bool(args.manifest)
    has_explicit_split = bool(args.train_data or args.val_data or args.test_data)
    has_split_config = bool(args.split_config)
    active_modes = sum([has_single_csv, has_manifest, has_explicit_split, has_split_config])
    if active_modes != 1:
        parser.error(
            "Choose exactly one input mode: --data, --manifest, --split-config, or "
            "the explicit trio --train-data/--val-data/--test-data."
        )

    if has_explicit_split and not (args.train_data and args.val_data and args.test_data):
        parser.error("--train-data, --val-data, and --test-data must be provided together.")
    if args.dataset_name and not args.manifest:
        parser.error("--dataset-name requires --manifest.")
    if args.list_datasets and not args.manifest:
        parser.error("--list-datasets requires --manifest.")
    return args


def load_manifest(manifest_path: Path) -> Dict[str, Dict[str, object]]:
    payload = json.loads(manifest_path.read_text())
    datasets = payload.get("datasets", [])
    result: Dict[str, Dict[str, object]] = {}
    for item in datasets:
        name = item.get("dataset_name")
        if isinstance(name, str):
            result[name] = item
    return result


def resolve_manifest_csv(manifest_path: Path, csv_path_value: str) -> Path:
    return resolve_manifest_artifact_path(manifest_path, csv_path_value)


def load_split_config(split_config_path: Path) -> Dict[str, object]:
    payload = json.loads(split_config_path.read_text())
    required = ["benchmark_name", "train_csv_path", "val_csv_path", "test_csv_path"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Split config is missing required fields: {', '.join(missing)}")
    return payload


def resolve_split_csv_path(split_config_path: Path, csv_path_value: str) -> Path:
    return resolve_split_artifact_path(split_config_path, csv_path_value)


def run_single_training(
    *,
    data_path: Optional[Path],
    split_paths: Optional[Dict[str, Path]],
    output_dir: Path,
    args: argparse.Namespace,
    dataset_name: str,
) -> Dict[str, float]:
    multiscale_windows = tuple(
        sorted(
            {
                int(part.strip())
                for part in str(args.multiscale_windows).split(",")
                if part.strip()
            }
        )
    )
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        flow_dim=args.flow_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_flow_blocks=args.num_flow_blocks,
        num_heads=args.num_heads,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        lambda_physical=args.lambda_physical,
        lambda_endpoint=args.lambda_endpoint,
        lambda_sequence_total=args.lambda_sequence_total,
        lambda_multiscale_total=args.lambda_multiscale_total,
        lambda_prefix_cumsum=args.lambda_prefix_cumsum,
        lambda_delta=args.lambda_delta,
        lambda_activity=args.lambda_activity,
        lambda_alarm=args.lambda_alarm,
        warmup_epochs_physical=args.warmup_epochs_physical,
        warmup_epochs_prefix=args.warmup_epochs_prefix,
        warmup_epochs_delta=args.warmup_epochs_delta,
        warmup_epochs_sequence_total=args.warmup_epochs_sequence_total,
        use_cue=not args.disable_cue,
        use_activity_gate=args.use_activity_gate,
        cue_dropout_prob=args.cue_dropout_prob,
        guidance_scale=args.guidance_scale,
        activity_threshold=args.activity_threshold,
        activity_sharpness=args.activity_sharpness,
        alarm_threshold=args.alarm_threshold,
        alarm_sharpness=args.alarm_sharpness,
        sample_steps=args.sample_steps,
        physical_sample_steps=args.physical_sample_steps,
        ode_solver=args.ode_solver,
        physical_ode_solver=args.physical_ode_solver,
        checkpoint_eval_max_batches=args.checkpoint_eval_max_batches,
        checkpoint_eval_use_calibration=args.checkpoint_eval_use_calibration,
        num_eval_samples=args.num_samples,
        eval_point_estimate_mode=args.eval_point_estimate_mode,
        train_sampling=args.train_sampling,
        branch_sampling=args.branch_sampling,
        branch_sampling_temperature=args.branch_sampling_temperature,
        augment_noise_std=args.augment_noise_std,
        augment_scale_min=args.augment_scale_min,
        augment_scale_max=args.augment_scale_max,
        train_synthetic_mode=args.train_synthetic_mode,
        val_synthetic_mode=args.val_synthetic_mode,
        test_synthetic_mode=args.test_synthetic_mode,
        checkpoint_metric=args.checkpoint_metric,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        output_dir=output_dir,
        self_condition=args.self_condition,
        self_condition_prob=args.self_condition_prob,
        flow_path=args.flow_path,
        flow_sigmoid_bias=args.flow_sigmoid_bias,
        flow_sigmoid_scale=args.flow_sigmoid_scale,
        multiscale_windows=multiscale_windows,
        endpoint_loss_type=args.endpoint_loss_type,
        endpoint_huber_delta=args.endpoint_huber_delta,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if split_paths is not None:
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
            train_synthetic_mode=config.train_synthetic_mode,
            val_synthetic_mode=config.val_synthetic_mode,
            test_synthetic_mode=config.test_synthetic_mode,
        )
    elif data_path is not None:
        train_loader, val_loader, test_loader, stats, num_branches = create_data_loaders(
            data_path=str(data_path),
            seq_len=config.seq_len,
            batch_size=config.batch_size,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            num_workers=config.num_workers,
            train_sampling=config.train_sampling,
            augment_noise_std=config.augment_noise_std,
            augment_scale_range=(config.augment_scale_min, config.augment_scale_max),
        )
    else:
        raise ValueError("Either data_path or split_paths must be provided.")

    trainer = build_trainer(config=config, num_branches=num_branches, stats=stats)
    if args.init_checkpoint:
        trainer.load_initial_checkpoint(Path(args.init_checkpoint).resolve())
    trainer.train(train_loader, val_loader)
    metrics = trainer.evaluate(test_loader, calibration_loader=val_loader)

    print(f"Final evaluation metrics for {dataset_name}:")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")

    run_config = {
        **config.__dict__,
        "dataset_name": dataset_name,
        "num_branches": num_branches,
    }
    if data_path is not None:
        run_config["data_path"] = str(data_path)
    if split_paths is not None:
        run_config["train_data_path"] = str(split_paths["train"])
        run_config["val_data_path"] = str(split_paths["val"])
        run_config["test_data_path"] = str(split_paths["test"])
    if args.init_checkpoint:
        run_config["init_checkpoint"] = str(Path(args.init_checkpoint).resolve())
    (config.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str))
    (config.output_dir / "normalization_stats.json").write_text(json.dumps(stats.to_dict(), indent=2))
    return metrics


def train_from_manifest(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest).resolve()
    dataset_map = load_manifest(manifest_path)
    if not dataset_map:
        raise ValueError(f"No datasets found in manifest: {manifest_path}")

    if args.list_datasets:
        for name, item in sorted(dataset_map.items()):
            print(
                f"{name} | branches={item.get('num_branches')} | "
                f"rows={item.get('dataset_rows_kept')}/{item.get('grid_rows_total')} | "
                f"range={item.get('grid_start')} -> {item.get('grid_end')}"
            )
        return

    selected_names: Sequence[str]
    if args.dataset_name:
        missing = [name for name in args.dataset_name if name not in dataset_map]
        if missing:
            raise ValueError(f"Datasets not found in manifest: {', '.join(missing)}")
        selected_names = args.dataset_name
    else:
        selected_names = sorted(dataset_map)

    base_output_dir = Path(args.output_dir)
    for dataset_name in selected_names:
        item = dataset_map[dataset_name]
        csv_path_value = item.get("csv_path")
        if not isinstance(csv_path_value, str):
            raise ValueError(f"Manifest entry for {dataset_name} does not contain a valid csv_path.")
        data_path = resolve_manifest_csv(manifest_path, csv_path_value)
        dataset_output_dir = base_output_dir / dataset_name
        print(f"Training dataset {dataset_name}")
        print(f"CSV: {data_path}")
        print(f"Output: {dataset_output_dir}")
        run_single_training(
            data_path=data_path,
            split_paths=None,
            output_dir=dataset_output_dir,
            args=args,
            dataset_name=dataset_name,
        )


def train_from_split_config(args: argparse.Namespace) -> None:
    split_config_path = Path(args.split_config).resolve()
    split_config = load_split_config(split_config_path)
    split_paths = {
        "train": resolve_split_csv_path(split_config_path, str(split_config["train_csv_path"])),
        "val": resolve_split_csv_path(split_config_path, str(split_config["val_csv_path"])),
        "test": resolve_split_csv_path(split_config_path, str(split_config["test_csv_path"])),
    }
    benchmark_name = str(split_config["benchmark_name"])
    output_dir = Path(args.output_dir)
    print(f"Training explicit split benchmark {benchmark_name}")
    for label, path in split_paths.items():
        print(f"{label}: {path}")
    run_single_training(
        data_path=None,
        split_paths=split_paths,
        output_dir=output_dir,
        args=args,
        dataset_name=benchmark_name,
    )


def main() -> None:
    args = parse_args()
    if args.manifest:
        train_from_manifest(args)
        return
    if args.split_config:
        train_from_split_config(args)
        return
    if args.train_data and args.val_data and args.test_data:
        output_dir = Path(args.output_dir)
        split_paths = {
            "train": Path(args.train_data).resolve(),
            "val": Path(args.val_data).resolve(),
            "test": Path(args.test_data).resolve(),
        }
        dataset_name = split_paths["train"].stem.replace(".train", "")
        run_single_training(
            data_path=None,
            split_paths=split_paths,
            output_dir=output_dir,
            args=args,
            dataset_name=dataset_name,
        )
        return

    data_path = Path(args.data).resolve()
    output_dir = Path(args.output_dir)
    dataset_name = data_path.stem
    run_single_training(
        data_path=data_path,
        split_paths=None,
        output_dir=output_dir,
        args=args,
        dataset_name=dataset_name,
    )


if __name__ == "__main__":
    main()
