import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List
import sys

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
GFRC_ROOT = REPO_ROOT / "gfrc_full_impl"
if str(GFRC_ROOT) not in sys.path:
    sys.path.insert(0, str(GFRC_ROOT))

from baselines.shared.common import StreamingEvaluator, create_loaders_from_split_config  # noqa: E402
from experiments.common import LoadedGFRC, LoadedBaseline, load_model_from_run_dir, write_csv  # noqa: E402
from config import TrainConfig  # type: ignore  # noqa: E402
from dataset import create_explicit_split_data_loaders, build_window_indices, infer_branch_columns  # type: ignore  # noqa: E402
from baselines.prophet.train import build_features, standardize_features  # noqa: E402
from trainer import build_trainer  # type: ignore  # noqa: E402


METHOD_META: Dict[str, Dict[str, object]] = {
    "GFRC": {
        "run_dir": "training_runs/shanse001_rebuilt_round3_prefix005_weakaug",
        "kind": "gfrc",
        "stochastic": True,
    },
    "CNN-BiLSTM": {
        "run_dir": "training_runs/baselines_rerun_best_v1/cnn_bilstm",
        "kind": "baseline",
        "stochastic": False,
    },
    "Prophet": {
        "run_dir": "training_runs/baselines_rerun_best_v1/prophet",
        "kind": "prophet",
        "stochastic": False,
    },
    "TFT": {
        "run_dir": "training_runs/baselines_rerun_best_v1/tft",
        "kind": "baseline",
        "stochastic": False,
    },
    "TimeXer": {
        "run_dir": "training_runs/baselines_rerun_best_v1/timexer",
        "kind": "baseline",
        "stochastic": False,
    },
    "Deterministic-Physics": {
        "run_dir": "training_runs/baselines_rerun_best_v1/deterministic_physics",
        "kind": "baseline",
        "stochastic": False,
    },
    "CVAE": {
        "run_dir": "training_runs/baselines_rerun_best_v1/cvae",
        "kind": "baseline",
        "stochastic": True,
    },
    "Diffusion": {
        "run_dir": "training_runs/baselines_rerun_best_v1/diffusion",
        "kind": "baseline",
        "stochastic": True,
    },
    "cINN": {
        "run_dir": "training_runs/baselines_rerun_best_v1/cinn",
        "kind": "baseline",
        "stochastic": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repeat evaluation for trained runs without retraining.")
    parser.add_argument("--split-config", default="data_preprocessing/output/benchmarks/shanse001_rebuilt_entity_split/split_config.json")
    parser.add_argument("--output-dir", default="experiments/output_repeat_eval5")
    parser.add_argument("--spec-json")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=20260511)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--method", action="append", help="Optional subset of methods to evaluate.")
    parser.add_argument("--num-samples-override", type=int)
    parser.add_argument("--gfrc-sample-steps-override", type=int)
    parser.add_argument("--gfrc-ode-solver-override", choices=["euler", "heun"])
    parser.add_argument("--gfrc-guidance-scale-override", type=float)
    parser.add_argument(
        "--gfrc-point-estimate-mode-override",
        choices=["stochastic_mean", "deterministic"],
        help="Override the point-estimate evaluation mode for GFRC.",
    )
    parser.add_argument("--gfrc-run-dir", help="Override the GFRC run directory used for evaluation.")
    parser.add_argument("--alarm-threshold", type=float, default=0.05)
    parser.add_argument("--disable-gfrc-calibration", action="store_true")
    parser.add_argument("--gfrc-eval-batch-size", type=int)
    parser.add_argument("--gfrc-calibration-batch-size", type=int)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sanitize_method_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def load_method_meta(spec_json: str | None) -> Dict[str, Dict[str, object]]:
    if spec_json is None:
        return METHOD_META
    payload = json.loads(Path(spec_json).read_text())
    methods = payload.get("methods", {})
    if not isinstance(methods, dict) or not methods:
        raise ValueError("Spec JSON must contain a non-empty 'methods' object.")
    return methods


def load_gfrc_split_config_local(split_config_path: Path) -> Dict[str, object]:
    payload = json.loads(split_config_path.read_text())
    required = ["benchmark_name", "train_csv_path", "val_csv_path", "test_csv_path"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Split config is missing required fields: {', '.join(missing)}")
    return payload


def resolve_split_artifact_path_local(split_config_path: Path, path_value: str) -> Path:
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

    seen = set()
    ordered = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(candidate)

    for candidate in ordered:
        if candidate.exists():
            return candidate.resolve()
    return ordered[-1].resolve()


def evaluate_gfrc(
    model: LoadedGFRC,
    split_config_path: str,
    num_samples_override: int | None,
    sample_steps_override: int | None,
    ode_solver_override: str | None,
    guidance_scale_override: float | None,
    point_estimate_mode_override: str | None,
    alarm_threshold: float,
    disable_calibration: bool,
    eval_batch_size_override: int | None,
    calibration_batch_size_override: int | None,
    eval_output_dir: Path,
) -> Dict[str, float]:
    source_checkpoint_path = model.run_dir / "best_gfrc_model.pt"
    checkpoint = torch.load(source_checkpoint_path, map_location="cpu", weights_only=False)
    config = TrainConfig(**checkpoint["config"])
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir = eval_output_dir
    eval_checkpoint_path = eval_output_dir / "best_gfrc_model.pt"
    if not eval_checkpoint_path.exists():
        shutil.copy2(source_checkpoint_path, eval_checkpoint_path)
    config.alarm_threshold = alarm_threshold
    if num_samples_override is not None:
        config.num_eval_samples = num_samples_override
    if sample_steps_override is not None:
        config.sample_steps = sample_steps_override
    if ode_solver_override is not None:
        config.ode_solver = ode_solver_override
    if guidance_scale_override is not None:
        config.guidance_scale = guidance_scale_override
    if point_estimate_mode_override is not None:
        config.eval_point_estimate_mode = point_estimate_mode_override
    eval_batch_size = eval_batch_size_override or config.batch_size
    calibration_batch_size = calibration_batch_size_override or eval_batch_size

    split_config_path_obj = Path(split_config_path).resolve()
    split_config = load_gfrc_split_config_local(split_config_path_obj)
    split_paths = {
        "train": resolve_split_artifact_path_local(split_config_path_obj, str(split_config["train_csv_path"])),
        "val": resolve_split_artifact_path_local(split_config_path_obj, str(split_config["val_csv_path"])),
        "test": resolve_split_artifact_path_local(split_config_path_obj, str(split_config["test_csv_path"])),
    }
    train_loader, val_loader, test_loader, stats, num_branches = create_explicit_split_data_loaders(
        train_data_path=str(split_paths["train"]),
        val_data_path=str(split_paths["val"]),
        test_data_path=str(split_paths["test"]),
        seq_len=config.seq_len,
        batch_size=eval_batch_size,
        num_workers=config.num_workers,
        train_sampling=config.train_sampling,
        augment_noise_std=config.augment_noise_std,
        augment_scale_range=(config.augment_scale_min, config.augment_scale_max),
    )
    del train_loader
    calibration_loader = val_loader
    if not disable_calibration and calibration_batch_size != eval_batch_size:
        _, calibration_loader, _, _, _ = create_explicit_split_data_loaders(
            train_data_path=str(split_paths["train"]),
            val_data_path=str(split_paths["val"]),
            test_data_path=str(split_paths["test"]),
            seq_len=config.seq_len,
            batch_size=calibration_batch_size,
            num_workers=config.num_workers,
            train_sampling=config.train_sampling,
            augment_noise_std=config.augment_noise_std,
            augment_scale_range=(config.augment_scale_min, config.augment_scale_max),
        )
    trainer = build_trainer(config=config, num_branches=num_branches, stats=stats)
    calibration_loader = None if disable_calibration else calibration_loader
    return trainer.evaluate(test_loader, calibration_loader=calibration_loader)


def evaluate_baseline(
    model: LoadedBaseline,
    split_config_path: str,
    batch_size: int | None,
    num_samples_override: int | None,
    alarm_threshold: float,
) -> Dict[str, float]:
    run_config = read_json(model.run_dir / "run_config.json")
    _, loaders = create_loaders_from_split_config(
        split_config_path=split_config_path,
        seq_len=int(run_config.get("seq_len", 100)),
        batch_size=batch_size or int(run_config.get("batch_size", 64)),
        num_workers=0,
    )
    _, _, test_loader, _, num_branches = loaders
    model.num_branches = num_branches
    if num_samples_override is not None:
        model.num_samples = num_samples_override
    evaluator = StreamingEvaluator(alpha=0.1, alarm_threshold=alarm_threshold)
    for batch in test_loader:
        x_r = batch["x_r"].to(model.device)
        x_p = batch["x_p"].to(model.device)
        branch_powers = batch["branch_powers"].to(model.device)
        branch_currents = batch["branch_currents"].to(model.device)
        for branch_index in range(num_branches):
            target_p = branch_powers[:, branch_index, :]
            if not model.use_cue:
                target_p = torch.zeros_like(target_p)
            target_y = branch_currents[:, branch_index, :]
            mean, samples = model.model.sample(x_r, x_p, target_p, num_samples=model.num_samples)
            evaluator.update(
                model.denormalize_branch_current(branch_index, mean.detach().cpu()),
                model.denormalize_branch_current(branch_index, target_y.detach().cpu()),
                samples=model.denormalize_branch_current(branch_index, samples.detach().cpu()),
            )
    return evaluator.finalize()


def evaluate_prophet(
    run_dir: Path,
    split_config_path: str,
    num_samples_override: int | None,
    alarm_threshold: float,
) -> Dict[str, float]:
    run_config = read_json(run_dir / "run_config.json")
    split_config_path_obj = Path(split_config_path).resolve()
    split_config = load_gfrc_split_config_local(split_config_path_obj)
    train_frame = pd.read_csv(resolve_split_artifact_path_local(split_config_path_obj, str(split_config["train_csv_path"])))
    test_frame = pd.read_csv(resolve_split_artifact_path_local(split_config_path_obj, str(split_config["test_csv_path"])))
    branch_power_cols, branch_current_cols = infer_branch_columns(train_frame)
    branch_limit = int(run_config.get("num_branches", len(branch_power_cols)))
    fourier_order = int(run_config.get("fourier_order", 5))
    seq_len = int(run_config.get("seq_len", 100))
    num_samples = num_samples_override or int(run_config.get("num_samples", 20))
    coefficients = read_json(run_dir / "coefficients.json")
    evaluator = StreamingEvaluator(alpha=0.1, alarm_threshold=alarm_threshold)
    test_starts = build_window_indices(test_frame, seq_len)

    for branch_index in range(branch_limit):
        power_col = branch_power_cols[branch_index]
        current_col = branch_current_cols[branch_index]
        train_x = build_features(train_frame, power_col, fourier_order)
        test_x = build_features(test_frame, power_col, fourier_order)
        train_x, scaled_arrays, _, _ = standardize_features(train_x, test_x)
        test_x = scaled_arrays[0]
        weights = np.asarray(coefficients[current_col], dtype=np.float32)
        pred_values = test_x @ weights
        target_values = test_frame[current_col].to_numpy(dtype=np.float32)

        for start in test_starts:
            stop = start + seq_len
            mean = torch.from_numpy(pred_values[start:stop]).unsqueeze(0)
            target = torch.from_numpy(target_values[start:stop]).unsqueeze(0)
            samples = mean.unsqueeze(0).repeat(num_samples, 1, 1)
            evaluator.update(mean, target, samples=samples)

    return evaluator.finalize()


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text())


def main() -> None:
    args = parse_args()
    root_dir = REPO_ROOT.resolve()
    output_dir = (root_dir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    method_meta = load_method_meta(args.spec_json)
    if args.gfrc_run_dir is not None and "GFRC" in method_meta:
        method_meta["GFRC"] = {**method_meta["GFRC"], "run_dir": args.gfrc_run_dir}
    selected_methods = args.method if args.method else list(method_meta.keys())
    per_repeat_rows: List[Dict[str, object]] = []

    for method in selected_methods:
        meta = method_meta[method]
        run_dir = (root_dir / str(meta["run_dir"])).resolve()
        method_slug = sanitize_method_name(method)
        method_out_dir = output_dir / method_slug
        method_out_dir.mkdir(parents=True, exist_ok=True)

        for repeat_idx in range(args.repeats):
            seed = args.base_seed + repeat_idx
            set_seed(seed)
            repeat_eval_dir = method_out_dir / f"repeat_{repeat_idx + 1}_gfrc_eval"
            if meta["kind"] == "gfrc":
                model = load_model_from_run_dir(run_dir, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                assert isinstance(model, LoadedGFRC)
                metrics = evaluate_gfrc(
                    model,
                    args.split_config,
                    args.num_samples_override,
                    args.gfrc_sample_steps_override,
                    args.gfrc_ode_solver_override,
                    args.gfrc_guidance_scale_override,
                    args.gfrc_point_estimate_mode_override,
                    args.alarm_threshold,
                    args.disable_gfrc_calibration,
                    args.gfrc_eval_batch_size,
                    args.gfrc_calibration_batch_size,
                    repeat_eval_dir,
                )
            elif meta["kind"] == "baseline":
                model = load_model_from_run_dir(run_dir, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                assert isinstance(model, LoadedBaseline)
                metrics = evaluate_baseline(
                    model,
                    args.split_config,
                    args.batch_size,
                    args.num_samples_override,
                    args.alarm_threshold,
                )
            elif meta["kind"] == "prophet":
                metrics = evaluate_prophet(run_dir, args.split_config, args.num_samples_override, args.alarm_threshold)
            else:
                raise ValueError(f"Unsupported kind: {meta['kind']}")

            payload = {
                "method": method,
                "repeat_index": repeat_idx + 1,
                "seed": seed,
                "stochastic_eval": bool(meta["stochastic"]),
                **metrics,
            }
            per_repeat_rows.append(payload)
            (method_out_dir / f"repeat_{repeat_idx + 1}.json").write_text(json.dumps(payload, indent=2))

    write_csv(
        output_dir / "per_repeat_metrics.csv",
        per_repeat_rows,
        fieldnames=[
            "method",
            "repeat_index",
            "seed",
            "stochastic_eval",
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
            "alarm_precision",
            "alarm_recall",
            "alarm_f1",
            "alarm_accuracy",
            "alarm_far",
            "time_per_call_seconds",
        ],
    )


if __name__ == "__main__":
    main()
