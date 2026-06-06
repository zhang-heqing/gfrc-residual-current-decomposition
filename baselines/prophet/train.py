import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.shared.common import DEFAULT_SPLIT_CONFIG, StreamingEvaluator, load_split_config, save_json
from data_preprocessing.path_utils import resolve_split_artifact_path
from gfrc_full_impl.dataset import build_window_indices, infer_branch_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Prophet-style additive baseline.")
    parser.add_argument("--split-config", default=str(DEFAULT_SPLIT_CONFIG))
    parser.add_argument("--output-dir", default="training_runs/baselines/prophet")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=100)
    parser.add_argument("--fourier-order", type=int, default=5)
    parser.add_argument("--ridge-alpha", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1, 1.0, 10.0])
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--limit-branches", type=int)
    return parser.parse_args()


def build_features(dataframe: pd.DataFrame, branch_power_col: str, fourier_order: int) -> np.ndarray:
    timestamp = pd.to_datetime(dataframe["timestamp"])
    minute_of_day = timestamp.dt.hour.to_numpy() * 60 + timestamp.dt.minute.to_numpy()
    day_of_week = timestamp.dt.dayofweek.to_numpy()
    trend = np.linspace(0.0, 1.0, len(dataframe), dtype=np.float32)

    feature_columns = [
        np.ones(len(dataframe), dtype=np.float32),
        trend,
        dataframe["total_residual_current"].to_numpy(dtype=np.float32),
        dataframe["total_power"].to_numpy(dtype=np.float32),
        dataframe[branch_power_col].to_numpy(dtype=np.float32),
    ]
    feature_columns.append(feature_columns[2] * feature_columns[4])
    feature_columns.append(feature_columns[3] * feature_columns[4])

    for order in range(1, fourier_order + 1):
        phase_day = 2.0 * np.pi * order * minute_of_day / 1440.0
        phase_week = 2.0 * np.pi * order * day_of_week / 7.0
        feature_columns.append(np.sin(phase_day).astype(np.float32))
        feature_columns.append(np.cos(phase_day).astype(np.float32))
        feature_columns.append(np.sin(phase_week).astype(np.float32))
        feature_columns.append(np.cos(phase_week).astype(np.float32))

    return np.stack(feature_columns, axis=1)


def standardize_features(
    train_x: np.ndarray,
    *other_arrays: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    mean[0] = 0.0
    std[0] = 1.0
    std = np.where(std < 1e-6, 1.0, std)
    train_scaled = (train_x - mean) / std
    other_scaled = [((array - mean) / std) for array in other_arrays]
    return train_scaled, other_scaled, mean, std


def fit_ridge(train_x: np.ndarray, train_y: np.ndarray, alpha: float) -> np.ndarray:
    regularizer = np.eye(train_x.shape[1], dtype=np.float32) * alpha
    regularizer[0, 0] = 0.0
    lhs = train_x.T @ train_x + regularizer
    rhs = train_x.T @ train_y
    return np.linalg.pinv(lhs) @ rhs


def evaluate_row_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    split_config_path = Path(args.split_config).resolve()
    split_config = load_split_config(args.split_config)
    train_frame = pd.read_csv(resolve_split_artifact_path(split_config_path, str(split_config["train_csv_path"])))
    val_frame = pd.read_csv(resolve_split_artifact_path(split_config_path, str(split_config["val_csv_path"])))
    test_frame = pd.read_csv(resolve_split_artifact_path(split_config_path, str(split_config["test_csv_path"])))

    branch_power_cols, branch_current_cols = infer_branch_columns(train_frame)
    branch_limit = len(branch_power_cols) if args.limit_branches is None else min(args.limit_branches, len(branch_power_cols))

    val_predictions = np.zeros((len(val_frame), branch_limit), dtype=np.float32)
    test_predictions = np.zeros((len(test_frame), branch_limit), dtype=np.float32)
    best_alphas = {}
    coefficients = {}

    for branch_index in range(branch_limit):
        power_col = branch_power_cols[branch_index]
        current_col = branch_current_cols[branch_index]

        train_x = build_features(train_frame, power_col, args.fourier_order)
        val_x = build_features(val_frame, power_col, args.fourier_order)
        test_x = build_features(test_frame, power_col, args.fourier_order)
        train_x, scaled_arrays, _, _ = standardize_features(train_x, val_x, test_x)
        val_x, test_x = scaled_arrays

        train_y = train_frame[current_col].to_numpy(dtype=np.float32)
        val_y = val_frame[current_col].to_numpy(dtype=np.float32)
        test_y = test_frame[current_col].to_numpy(dtype=np.float32)

        best_alpha = None
        best_weights = None
        best_rmse = float("inf")
        for alpha in args.ridge_alpha:
            weights = fit_ridge(train_x, train_y, alpha)
            pred = val_x @ weights
            rmse = evaluate_row_rmse(pred, val_y)
            if rmse < best_rmse:
                best_rmse = rmse
                best_alpha = alpha
                best_weights = weights

        if best_weights is None or best_alpha is None:
            raise RuntimeError(f"Failed to fit branch {branch_index + 1}")

        val_predictions[:, branch_index] = val_x @ best_weights
        test_predictions[:, branch_index] = test_x @ best_weights
        best_alphas[current_col] = float(best_alpha)
        coefficients[current_col] = best_weights.tolist()

    evaluator = StreamingEvaluator(alpha=0.1)
    test_starts = build_window_indices(test_frame, args.seq_len)
    for branch_index in range(branch_limit):
        target_values = test_frame[branch_current_cols[branch_index]].to_numpy(dtype=np.float32)
        pred_values = test_predictions[:, branch_index]

        for start in test_starts:
            stop = start + args.seq_len
            mean = torch.from_numpy(pred_values[start:stop]).unsqueeze(0)
            target = torch.from_numpy(target_values[start:stop]).unsqueeze(0)
            samples = mean.unsqueeze(0).repeat(args.num_samples, 1, 1)
            evaluator.update(mean, target, samples=samples)

    metrics = evaluator.finalize()
    save_json(output_dir / "evaluation_metrics.json", metrics)
    save_json(
        output_dir / "run_config.json",
        {
            **vars(args),
            "benchmark_name": split_config["benchmark_name"],
            "num_branches": branch_limit,
            "best_alphas": best_alphas,
        },
    )
    save_json(output_dir / "coefficients.json", coefficients)

    print("Final test metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
