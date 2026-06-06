import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from dataset import NormalizationStats, infer_branch_power_columns
from model import GFRCModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GFRC inference from a trained model directory.")
    parser.add_argument("--model-dir", required=True, help="Training output directory containing checkpoint and config.")
    parser.add_argument("--data", required=True, help="Path to the input CSV file.")
    parser.add_argument("--output-csv", required=True, help="Path for the inference output CSV.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=None, help="Override Monte Carlo sample count.")
    parser.add_argument("--sample-steps", type=int, default=None, help="Override ODE sampling steps.")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Disable stochastic sampling and dropout; useful for a single deterministic pass.",
    )
    return parser.parse_args()


def load_training_artifacts(model_dir: Path):
    run_config = json.loads((model_dir / "run_config.json").read_text())
    stats = NormalizationStats.from_dict(json.loads((model_dir / "normalization_stats.json").read_text()))
    checkpoint = torch.load(model_dir / "best_gfrc_model.pt", map_location="cpu", weights_only=False)
    return run_config, stats, checkpoint


def build_model(run_config: Dict[str, object], checkpoint: Dict[str, object], device):
    model = GFRCModel(
        hidden_dim=int(run_config["hidden_dim"]),
        flow_dim=int(run_config.get("flow_dim", 256)),
        num_encoder_layers=int(run_config.get("num_encoder_layers", 2)),
        num_flow_blocks=int(run_config.get("num_flow_blocks", 4)),
        num_heads=int(run_config.get("num_heads", 4)),
        dropout=float(run_config.get("dropout", 0.1)),
        self_condition=bool(run_config.get("self_condition", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def prepare_inputs(
    dataframe,
    seq_len: int,
    stats,
    expected_num_branches: int,
):
    branch_power_cols = infer_branch_power_columns(dataframe)
    if len(branch_power_cols) != expected_num_branches:
        raise ValueError(
            f"Input CSV has {len(branch_power_cols)} branch power columns, but the trained model expects {expected_num_branches}."
        )
    if len(dataframe) < seq_len:
        raise ValueError(f"Input CSV has {len(dataframe)} rows, shorter than seq_len={seq_len}.")

    total_current = dataframe["total_residual_current"].to_numpy(dtype=np.float32)
    total_power = dataframe["total_power"].to_numpy(dtype=np.float32)
    branch_powers = dataframe[branch_power_cols].to_numpy(dtype=np.float32).T

    total_current = stats.normalize_total_current(total_current)
    total_power = stats.normalize_total_power(total_power)
    branch_powers = stats.normalize_branch_powers(branch_powers)
    return total_current, total_power, branch_powers, branch_power_cols


def compute_window_starts(dataframe: pd.DataFrame, seq_len: int) -> List[int]:
    if "segment_id" not in dataframe.columns:
        return list(range(0, len(dataframe) - seq_len + 1))

    starts: List[int] = []
    current_segment = object()
    current_positions: List[int] = []
    for position, segment_id in enumerate(dataframe["segment_id"].tolist()):
        if current_positions and segment_id != current_segment:
            if len(current_positions) >= seq_len:
                starts.extend(range(current_positions[0], current_positions[-1] - seq_len + 2))
            current_positions = []
        current_segment = segment_id
        current_positions.append(position)
    if current_positions and len(current_positions) >= seq_len:
        starts.extend(range(current_positions[0], current_positions[-1] - seq_len + 2))
    return starts


def batched_window_starts(dataframe: pd.DataFrame, seq_len: int, batch_size: int) -> Sequence[Sequence[int]]:
    starts = compute_window_starts(dataframe, seq_len)
    return [starts[index : index + batch_size] for index in range(0, len(starts), batch_size)]


def infer_dataframe(
    *,
    dataframe,
    model,
    stats,
    seq_len: int,
    num_branches: int,
    batch_size: int,
    sample_steps: int,
    num_samples: int,
    stochastic: bool,
    enable_dropout: bool,
    ode_solver: str,
    guidance_scale: float,
    self_condition: bool,
    flow_path: str,
    flow_sigmoid_bias: float,
    flow_sigmoid_scale: float,
    device,
):
    total_current, total_power, branch_powers, _ = prepare_inputs(
        dataframe=dataframe,
        seq_len=seq_len,
        stats=stats,
        expected_num_branches=num_branches,
    )
    results: List[Dict[str, float]] = []

    for batch_starts in batched_window_starts(dataframe, seq_len, batch_size):
        x_r = np.stack([total_current[start : start + seq_len] for start in batch_starts], axis=0)
        x_p = np.stack([total_power[start : start + seq_len] for start in batch_starts], axis=0)
        branch_power_batch = np.stack(
            [branch_powers[:, start : start + seq_len] for start in batch_starts],
            axis=0,
        )

        x_r_tensor = torch.from_numpy(x_r).to(device)
        x_p_tensor = torch.from_numpy(x_p).to(device)
        branch_power_tensor = torch.from_numpy(branch_power_batch).to(device)

        branch_last_means: List[np.ndarray] = []
        branch_last_lowers: List[np.ndarray] = []
        branch_last_uppers: List[np.ndarray] = []

        for branch_index in range(num_branches):
            p_k = branch_power_tensor[:, branch_index, :]
            mean, _, samples = model.sample(
                x_r=x_r_tensor,
                x_p=x_p_tensor,
                p_k=p_k,
                num_steps=sample_steps,
                num_samples=num_samples,
                stochastic=stochastic,
                enable_dropout=enable_dropout,
                solver=ode_solver,
                guidance_scale=guidance_scale,
                self_condition=self_condition,
                flow_path=flow_path,
                flow_sigmoid_bias=flow_sigmoid_bias,
                flow_sigmoid_scale=flow_sigmoid_scale,
            )
            samples_np = samples[:, :, -1].detach().cpu().numpy().astype(np.float32)
            mean_np = mean[:, -1].detach().cpu().numpy().astype(np.float32)
            mean_np = mean_np * stats.branch_current_std[branch_index] + stats.branch_current_mean[branch_index]
            branch_last_means.append(mean_np)

            if num_samples > 1:
                lower_np = np.quantile(samples_np, 0.05, axis=0).astype(np.float32)
                upper_np = np.quantile(samples_np, 0.95, axis=0).astype(np.float32)
                lower_np = lower_np * stats.branch_current_std[branch_index] + stats.branch_current_mean[branch_index]
                upper_np = upper_np * stats.branch_current_std[branch_index] + stats.branch_current_mean[branch_index]
            else:
                lower_np = mean_np
                upper_np = mean_np
            branch_last_lowers.append(lower_np)
            branch_last_uppers.append(upper_np)

        for item_index, start in enumerate(batch_starts):
            row_index = start + seq_len - 1
            row: Dict[str, float] = {
                "window_start_index": float(start),
                "row_index": float(row_index),
            }
            total_pred = 0.0
            for branch_index in range(num_branches):
                mean_value = float(branch_last_means[branch_index][item_index])
                lower_value = float(branch_last_lowers[branch_index][item_index])
                upper_value = float(branch_last_uppers[branch_index][item_index])
                row[f"pred_branch_{branch_index + 1}_current_mean"] = mean_value
                row[f"pred_branch_{branch_index + 1}_current_p05"] = lower_value
                row[f"pred_branch_{branch_index + 1}_current_p95"] = upper_value
                total_pred += mean_value
            row["pred_total_residual_current"] = total_pred
            results.append(row)

    output = dataframe.iloc[seq_len - 1 :].reset_index(drop=True).copy()
    pred_frame = pd.DataFrame(results)
    output.insert(0, "window_start_index", pred_frame["window_start_index"].astype(int))
    output.insert(1, "row_index", pred_frame["row_index"].astype(int))
    for column in pred_frame.columns:
        if column in {"window_start_index", "row_index"}:
            continue
        output[column] = pred_frame[column]
    return output


def main() -> None:
    args = parse_args()

    model_dir = Path(args.model_dir).resolve()
    data_path = Path(args.data).resolve()
    output_csv = Path(args.output_csv).resolve()

    run_config, stats, checkpoint = load_training_artifacts(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(run_config, checkpoint, device=device)

    seq_len = int(run_config["seq_len"])
    sample_steps = int(args.sample_steps if args.sample_steps is not None else run_config.get("sample_steps", 80))
    num_samples = int(args.num_samples if args.num_samples is not None else run_config.get("num_eval_samples", 20))
    num_branches = int(run_config["num_branches"])
    ode_solver = str(run_config.get("ode_solver", "euler"))
    guidance_scale = float(run_config.get("guidance_scale", 1.0))
    self_condition = bool(run_config.get("self_condition", False))
    flow_path = str(run_config.get("flow_path", "linear"))
    flow_sigmoid_bias = float(run_config.get("flow_sigmoid_bias", 0.0))
    flow_sigmoid_scale = float(run_config.get("flow_sigmoid_scale", 2.0))

    dataframe = pd.read_csv(data_path)
    output = infer_dataframe(
        dataframe=dataframe,
        model=model,
        stats=stats,
        seq_len=seq_len,
        num_branches=num_branches,
        batch_size=args.batch_size,
        sample_steps=sample_steps,
        num_samples=num_samples,
        stochastic=not args.deterministic,
        enable_dropout=not args.deterministic and num_samples > 1,
        ode_solver=ode_solver,
        guidance_scale=guidance_scale,
        self_condition=self_condition,
        flow_path=flow_path,
        flow_sigmoid_bias=flow_sigmoid_bias,
        flow_sigmoid_scale=flow_sigmoid_scale,
        device=device,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False)
    print(f"Wrote inference output to {output_csv}")
    print(f"Rows: {len(output)} | Device: {device} | Samples per branch: {num_samples}")


if __name__ == "__main__":
    main()
