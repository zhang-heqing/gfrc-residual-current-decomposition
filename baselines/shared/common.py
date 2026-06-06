import json
import math
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from baselines.shared.bootstrap import REPO_ROOT  # noqa: F401
from data_preprocessing.path_utils import resolve_split_artifact_path
from gfrc_full_impl.dataset import create_explicit_split_data_loaders


DEFAULT_SPLIT_CONFIG = (
    REPO_ROOT
    / "data_preprocessing"
    / "output"
    / "benchmarks"
    / "shanse001_rebuilt_entity_split"
    / "split_config.json"
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def load_split_config(split_config_path: str) -> Dict[str, object]:
    payload = json.loads(Path(split_config_path).read_text())
    required = ["benchmark_name", "train_csv_path", "val_csv_path", "test_csv_path"]
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"Split config missing fields: {', '.join(missing)}")
    return payload


def create_loaders_from_split_config(
    split_config_path: str,
    *,
    seq_len: int,
    batch_size: int,
    num_workers: int,
    train_sampling: str = "signal_weighted",
):
    split_config = load_split_config(split_config_path)
    split_config_file = Path(split_config_path).resolve()
    train_path = str(resolve_split_artifact_path(split_config_file, str(split_config["train_csv_path"])))
    val_path = str(resolve_split_artifact_path(split_config_file, str(split_config["val_csv_path"])))
    test_path = str(resolve_split_artifact_path(split_config_file, str(split_config["test_csv_path"])))
    loaders = create_explicit_split_data_loaders(
        train_data_path=train_path,
        val_data_path=val_path,
        test_data_path=test_path,
        seq_len=seq_len,
        batch_size=batch_size,
        num_workers=num_workers,
        train_sampling=train_sampling,
    )
    return split_config, loaders


def sample_target_branch(
    branch_powers: torch.Tensor,
    branch_currents: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, num_branches, _ = branch_powers.shape
    branch_idx = torch.randint(0, num_branches, (batch_size,), device=branch_powers.device)
    gather_idx = branch_idx[:, None, None].expand(-1, 1, branch_powers.size(-1))
    target_p = branch_powers.gather(1, gather_idx).squeeze(1)
    target_y = branch_currents.gather(1, gather_idx).squeeze(1)
    return branch_idx, target_p, target_y


class StreamingEvaluator:
    def __init__(self, alpha: float = 0.1, alarm_threshold: float | None = None) -> None:
        self.alpha = alpha
        self.alarm_threshold = alarm_threshold
        self.sum_sq_err = 0.0
        self.sum_abs_err = 0.0
        self.num_values = 0
        self.sum_abs_total_err = 0.0
        self.sum_abs_total_target = 0.0
        self.sum_clea = 0.0
        self.rce_values = []
        self.num_sequences = 0
        self.sum_crps = 0.0
        self.covered_values = 0.0
        self.total_interval_width = 0.0
        self.interval_values = 0
        self.batch_times = []
        self.alarm_tp = 0
        self.alarm_fp = 0
        self.alarm_fn = 0
        self.alarm_tn = 0

    def update(
        self,
        mean: torch.Tensor,
        target: torch.Tensor,
        samples: Optional[torch.Tensor] = None,
        elapsed_seconds: Optional[float] = None,
    ) -> None:
        mean = mean.detach().cpu()
        target = target.detach().cpu()
        diff = mean - target

        self.sum_sq_err += float((diff ** 2).sum().item())
        self.sum_abs_err += float(diff.abs().sum().item())
        self.num_values += int(diff.numel())

        mean_total = mean.sum(dim=-1)
        target_total = target.sum(dim=-1)
        total_err = (mean_total - target_total).abs()
        target_abs = target_total.abs().clamp_min(1e-8)
        rce = total_err / target_abs
        self.sum_abs_total_err += float(total_err.sum().item())
        self.sum_abs_total_target += float(target_abs.sum().item())
        self.sum_clea += float((1.0 - total_err / target_abs).sum().item())
        self.rce_values.extend(rce.reshape(-1).tolist())
        self.num_sequences += int(target_total.numel())

        sample_tensor = samples.detach().cpu() if samples is not None else mean.unsqueeze(0)
        term_1 = torch.mean(torch.abs(sample_tensor - target.unsqueeze(0)), dim=0)
        pairwise = torch.abs(sample_tensor.unsqueeze(0) - sample_tensor.unsqueeze(1))
        term_2 = 0.5 * torch.mean(pairwise, dim=(0, 1))
        crps = term_1 - term_2
        self.sum_crps += float(crps.sum().item())

        lower = torch.quantile(sample_tensor, self.alpha / 2.0, dim=0)
        upper = torch.quantile(sample_tensor, 1.0 - self.alpha / 2.0, dim=0)
        covered = ((target >= lower) & (target <= upper)).float()
        self.covered_values += float(covered.sum().item())
        self.total_interval_width += float((upper - lower).sum().item())
        self.interval_values += int(target.numel())

        if self.alarm_threshold is not None:
            pred_alarm = mean.abs().amax(dim=-1) >= self.alarm_threshold
            target_alarm = target.abs().amax(dim=-1) >= self.alarm_threshold
            self.alarm_tp += int(torch.logical_and(pred_alarm, target_alarm).sum().item())
            self.alarm_fp += int(torch.logical_and(pred_alarm, ~target_alarm).sum().item())
            self.alarm_fn += int(torch.logical_and(~pred_alarm, target_alarm).sum().item())
            self.alarm_tn += int(torch.logical_and(~pred_alarm, ~target_alarm).sum().item())

        if elapsed_seconds is not None:
            self.batch_times.append(float(elapsed_seconds))

    def finalize(self) -> Dict[str, float]:
        rmse = math.sqrt(self.sum_sq_err / max(self.num_values, 1))
        mae = self.sum_abs_err / max(self.num_values, 1)
        ntae = self.sum_abs_total_err / max(self.sum_abs_total_target, 1e-8)
        clea = self.sum_clea / max(self.num_sequences, 1)
        rce_array = np.asarray(self.rce_values, dtype=np.float64) if self.rce_values else np.asarray([float("nan")])
        med_rce = float(np.nanmedian(rce_array))
        p90_rce = float(np.nanpercentile(rce_array, 90))
        acc_at_100 = float(np.nanmean((rce_array <= 1.0).astype(np.float64)))
        crps = self.sum_crps / max(self.num_values, 1)
        coverage = self.covered_values / max(self.interval_values, 1)
        width = self.total_interval_width / max(self.interval_values, 1)
        precision = float(self.alarm_tp / max(self.alarm_tp + self.alarm_fp, 1))
        recall = float(self.alarm_tp / max(self.alarm_tp + self.alarm_fn, 1))
        return {
            "mse": self.sum_sq_err / max(self.num_values, 1),
            "mae": mae,
            "rmse": rmse,
            "ntae": ntae,
            "clea": clea,
            "med_rce": med_rce,
            "p90_rce": p90_rce,
            "acc_at_100": acc_at_100,
            "crps": crps,
            f"picp_{int((1.0 - self.alpha) * 100)}": coverage,
            f"mpiw_{int((1.0 - self.alpha) * 100)}": width,
            "time_per_call_seconds": float(np.mean(self.batch_times)) if self.batch_times else 0.0,
            **(
                {
                    "alarm_precision": precision,
                    "alarm_recall": recall,
                    "alarm_f1": float(2.0 * precision * recall / max(precision + recall, 1e-8)),
                    "alarm_accuracy": float(
                        (self.alarm_tp + self.alarm_tn)
                        / max(self.alarm_tp + self.alarm_tn + self.alarm_fp + self.alarm_fn, 1)
                    ),
                    "alarm_far": float(self.alarm_fp / max(self.alarm_fp + self.alarm_tn, 1)),
                }
                if self.alarm_threshold is not None
                else {}
            ),
        }
