from typing import Dict

import torch


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((pred - target) ** 2))


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def ntae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_total = pred.sum(dim=-1)
    target_total = target.sum(dim=-1)
    numerator = torch.sum(torch.abs(pred_total - target_total))
    denominator = torch.sum(torch.abs(target_total)).clamp_min(1e-8)
    return numerator / denominator


def clea(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_total = pred.sum(dim=-1)
    target_total = target.sum(dim=-1)
    per_sequence = 1.0 - torch.abs(pred_total - target_total) / torch.abs(target_total).clamp_min(1e-8)
    return per_sequence.mean()


def relative_cumulative_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_total = pred.sum(dim=-1)
    target_total = target.sum(dim=-1)
    return torch.abs(pred_total - target_total) / torch.abs(target_total).clamp_min(1e-8)


def cumulative_distribution_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    per_sequence = relative_cumulative_error(pred, target)
    med_rce = torch.quantile(per_sequence, 0.5)
    p90_rce = torch.quantile(per_sequence, 0.9)
    acc_at_100 = (per_sequence <= 1.0).float().mean()
    return {
        "med_rce": float(med_rce.item()),
        "p90_rce": float(p90_rce.item()),
        "acc_at_100": float(acc_at_100.item()),
    }


def crps_from_samples(samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # samples: (num_samples, batch, seq_len), target: (batch, seq_len)
    term_1 = torch.mean(torch.abs(samples - target.unsqueeze(0)))
    pairwise = torch.abs(samples.unsqueeze(0) - samples.unsqueeze(1))
    term_2 = 0.5 * torch.mean(pairwise)
    return term_1 - term_2


def coverage_and_width(
    samples: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.1,
) -> Dict[str, float]:
    lower = torch.quantile(samples, alpha / 2.0, dim=0)
    upper = torch.quantile(samples, 1.0 - alpha / 2.0, dim=0)
    covered = ((target >= lower) & (target <= upper)).float().mean()
    width = (upper - lower).mean()
    return {
        f"picp_{int((1.0 - alpha) * 100)}": float(covered.item()),
        f"mpiw_{int((1.0 - alpha) * 100)}": float(width.item()),
    }


def compute_point_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    metrics = {
        "mse": float(mse(pred, target).item()),
        "mae": float(mae(pred, target).item()),
        "rmse": float(rmse(pred, target).item()),
        "ntae": float(ntae(pred, target).item()),
        "clea": float(clea(pred, target).item()),
    }
    metrics.update(cumulative_distribution_metrics(pred, target))
    return metrics
