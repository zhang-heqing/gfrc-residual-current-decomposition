import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Tuple

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.shared.common import StreamingEvaluator, create_loaders_from_split_config, load_split_config  # noqa: E402
from baselines.shared.models import build_model  # noqa: E402
from gfrc_full_impl.config import TrainConfig  # noqa: E402
from gfrc_full_impl.dataset import NormalizationStats  # noqa: E402
from gfrc_full_impl.model import GFRCModel  # noqa: E402


DEFAULT_SPLIT_CONFIG = (
    REPO_ROOT
    / "data_preprocessing"
    / "output"
    / "benchmarks"
    / "shanse001_rebuilt_entity_split"
    / "split_config.json"
)


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text())


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> None:
    ensure_parent(path)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def infer_param_count_from_checkpoint(checkpoint_path: Path, state_key: str) -> int:
    if not checkpoint_path.exists():
        return 0
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get(state_key, {})
    return int(sum(tensor.numel() for tensor in state_dict.values()))


def project_branch_predictions_to_total(
    predicted_branches: torch.Tensor,
    total_current: torch.Tensor,
    branch_powers: torch.Tensor | None = None,
    mode: str = "power_weighted",
    alpha: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    if predicted_branches.dim() != 3:
        raise ValueError(f"predicted_branches must have shape [B, K, T], got {tuple(predicted_branches.shape)}")
    if total_current.dim() != 2:
        raise ValueError(f"total_current must have shape [B, T], got {tuple(total_current.shape)}")

    pred_sum = predicted_branches.sum(dim=1)
    residual = total_current - pred_sum

    if mode == "uniform":
        weights = torch.full_like(predicted_branches, 1.0 / predicted_branches.size(1))
    elif mode == "magnitude":
        weights = torch.abs(predicted_branches)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(eps)
    elif mode == "power_weighted":
        if branch_powers is not None:
            weights = torch.abs(branch_powers)
        else:
            weights = torch.abs(predicted_branches)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(eps)
    else:
        raise ValueError(f"Unsupported projection mode: {mode}")

    projected = predicted_branches + weights * residual.unsqueeze(1)
    return predicted_branches + alpha * (projected - predicted_branches)


def compute_alarm_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
) -> Dict[str, float]:
    pred_alarm = pred.abs().amax(dim=-1) >= threshold
    target_alarm = target.abs().amax(dim=-1) >= threshold

    tp = torch.logical_and(pred_alarm, target_alarm).sum().item()
    fp = torch.logical_and(pred_alarm, ~target_alarm).sum().item()
    fn = torch.logical_and(~pred_alarm, target_alarm).sum().item()
    tn = torch.logical_and(~pred_alarm, ~target_alarm).sum().item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    far = fp / max(fp + tn, 1)

    return {
        "alarm_precision": float(precision),
        "alarm_recall": float(recall),
        "alarm_f1": float(f1),
        "alarm_accuracy": float(accuracy),
        "alarm_far": float(far),
    }


class LoadedModel:
    def __init__(self, name: str, run_dir: Path, device: torch.device, sample_steps: int, num_samples: int, use_cue: bool) -> None:
        self.name = name
        self.run_dir = run_dir
        self.device = device
        self.sample_steps = sample_steps
        self.num_samples = num_samples
        self.use_cue = use_cue
        self.num_branches = 0
        self.stats: NormalizationStats | None = None

    def sample(self, x_r: torch.Tensor, x_p: torch.Tensor, p_k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def predict_point(self, x_r: torch.Tensor, x_p: torch.Tensor, p_k: torch.Tensor) -> torch.Tensor:
        mean, _ = self.sample(x_r, x_p, p_k)
        return mean

    def denormalize_branch_currents(self, value: torch.Tensor) -> torch.Tensor:
        if self.stats is None:
            return value
        std = torch.as_tensor(self.stats.branch_current_std, dtype=value.dtype, device=value.device)
        mean = torch.as_tensor(self.stats.branch_current_mean, dtype=value.dtype, device=value.device)
        if value.dim() == 2:
            return value * std[0] + mean[0]
        if value.dim() == 3:
            return value * std.view(1, -1, 1) + mean.view(1, -1, 1)
        return value

    def denormalize_branch_current(self, branch_index: int, value: torch.Tensor) -> torch.Tensor:
        if self.stats is None:
            return value
        std = float(self.stats.branch_current_std[branch_index])
        mean = float(self.stats.branch_current_mean[branch_index])
        return value * std + mean

    def denormalize_total_current(self, value: torch.Tensor) -> torch.Tensor:
        if self.stats is None:
            return value
        return value * float(self.stats.total_current_std) + float(self.stats.total_current_mean)

    def denormalize_total_power(self, value: torch.Tensor) -> torch.Tensor:
        if self.stats is None:
            return value
        return value * float(self.stats.total_power_std) + float(self.stats.total_power_mean)

    def denormalize_branch_power(self, branch_index: int, value: torch.Tensor) -> torch.Tensor:
        if self.stats is None:
            return value
        std = float(self.stats.branch_power_std[branch_index])
        mean = float(self.stats.branch_power_mean[branch_index])
        return value * std + mean


class LoadedGFRC(LoadedModel):
    def __init__(self, run_dir: Path, device: torch.device) -> None:
        run_config = read_json(run_dir / "run_config.json")
        super().__init__(
            name="GFRC",
            run_dir=run_dir,
            device=device,
            sample_steps=int(run_config.get("sample_steps", 80)),
            num_samples=int(run_config.get("num_eval_samples", run_config.get("num_samples", 20))),
            use_cue=bool(run_config.get("use_cue", True)),
        )
        self.ode_solver = str(run_config.get("ode_solver", "euler"))
        self.guidance_scale = float(run_config.get("guidance_scale", 1.0))
        self.eval_point_estimate_mode = str(run_config.get("eval_point_estimate_mode", "stochastic_mean"))
        self.eval_stochastic_samples = bool(run_config.get("eval_stochastic_samples", True))
        self.eval_enable_dropout = bool(run_config.get("eval_enable_dropout", True))
        checkpoint = torch.load(run_dir / "best_gfrc_model.pt", map_location=device, weights_only=False)
        config_dict = checkpoint["config"]
        config = TrainConfig(**config_dict)
        self.model = GFRCModel(
            hidden_dim=config.hidden_dim,
            flow_dim=config.flow_dim,
            num_encoder_layers=config.num_encoder_layers,
            num_flow_blocks=config.num_flow_blocks,
            num_heads=config.num_heads,
            dropout=config.dropout,
            self_condition=bool(getattr(config, "self_condition", False)),
            use_activity_gate=bool(getattr(config, "use_activity_gate", False)),
        ).to(device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.num_branches = int(checkpoint["num_branches"])
        self.self_condition = bool(getattr(config, "self_condition", False))
        self.flow_path = str(getattr(config, "flow_path", "linear"))
        self.flow_sigmoid_bias = float(getattr(config, "flow_sigmoid_bias", 0.0))
        self.flow_sigmoid_scale = float(getattr(config, "flow_sigmoid_scale", 2.0))
        norm_path = run_dir / "normalization_stats.json"
        if norm_path.exists():
            self.stats = NormalizationStats.from_dict(read_json(norm_path))

    def sample(self, x_r: torch.Tensor, x_p: torch.Tensor, p_k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.use_cue:
            p_k = torch.zeros_like(p_k)
        mean, _, samples = self.model.sample(
            x_r=x_r,
            x_p=x_p,
            p_k=p_k,
            num_steps=self.sample_steps,
            num_samples=self.num_samples,
            stochastic=True,
            enable_dropout=True,
            solver=self.ode_solver,
            guidance_scale=self.guidance_scale,
            self_condition=self.self_condition,
            flow_path=self.flow_path,
            flow_sigmoid_bias=self.flow_sigmoid_bias,
            flow_sigmoid_scale=self.flow_sigmoid_scale,
        )
        return mean, samples

    def predict_point(self, x_r: torch.Tensor, x_p: torch.Tensor, p_k: torch.Tensor) -> torch.Tensor:
        if not self.use_cue:
            p_k = torch.zeros_like(p_k)
        if self.eval_point_estimate_mode == "deterministic":
            mean, _, _ = self.model.sample(
                x_r=x_r,
                x_p=x_p,
                p_k=p_k,
                num_steps=self.sample_steps,
                num_samples=1,
                stochastic=False,
                enable_dropout=False,
                solver=self.ode_solver,
                guidance_scale=self.guidance_scale,
                self_condition=self.self_condition,
                flow_path=self.flow_path,
                flow_sigmoid_bias=self.flow_sigmoid_bias,
                flow_sigmoid_scale=self.flow_sigmoid_scale,
            )
            return mean
        mean, _ = self.sample(x_r, x_p, p_k)
        return mean


class LoadedBaseline(LoadedModel):
    def __init__(self, run_dir: Path, device: torch.device) -> None:
        run_config = read_json(run_dir / "run_config.json")
        method_name = str(run_config["method_name"])
        super().__init__(
            name=method_name,
            run_dir=run_dir,
            device=device,
            sample_steps=0,
            num_samples=int(run_config.get("num_samples", 20)),
            use_cue=bool(run_config.get("use_cue", True)),
        )
        checkpoint = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
        args = SimpleNamespace(**checkpoint["args"])
        self.model = build_model(method_name, args).to(device)
        self.model.use_cue = bool(run_config.get("use_cue", True))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.num_branches = int(checkpoint["num_branches"])
        norm_path = run_dir / "normalization_stats.json"
        if norm_path.exists():
            self.stats = NormalizationStats.from_dict(read_json(norm_path))

    def sample(self, x_r: torch.Tensor, x_p: torch.Tensor, p_k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.use_cue:
            p_k = torch.zeros_like(p_k)
        mean, samples = self.model.sample(x_r, x_p, p_k, num_samples=self.num_samples)
        return mean, samples


def load_model_from_run_dir(run_dir: Path, device: torch.device) -> LoadedModel:
    if (run_dir / "best_gfrc_model.pt").exists():
        return LoadedGFRC(run_dir, device)
    if (run_dir / "best_model.pt").exists():
        return LoadedBaseline(run_dir, device)
    raise FileNotFoundError(f"No supported checkpoint found in {run_dir}")


def build_test_loader(split_config_path: str, seq_len: int, batch_size: int):
    _, loaders = create_loaders_from_split_config(
        split_config_path=split_config_path,
        seq_len=seq_len,
        batch_size=batch_size,
        num_workers=0,
    )
    _, _, test_loader, _, num_branches = loaders
    return test_loader, num_branches
