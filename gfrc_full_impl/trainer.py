import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from config import TrainConfig
from dataset import NormalizationStats
from metrics import compute_point_metrics, coverage_and_width, crps_from_samples
from experiments.common import compute_alarm_metrics
from model import GFRCModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GFRCTrainer:
    def __init__(self, config: TrainConfig, num_branches: int, stats: NormalizationStats) -> None:
        self.config = config
        self.num_branches = num_branches
        self.stats = stats
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GFRCModel(
            hidden_dim=config.hidden_dim,
            flow_dim=config.flow_dim,
            num_encoder_layers=config.num_encoder_layers,
            num_flow_blocks=config.num_flow_blocks,
            num_heads=config.num_heads,
            dropout=config.dropout,
            self_condition=config.self_condition,
            use_activity_gate=config.use_activity_gate,
        ).to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.epochs)
        self.best_checkpoint_metric = float("inf")
        self.best_checkpoint_epoch = 0
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "epoch_seconds": [],
            "checkpoint_metric_values": [],
        }
        self.branch_power_mean = torch.as_tensor(stats.branch_power_mean, dtype=torch.float32, device=self.device)
        self.branch_power_std = torch.as_tensor(stats.branch_power_std, dtype=torch.float32, device=self.device)
        self.branch_current_mean = torch.as_tensor(stats.branch_current_mean, dtype=torch.float32, device=self.device)
        self.branch_current_std = torch.as_tensor(stats.branch_current_std, dtype=torch.float32, device=self.device)
        self.total_current_mean = torch.tensor(float(stats.total_current_mean), dtype=torch.float32, device=self.device)
        self.total_current_std = torch.tensor(float(stats.total_current_std), dtype=torch.float32, device=self.device)
        self.current_epoch = 1

    def _scheduled_weight(self, target: float, warmup_epochs: int) -> float:
        if target <= 0.0:
            return 0.0
        if warmup_epochs <= 0:
            return float(target)
        progress = min(max(self.current_epoch, 1) / float(warmup_epochs), 1.0)
        return float(target) * progress

    def _maybe_disable_cue(self, target_p: torch.Tensor) -> torch.Tensor:
        if self.config.use_cue:
            return target_p
        return torch.zeros_like(target_p)

    def _maybe_dropout_cue(self, target_p: torch.Tensor, train: bool) -> torch.Tensor:
        target_p = self._maybe_disable_cue(target_p)
        if not train or self.config.cue_dropout_prob <= 0.0:
            return target_p
        keep_mask = (
            torch.rand(target_p.size(0), 1, device=target_p.device, dtype=target_p.dtype)
            >= self.config.cue_dropout_prob
        ).float()
        return target_p * keep_mask

    def _flow_path(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return GFRCModel._path_map(
            t,
            self.config.flow_path,
            bias=float(self.config.flow_sigmoid_bias),
            scale=float(self.config.flow_sigmoid_scale),
        )

    def _construct_flow_batch(
        self,
        x_r: torch.Tensor,
        x_p: torch.Tensor,
        branch_powers: torch.Tensor,
        branch_currents: torch.Tensor,
        train: bool,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        branch_idx, target_p, target_y = self._sample_target_branch(branch_powers, branch_currents)
        target_p = self._maybe_dropout_cue(target_p, train=train)

        y_1 = target_y.unsqueeze(-1)
        y_0 = torch.randn_like(y_1)
        t = torch.rand(x_r.size(0), 1, device=self.device)
        alpha, alpha_prime = self._flow_path(t)
        x_t = (1.0 - alpha).unsqueeze(-1) * y_0 + alpha.unsqueeze(-1) * y_1
        v_target = alpha_prime.unsqueeze(-1) * (y_1 - y_0)

        self_condition_signal = None
        if self.config.self_condition:
            use_self_condition = (not train) or (
                torch.rand(1, device=self.device).item() < self.config.self_condition_prob
            )
            if use_self_condition:
                with torch.no_grad():
                    v_seed = self.model(x_t=x_t, x_r=x_r, x_p=x_p, p_k=target_p, t=t)
                    self_condition_signal = x_t + ((1.0 - alpha) / alpha_prime.clamp_min(1e-4)).unsqueeze(-1) * v_seed
            else:
                self_condition_signal = torch.zeros_like(x_t)

        return branch_idx, target_p, target_y, t, alpha, alpha_prime, x_t, v_target, self_condition_signal

    def _cumulative_distribution_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        pred_total = pred.sum(dim=-1)
        target_total = target.sum(dim=-1)
        per_sequence = torch.abs(pred_total - target_total) / torch.abs(target_total).clamp_min(1e-8)
        return {
            "med_rce": float(torch.quantile(per_sequence, 0.5).item()),
            "p90_rce": float(torch.quantile(per_sequence, 0.9).item()),
            "acc_at_100": float((per_sequence <= 1.0).float().mean().item()),
        }

    def flow_matching_loss(self, v_pred: torch.Tensor, v_target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(v_pred, v_target)

    def endpoint_loss(self, pred_sequence: torch.Tensor, target_sequence: torch.Tensor) -> torch.Tensor:
        if self.config.endpoint_loss_type == "huber":
            return F.huber_loss(pred_sequence, target_sequence, delta=float(self.config.endpoint_huber_delta))
        return F.mse_loss(pred_sequence, target_sequence)

    def multiscale_total_loss(self, pred_sequence: torch.Tensor, target_sequence: torch.Tensor) -> torch.Tensor:
        windows = tuple(int(window) for window in self.config.multiscale_windows if int(window) > 1)
        if not windows:
            return pred_sequence.new_tensor(0.0)

        losses = []
        seq_len = pred_sequence.size(1)
        for window in windows:
            window = min(window, seq_len)
            pad = (-seq_len) % window
            if pad > 0:
                pred_padded = F.pad(pred_sequence, (0, pad))
                target_padded = F.pad(target_sequence, (0, pad))
            else:
                pred_padded = pred_sequence
                target_padded = target_sequence
            pred_block = pred_padded.view(pred_padded.size(0), -1, window).sum(dim=-1)
            target_block = target_padded.view(target_padded.size(0), -1, window).sum(dim=-1)
            numerator = torch.sum(torch.abs(pred_block - target_block))
            denominator = torch.sum(torch.abs(target_block)).clamp_min(1e-6)
            losses.append(numerator / denominator)
        return torch.stack(losses).mean()

    def delta_loss(self, pred_sequence: torch.Tensor, target_sequence: torch.Tensor) -> torch.Tensor:
        pred_delta = pred_sequence[:, 1:] - pred_sequence[:, :-1]
        target_delta = target_sequence[:, 1:] - target_sequence[:, :-1]
        return F.l1_loss(pred_delta, target_delta)

    def activity_loss(self, activity_logits: torch.Tensor, target_sequence: torch.Tensor) -> torch.Tensor:
        threshold = float(self.config.activity_threshold)
        sharpness = float(self.config.activity_sharpness)
        activity_target = torch.sigmoid((target_sequence.abs() - threshold) * sharpness)
        return F.binary_cross_entropy_with_logits(activity_logits.squeeze(-1), activity_target)

    def alarm_loss(self, pred_sequence: torch.Tensor, target_sequence: torch.Tensor) -> torch.Tensor:
        threshold = float(self.config.alarm_threshold)
        sharpness = float(self.config.alarm_sharpness)
        pred_peak = pred_sequence.abs().amax(dim=1)
        target_peak = target_sequence.abs().amax(dim=1)
        pred_alarm = torch.sigmoid((pred_peak - threshold) * sharpness)
        target_alarm = (target_peak >= threshold).float()
        return F.binary_cross_entropy(pred_alarm, target_alarm)

    def _denormalize_branch_sequence(
        self,
        sequence: torch.Tensor,
        branch_idx: torch.Tensor,
    ) -> torch.Tensor:
        branch_mean = self.branch_current_mean[branch_idx].view(-1, 1)
        branch_std = self.branch_current_std[branch_idx].view(-1, 1)
        return sequence * branch_std + branch_mean

    def _denormalize_all_branch_sequences(self, sequences: torch.Tensor) -> torch.Tensor:
        branch_mean = self.branch_current_mean.view(1, -1, 1)
        branch_std = self.branch_current_std.view(1, -1, 1)
        return sequences * branch_std + branch_mean

    def _denormalize_branch_power_sequence(
        self,
        sequence: torch.Tensor,
        branch_idx: int,
    ) -> torch.Tensor:
        return sequence * self.branch_power_std[branch_idx] + self.branch_power_mean[branch_idx]

    def _denormalize_all_branch_powers(self, sequences: torch.Tensor) -> torch.Tensor:
        branch_mean = self.branch_power_mean.view(1, -1, 1)
        branch_std = self.branch_power_std.view(1, -1, 1)
        return sequences * branch_std + branch_mean

    def physical_constraint_loss(
        self,
        x_r: torch.Tensor,
        x_p: torch.Tensor,
        branch_powers: torch.Tensor,
    ) -> torch.Tensor:
        branch_power_input = branch_powers if self.config.use_cue else torch.zeros_like(branch_powers)
        branch_predictions = self.model.predict_all_branches(
            x_r=x_r,
            x_p=x_p,
            branch_powers=branch_power_input,
            num_steps=self.config.physical_sample_steps,
            stochastic=False,
            solver=self.config.physical_ode_solver,
            self_condition=self.config.self_condition,
            flow_path=self.config.flow_path,
            flow_sigmoid_bias=self.config.flow_sigmoid_bias,
            flow_sigmoid_scale=self.config.flow_sigmoid_scale,
        )
        branch_predictions = self._denormalize_all_branch_sequences(branch_predictions)
        total_prediction = branch_predictions.sum(dim=1)
        total_prediction = (total_prediction - self.total_current_mean) / self.total_current_std
        return F.mse_loss(total_prediction, x_r)

    def _sample_target_branch(
        self,
        branch_powers: torch.Tensor,
        branch_currents: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = branch_powers.size(0)
        if self.config.branch_sampling == "signal_weighted":
            branch_currents_denorm = self._denormalize_all_branch_sequences(branch_currents)
            branch_powers_denorm = self._denormalize_all_branch_powers(branch_powers)
            current_activity = branch_currents_denorm.abs().mean(dim=-1)
            power_activity = branch_powers_denorm.abs().mean(dim=-1)
            current_probs = current_activity / current_activity.sum(dim=1, keepdim=True).clamp_min(1e-6)
            power_probs = power_activity / power_activity.sum(dim=1, keepdim=True).clamp_min(1e-6)
            branch_probs = 0.8 * current_probs + 0.2 * power_probs
            branch_probs = branch_probs.clamp_min(1e-6)
            branch_logits = torch.log(branch_probs) / max(self.config.branch_sampling_temperature, 1e-3)
            branch_probs = torch.softmax(branch_logits, dim=1)
            branch_idx = torch.multinomial(branch_probs, num_samples=1).squeeze(1)
        else:
            branch_idx = torch.randint(0, self.num_branches, (batch_size,), device=branch_powers.device)
        gather_idx = branch_idx[:, None, None].expand(-1, 1, branch_powers.size(-1))
        target_p = branch_powers.gather(1, gather_idx).squeeze(1)
        target_y = branch_currents.gather(1, gather_idx).squeeze(1)
        return branch_idx, target_p, target_y

    def _step(self, batch: Dict[str, torch.Tensor], train: bool) -> Dict[str, float]:
        x_r = batch["x_r"].to(self.device)
        x_p = batch["x_p"].to(self.device)
        branch_powers = batch["branch_powers"].to(self.device)
        branch_currents = batch["branch_currents"].to(self.device)

        branch_idx, target_p, target_y, t, alpha, alpha_prime, x_t, v_target, self_condition_signal = self._construct_flow_batch(
            x_r=x_r,
            x_p=x_p,
            branch_powers=branch_powers,
            branch_currents=branch_currents,
            train=train,
        )

        if train:
            self.optimizer.zero_grad()

        condition = self.model.encode_condition(x_r=x_r, x_p=x_p, p_k=target_p)
        v_pred = self.model.flow_from_condition(
            x_t=x_t,
            t=t,
            condition=condition,
            self_condition=self_condition_signal,
        )
        flow_loss = self.flow_matching_loss(v_pred, v_target)
        endpoint_coeff = ((1.0 - alpha) / alpha_prime.clamp_min(1e-4)).unsqueeze(-1)
        endpoint_pred = x_t + endpoint_coeff * v_pred
        target_y_denorm = self._denormalize_branch_sequence(target_y, branch_idx)
        activity_logits = self.model.activity_logits_from_condition(condition)
        if self.config.use_activity_gate:
            endpoint_pred = endpoint_pred * torch.sigmoid(activity_logits)
        endpoint_pred_denorm = self._denormalize_branch_sequence(endpoint_pred.squeeze(-1), branch_idx)
        endpoint_loss = self.endpoint_loss(endpoint_pred_denorm, target_y_denorm)

        endpoint_total = endpoint_pred_denorm.sum(dim=1)
        target_total = target_y_denorm.sum(dim=1)
        sequence_total_loss = torch.sum(torch.abs(endpoint_total - target_total)) / torch.sum(
            target_total.abs()
        ).clamp_min(1e-6)
        multiscale_total_loss = self.multiscale_total_loss(endpoint_pred_denorm, target_y_denorm)
        prefix_cumsum_loss = F.l1_loss(
            endpoint_pred_denorm.cumsum(dim=1),
            target_y_denorm.cumsum(dim=1),
        )
        delta_loss = self.delta_loss(endpoint_pred_denorm, target_y_denorm)
        activity_loss = self.activity_loss(activity_logits, target_y_denorm)
        alarm_loss = self.alarm_loss(endpoint_pred_denorm, target_y_denorm)

        physical_loss = self.physical_constraint_loss(x_r=x_r, x_p=x_p, branch_powers=branch_powers)
        lambda_sequence_total = self._scheduled_weight(
            self.config.lambda_sequence_total,
            int(getattr(self.config, "warmup_epochs_sequence_total", 0)),
        )
        lambda_prefix_cumsum = self._scheduled_weight(
            self.config.lambda_prefix_cumsum,
            int(getattr(self.config, "warmup_epochs_prefix", 0)),
        )
        lambda_delta = self._scheduled_weight(
            self.config.lambda_delta,
            int(getattr(self.config, "warmup_epochs_delta", 0)),
        )
        lambda_physical = self._scheduled_weight(
            self.config.lambda_physical,
            int(getattr(self.config, "warmup_epochs_physical", 0)),
        )
        total_loss = (
            flow_loss
            + self.config.lambda_endpoint * endpoint_loss
            + lambda_sequence_total * sequence_total_loss
            + self.config.lambda_multiscale_total * multiscale_total_loss
            + lambda_prefix_cumsum * prefix_cumsum_loss
            + lambda_delta * delta_loss
            + self.config.lambda_activity * activity_loss
            + self.config.lambda_alarm * alarm_loss
            + lambda_physical * physical_loss
        )

        if train:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            self.optimizer.step()

        return {
            "loss": float(total_loss.item()),
            "flow_loss": float(flow_loss.item()),
            "endpoint_loss": float(endpoint_loss.item()),
            "sequence_total_loss": float(sequence_total_loss.item()),
            "multiscale_total_loss": float(multiscale_total_loss.item()),
            "prefix_cumsum_loss": float(prefix_cumsum_loss.item()),
            "delta_loss": float(delta_loss.item()),
            "activity_loss": float(activity_loss.item()),
            "alarm_loss": float(alarm_loss.item()),
            "physical_loss": float(physical_loss.item()),
        }

    def run_epoch(self, loader, train: bool) -> Dict[str, float]:
        self.model.train(mode=train)
        prefix = "train" if train else "val"
        running = {
            "loss": 0.0,
            "flow_loss": 0.0,
            "endpoint_loss": 0.0,
            "sequence_total_loss": 0.0,
            "multiscale_total_loss": 0.0,
            "prefix_cumsum_loss": 0.0,
            "delta_loss": 0.0,
            "activity_loss": 0.0,
            "alarm_loss": 0.0,
            "physical_loss": 0.0,
        }
        max_batches = self.config.max_train_batches if train else self.config.max_val_batches
        progress_total = len(loader)
        if max_batches is not None:
            progress_total = min(progress_total, max_batches)
        progress = tqdm(loader, total=progress_total, desc=prefix.capitalize(), leave=False)
        batch_count = 0
        for batch in progress:
            with torch.set_grad_enabled(train):
                metrics = self._step(batch, train=train)
            for key, value in metrics.items():
                running[key] += value
            progress.set_postfix(
                loss=f"{metrics['loss']:.4f}",
                flow=f"{metrics['flow_loss']:.4f}",
                end=f"{metrics['endpoint_loss']:.4f}",
                seq=f"{metrics['sequence_total_loss']:.4f}",
                ms=f"{metrics['multiscale_total_loss']:.4f}",
                pre=f"{metrics['prefix_cumsum_loss']:.4f}",
                delta=f"{metrics['delta_loss']:.4f}",
                act=f"{metrics['activity_loss']:.4f}",
                alarm=f"{metrics['alarm_loss']:.4f}",
                phys=f"{metrics['physical_loss']:.4f}",
            )
            batch_count += 1
            if max_batches is not None and batch_count >= max_batches:
                break

        count = max(batch_count, 1)
        return {key: value / count for key, value in running.items()}

    def save_checkpoint(self, epoch: int, checkpoint_metric_value: float) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.__dict__,
            "best_checkpoint_metric": checkpoint_metric_value,
            "checkpoint_metric_name": self.config.checkpoint_metric,
            "num_branches": self.num_branches,
        }
        torch.save(checkpoint, self.output_dir / "best_gfrc_model.pt")

    def load_initial_checkpoint(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        source_state = checkpoint["model_state_dict"]
        current_state = self.model.state_dict()
        patched_state = {}
        for key, current_tensor in current_state.items():
            source_tensor = source_state.get(key)
            if source_tensor is None:
                patched_state[key] = current_tensor
                continue
            if source_tensor.shape == current_tensor.shape:
                patched_state[key] = source_tensor
                continue
            if key == "flow.input_proj.weight" and source_tensor.ndim == 2 and current_tensor.ndim == 2:
                copied = current_tensor.clone()
                rows = min(copied.shape[0], source_tensor.shape[0])
                cols = min(copied.shape[1], source_tensor.shape[1])
                copied[:rows, :cols] = source_tensor[:rows, :cols]
                patched_state[key] = copied
                continue
            patched_state[key] = current_tensor
        self.model.load_state_dict(patched_state, strict=False)
        print(f"Loaded initial weights from {checkpoint_path}")

    def _uses_eval_checkpoint_metric(self) -> bool:
        return self.config.checkpoint_metric.startswith("eval_")

    def _resolve_checkpoint_metric_value(
        self,
        train_val_metrics: Dict[str, float],
        checkpoint_eval_metrics: Dict[str, float] | None,
    ) -> float:
        if not self._uses_eval_checkpoint_metric():
            return float(train_val_metrics[self.config.checkpoint_metric])
        if checkpoint_eval_metrics is None:
            raise ValueError("checkpoint_eval_metrics is required for eval-based checkpoint selection.")

        metric_name = self.config.checkpoint_metric
        if metric_name == "eval_rmse":
            return float(checkpoint_eval_metrics["rmse"])
        if metric_name == "eval_ntae":
            return float(checkpoint_eval_metrics["ntae"])
        if metric_name == "eval_med_rce":
            return float(checkpoint_eval_metrics["med_rce"])
        if metric_name == "eval_p90_rce":
            return float(checkpoint_eval_metrics["p90_rce"])
        if metric_name == "eval_acc_at_100_neg":
            return float(checkpoint_eval_metrics["eval_acc_at_100_neg"])
        if metric_name == "eval_clea_neg":
            return float(checkpoint_eval_metrics["eval_clea_neg"])
        if metric_name == "eval_alarm_f1_neg":
            return float(checkpoint_eval_metrics["eval_alarm_f1_neg"])
        raise ValueError(f"Unsupported checkpoint metric: {metric_name}")

    def _estimate_checkpoint_eval_metrics(self, loader) -> Dict[str, float]:
        self.model.eval()

        max_batches = self.config.checkpoint_eval_max_batches
        calibrators = None
        if bool(getattr(self.config, "checkpoint_eval_use_calibration", False)):
            calibrators = self._fit_sequence_total_calibrators(loader, max_batches=max_batches)

        point_estimate_mode = str(getattr(self.config, "eval_point_estimate_mode", "stochastic_mean"))
        use_deterministic_point_estimate = point_estimate_mode == "deterministic"
        eval_stochastic_samples = bool(getattr(self.config, "eval_stochastic_samples", True))
        eval_enable_dropout = bool(getattr(self.config, "eval_enable_dropout", True))
        all_means = []
        all_targets = []
        batch_count = 0

        for batch in tqdm(loader, desc="CheckpointEval", leave=False):
            x_r = batch["x_r"].to(self.device)
            x_p = batch["x_p"].to(self.device)
            branch_powers = batch["branch_powers"].to(self.device)
            branch_currents = batch["branch_currents"].to(self.device)

            for branch_index in range(self.num_branches):
                p_k = branch_powers[:, branch_index, :]
                p_k = self._maybe_disable_cue(p_k)
                target = branch_currents[:, branch_index, :]

                if use_deterministic_point_estimate:
                    mean, _, _ = self.model.sample(
                        x_r=x_r,
                        x_p=x_p,
                        p_k=p_k,
                        num_steps=self.config.sample_steps,
                        num_samples=1,
                        stochastic=False,
                        enable_dropout=False,
                        solver=self.config.ode_solver,
                        guidance_scale=self.config.guidance_scale,
                        self_condition=self.config.self_condition,
                        flow_path=self.config.flow_path,
                        flow_sigmoid_bias=self.config.flow_sigmoid_bias,
                        flow_sigmoid_scale=self.config.flow_sigmoid_scale,
                    )
                else:
                    mean, _, _ = self.model.sample(
                        x_r=x_r,
                        x_p=x_p,
                        p_k=p_k,
                        num_steps=self.config.sample_steps,
                        num_samples=self.config.num_eval_samples,
                        stochastic=eval_stochastic_samples,
                        enable_dropout=eval_enable_dropout,
                        solver=self.config.ode_solver,
                        guidance_scale=self.config.guidance_scale,
                        self_condition=self.config.self_condition,
                        flow_path=self.config.flow_path,
                        flow_sigmoid_bias=self.config.flow_sigmoid_bias,
                        flow_sigmoid_scale=self.config.flow_sigmoid_scale,
                    )
                std = float(self.stats.branch_current_std[branch_index])
                offset = float(self.stats.branch_current_mean[branch_index])
                mean = mean.detach().cpu() * std + offset
                target = target.detach().cpu() * std + offset
                if calibrators is not None:
                    cue_sequence = p_k.detach().cpu() * float(self.stats.branch_power_std[branch_index]) + float(
                        self.stats.branch_power_mean[branch_index]
                    )
                    mean, _ = self._apply_sequence_total_calibration(
                        branch_index=branch_index,
                        mean=mean,
                        samples=mean.unsqueeze(0),
                        cue_sequence=cue_sequence,
                        calibrators=calibrators,
                    )

                all_means.append(mean)
                all_targets.append(target)

            batch_count += 1
            if max_batches is not None and batch_count >= max_batches:
                break

        pred = torch.cat(all_means, dim=0)
        target = torch.cat(all_targets, dim=0)
        metrics = compute_point_metrics(pred, target)
        if "acc_at_100" not in metrics or "med_rce" not in metrics or "p90_rce" not in metrics:
            metrics.update(self._cumulative_distribution_metrics(pred, target))
        metrics.update(compute_alarm_metrics(pred, target, threshold=float(self.config.alarm_threshold)))
        metrics["eval_acc_at_100_neg"] = -metrics["acc_at_100"]
        metrics["eval_clea_neg"] = -metrics["clea"]
        metrics["eval_alarm_f1_neg"] = -metrics["alarm_f1"]
        return metrics

    def train(self, train_loader, val_loader) -> None:
        print(f"Training GFRC on {self.device}")
        print(f"Model parameters: {sum(param.numel() for param in self.model.parameters()):,}")
        epochs_without_improvement = 0
        stopped_early = False
        for epoch in range(1, self.config.epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.perf_counter()
            train_metrics = self.run_epoch(train_loader, train=True)
            val_metrics = self.run_epoch(val_loader, train=False)
            self.scheduler.step()
            epoch_seconds = time.perf_counter() - epoch_start

            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["epoch_seconds"].append(epoch_seconds)

            checkpoint_eval_metrics = None
            if self._uses_eval_checkpoint_metric():
                checkpoint_eval_metrics = self._estimate_checkpoint_eval_metrics(val_loader)
            checkpoint_metric_value = self._resolve_checkpoint_metric_value(val_metrics, checkpoint_eval_metrics)
            self.history["checkpoint_metric_values"].append(checkpoint_metric_value)
            improvement_threshold = self.best_checkpoint_metric - float(self.config.early_stopping_min_delta)
            if checkpoint_metric_value < improvement_threshold:
                self.best_checkpoint_metric = checkpoint_metric_value
                self.best_checkpoint_epoch = epoch
                self.save_checkpoint(epoch=epoch, checkpoint_metric_value=self.best_checkpoint_metric)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            eval_suffix = ""
            if checkpoint_eval_metrics is not None:
                eval_suffix = (
                    f" | eval_rmse={checkpoint_eval_metrics['rmse']:.4f}"
                    f" eval_ntae={checkpoint_eval_metrics['ntae']:.4f}"
                    f" eval_clea={checkpoint_eval_metrics['clea']:.4f}"
                    f" eval_alarm_f1={checkpoint_eval_metrics['alarm_f1']:.4f}"
                )
            print(
                f"Epoch {epoch:03d} | "
                f"train={train_metrics['loss']:.4f} "
                f"(flow={train_metrics['flow_loss']:.4f}, end={train_metrics['endpoint_loss']:.4f}, "
                f"seq={train_metrics['sequence_total_loss']:.4f}, ms={train_metrics['multiscale_total_loss']:.4f}, "
                f"delta={train_metrics['delta_loss']:.4f}, act={train_metrics['activity_loss']:.4f}, "
                f"phys={train_metrics['physical_loss']:.4f}) | "
                f"val_ms={val_metrics['multiscale_total_loss']:.4f} "
                f"val_delta={val_metrics['delta_loss']:.4f} | "
                f"val={val_metrics['loss']:.4f} | "
                f"best_{self.config.checkpoint_metric}={self.best_checkpoint_metric:.4f} | "
                f"time={epoch_seconds:.1f}s"
                f"{eval_suffix}"
            )

            if (
                self.config.early_stopping_patience is not None
                and epochs_without_improvement >= int(self.config.early_stopping_patience)
            ):
                stopped_early = True
                print(
                    "Early stopping triggered "
                    f"at epoch {epoch:03d} after {epochs_without_improvement} epochs "
                    f"without improving {self.config.checkpoint_metric}."
                )
                break

        history_path = self.output_dir / "training_history.json"
        history_path.write_text(json.dumps(self.history, indent=2))
        summary_path = self.output_dir / "training_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "best_checkpoint_metric": self.best_checkpoint_metric,
                    "best_checkpoint_epoch": self.best_checkpoint_epoch,
                    "checkpoint_metric_name": self.config.checkpoint_metric,
                    "epoch_seconds": self.history["epoch_seconds"],
                    "checkpoint_metric_values": self.history["checkpoint_metric_values"],
                    "epochs_ran": len(self.history["epoch_seconds"]),
                    "stopped_early": stopped_early,
                    "early_stopping_patience": self.config.early_stopping_patience,
                    "early_stopping_min_delta": self.config.early_stopping_min_delta,
                    "avg_epoch_seconds": float(np.mean(self.history["epoch_seconds"])) if self.history["epoch_seconds"] else 0.0,
                },
                indent=2,
            )
        )

    def load_best_checkpoint(self) -> None:
        checkpoint_path = self.output_dir / "best_gfrc_model.pt"
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])

    def _fit_sequence_total_calibrators(self, loader, max_batches: int | None = None) -> List[Dict[str, object] | None]:
        per_branch_pred_totals: List[List[np.ndarray]] = [[] for _ in range(self.num_branches)]
        per_branch_cue_totals: List[List[np.ndarray]] = [[] for _ in range(self.num_branches)]
        per_branch_target_totals: List[List[np.ndarray]] = [[] for _ in range(self.num_branches)]
        batch_count = 0

        for batch in tqdm(loader, desc="Calibrate", leave=False):
            x_r = batch["x_r"].to(self.device)
            x_p = batch["x_p"].to(self.device)
            branch_powers = batch["branch_powers"].to(self.device)
            branch_currents = batch["branch_currents"].to(self.device)

            for branch_index in range(self.num_branches):
                p_k = branch_powers[:, branch_index, :]
                p_k = self._maybe_disable_cue(p_k)
                target = branch_currents[:, branch_index, :]

                mean, _, _ = self.model.sample(
                    x_r=x_r,
                    x_p=x_p,
                    p_k=p_k,
                    num_steps=self.config.sample_steps,
                    num_samples=self.config.num_eval_samples,
                    stochastic=True,
                    enable_dropout=True,
                    solver=self.config.ode_solver,
                    guidance_scale=self.config.guidance_scale,
                    self_condition=self.config.self_condition,
                    flow_path=self.config.flow_path,
                    flow_sigmoid_bias=self.config.flow_sigmoid_bias,
                    flow_sigmoid_scale=self.config.flow_sigmoid_scale,
                )

                mean_denorm = mean.detach().cpu() * float(self.stats.branch_current_std[branch_index]) + float(
                    self.stats.branch_current_mean[branch_index]
                )
                target_denorm = target.detach().cpu() * float(self.stats.branch_current_std[branch_index]) + float(
                    self.stats.branch_current_mean[branch_index]
                )
                cue_denorm = p_k.detach().cpu() * float(self.stats.branch_power_std[branch_index]) + float(
                    self.stats.branch_power_mean[branch_index]
                )

                per_branch_pred_totals[branch_index].append(mean_denorm.sum(dim=1).numpy())
                per_branch_cue_totals[branch_index].append(cue_denorm.sum(dim=1).numpy())
                per_branch_target_totals[branch_index].append(target_denorm.sum(dim=1).numpy())

            batch_count += 1
            if max_batches is not None and batch_count >= max_batches:
                break

        calibrators: List[Dict[str, object] | None] = []
        for branch_index in range(self.num_branches):
            pred_total = np.concatenate(per_branch_pred_totals[branch_index], axis=0)
            cue_total = np.concatenate(per_branch_cue_totals[branch_index], axis=0)
            target_total = np.concatenate(per_branch_target_totals[branch_index], axis=0)

            features = np.stack([pred_total, cue_total], axis=1).astype(np.float64)
            feature_mean = features.mean(axis=0)
            feature_std = features.std(axis=0)
            feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
            standardized = (features - feature_mean) / feature_std
            design = np.concatenate([standardized, np.ones((standardized.shape[0], 1), dtype=np.float64)], axis=1)
            coef, *_ = np.linalg.lstsq(design, target_total.astype(np.float64), rcond=None)
            calibrators.append(
                {
                    "feature_mean": feature_mean.tolist(),
                    "feature_std": feature_std.tolist(),
                    "coef": coef.tolist(),
                }
            )
        return calibrators

    def _apply_sequence_total_calibration(
        self,
        *,
        branch_index: int,
        mean: torch.Tensor,
        samples: torch.Tensor,
        cue_sequence: torch.Tensor,
        calibrators: List[Dict[str, object] | None],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        calibrator = calibrators[branch_index]
        if calibrator is None:
            return mean, samples

        pred_total = mean.sum(dim=1).numpy().astype(np.float64)
        cue_total = cue_sequence.sum(dim=1).numpy().astype(np.float64)
        feature_mean = np.asarray(calibrator["feature_mean"], dtype=np.float64)
        feature_std = np.asarray(calibrator["feature_std"], dtype=np.float64)
        coef = np.asarray(calibrator["coef"], dtype=np.float64)
        features = np.stack([pred_total, cue_total], axis=1)
        standardized = (features - feature_mean) / feature_std
        calibrated_total = standardized @ coef[:2] + coef[2]
        calibrated_total = np.clip(calibrated_total, 1e-6, None)

        scale = calibrated_total / np.clip(pred_total, 1e-6, None)
        scale = np.clip(scale, 0.2, 5.0).astype(np.float32)
        scale_tensor = torch.from_numpy(scale)
        mean = mean * scale_tensor.unsqueeze(1)
        samples = samples * scale_tensor.view(1, -1, 1)
        return mean, samples

    def evaluate(self, loader, calibration_loader=None) -> Dict[str, float]:
        self.load_best_checkpoint()
        self.model.eval()

        calibrators = None
        if calibration_loader is not None:
            calibrators = self._fit_sequence_total_calibrators(calibration_loader)

        point_estimate_mode = str(getattr(self.config, "eval_point_estimate_mode", "stochastic_mean"))
        use_deterministic_point_estimate = point_estimate_mode == "deterministic"
        eval_stochastic_samples = bool(getattr(self.config, "eval_stochastic_samples", True))
        eval_enable_dropout = bool(getattr(self.config, "eval_enable_dropout", True))

        all_means = []
        all_targets = []
        all_samples = []
        batch_times = []

        for batch in tqdm(loader, desc="Evaluate", leave=False):
            x_r = batch["x_r"].to(self.device)
            x_p = batch["x_p"].to(self.device)
            branch_powers = batch["branch_powers"].to(self.device)
            branch_currents = batch["branch_currents"].to(self.device)

            for branch_index in range(self.num_branches):
                p_k = branch_powers[:, branch_index, :]
                p_k = self._maybe_disable_cue(p_k)
                target = branch_currents[:, branch_index, :]

                point_mean = None
                if use_deterministic_point_estimate:
                    point_mean, _, _ = self.model.sample(
                        x_r=x_r,
                        x_p=x_p,
                        p_k=p_k,
                        num_steps=self.config.sample_steps,
                        num_samples=1,
                        stochastic=False,
                        enable_dropout=False,
                        solver=self.config.ode_solver,
                        guidance_scale=self.config.guidance_scale,
                        self_condition=self.config.self_condition,
                        flow_path=self.config.flow_path,
                        flow_sigmoid_bias=self.config.flow_sigmoid_bias,
                        flow_sigmoid_scale=self.config.flow_sigmoid_scale,
                    )

                start_time = time.perf_counter()
                mean, _, samples = self.model.sample(
                    x_r=x_r,
                    x_p=x_p,
                    p_k=p_k,
                    num_steps=self.config.sample_steps,
                    num_samples=self.config.num_eval_samples,
                    stochastic=eval_stochastic_samples,
                    enable_dropout=eval_enable_dropout,
                    solver=self.config.ode_solver,
                    guidance_scale=self.config.guidance_scale,
                    self_condition=self.config.self_condition,
                    flow_path=self.config.flow_path,
                    flow_sigmoid_bias=self.config.flow_sigmoid_bias,
                    flow_sigmoid_scale=self.config.flow_sigmoid_scale,
                )
                batch_times.append(time.perf_counter() - start_time)

                std = float(self.stats.branch_current_std[branch_index])
                offset = float(self.stats.branch_current_mean[branch_index])
                if point_mean is not None:
                    point_mean = point_mean.detach().cpu() * std + offset
                mean = mean.detach().cpu() * std + offset
                all_targets.append(target.detach().cpu() * std + offset)
                samples = samples.detach().cpu() * std + offset

                if calibrators is not None:
                    cue_sequence = p_k.detach().cpu() * float(self.stats.branch_power_std[branch_index]) + float(
                        self.stats.branch_power_mean[branch_index]
                    )
                    if point_mean is not None:
                        point_mean, _ = self._apply_sequence_total_calibration(
                            branch_index=branch_index,
                            mean=point_mean,
                            samples=point_mean.unsqueeze(0),
                            cue_sequence=cue_sequence,
                            calibrators=calibrators,
                        )
                    mean, samples = self._apply_sequence_total_calibration(
                        branch_index=branch_index,
                        mean=mean,
                        samples=samples,
                        cue_sequence=cue_sequence,
                        calibrators=calibrators,
                    )

                all_means.append(point_mean if point_mean is not None else mean)
                all_samples.append(samples)

        pred = torch.cat(all_means, dim=0)
        target = torch.cat(all_targets, dim=0)
        samples = torch.cat(all_samples, dim=1)

        metrics = compute_point_metrics(pred, target)
        if "acc_at_100" not in metrics or "med_rce" not in metrics or "p90_rce" not in metrics:
            metrics.update(self._cumulative_distribution_metrics(pred, target))
        metrics.update(compute_alarm_metrics(pred, target, threshold=float(self.config.alarm_threshold)))
        metrics["crps"] = float(crps_from_samples(samples, target).item())
        metrics.update(coverage_and_width(samples, target, alpha=0.1))
        metrics["time_per_call_seconds"] = float(np.mean(batch_times)) if batch_times else 0.0

        report_path = self.output_dir / "evaluation_metrics.json"
        report_path.write_text(json.dumps(metrics, indent=2))
        return metrics


def build_trainer(config: TrainConfig, num_branches: int, stats: NormalizationStats) -> GFRCTrainer:
    set_seed(config.seed)
    return GFRCTrainer(config=config, num_branches=num_branches, stats=stats)
