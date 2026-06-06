import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half_dim = dim // 2
    scale = math.log(10000.0) / max(half_dim - 1, 1)
    freq = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -scale)
    args = t * freq.unsqueeze(0)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class MultiModalFeatureEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.current_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.current_rnn = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.power_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
        )
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.fusion_norm = nn.LayerNorm(hidden_dim * 2)

    def forward(self, x_r: torch.Tensor, x_p: torch.Tensor, p_k: torch.Tensor) -> torch.Tensor:
        current = self.current_conv(x_r.unsqueeze(1)).transpose(1, 2)
        current, _ = self.current_rnn(current)

        power = torch.stack([x_p, p_k], dim=-1)
        power = self.power_mlp(power)

        fused, _ = self.attention(query=current, key=power, value=power)
        return self.fusion_norm(current + fused)


class FiLMResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm_1 = nn.LayerNorm(hidden_dim)
        self.norm_2 = nn.LayerNorm(hidden_dim)
        self.fc_1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.gamma = nn.Linear(hidden_dim, hidden_dim)
        self.beta = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm_1(x)
        h = self.fc_1(h)
        h = F.silu(h)
        h = self.dropout(h)

        gamma = self.gamma(condition)
        beta = self.beta(condition)
        h = h * (1.0 + gamma) + beta

        h = self.norm_2(h)
        h = self.fc_2(h)
        h = self.dropout(h)
        return residual + h


class ConditionalFlowNetwork(nn.Module):
    def __init__(
        self,
        condition_dim: int,
        hidden_dim: int,
        num_blocks: int,
        dropout: float,
        self_condition: bool = False,
    ) -> None:
        super().__init__()
        self.self_condition = self_condition
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.condition_proj = nn.Linear(condition_dim, hidden_dim)
        input_dim = 1 + hidden_dim + hidden_dim + (1 if self_condition else 0)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([FiLMResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        self_condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x_t: (batch, seq_len, 1), t: (batch, 1)
        time_embed = sinusoidal_time_embedding(t, self.condition_proj.out_features)
        time_embed = self.time_mlp(time_embed).unsqueeze(1).expand(-1, x_t.size(1), -1)

        cond_proj = self.condition_proj(condition)
        if self.self_condition:
            if self_condition is None:
                self_condition = torch.zeros_like(x_t)
            h = torch.cat([x_t, time_embed, cond_proj, self_condition], dim=-1)
        else:
            h = torch.cat([x_t, time_embed, cond_proj], dim=-1)
        h = self.input_proj(h)
        for block in self.blocks:
            h = block(h, cond_proj)
        return self.output_proj(h)


class GFRCModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        flow_dim: int,
        num_encoder_layers: int,
        num_flow_blocks: int,
        num_heads: int,
        dropout: float,
        self_condition: bool = False,
        use_activity_gate: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = MultiModalFeatureEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.flow = ConditionalFlowNetwork(
            condition_dim=hidden_dim * 2,
            hidden_dim=flow_dim,
            num_blocks=num_flow_blocks,
            dropout=dropout,
            self_condition=self_condition,
        )
        self.self_condition = self_condition
        self.use_activity_gate = use_activity_gate
        self.activity_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode_condition(self, x_r: torch.Tensor, x_p: torch.Tensor, p_k: torch.Tensor) -> torch.Tensor:
        return self.encoder(x_r, x_p, p_k)

    def flow_from_condition(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        self_condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.flow(x_t, t, condition, self_condition=self_condition)

    def activity_logits_from_condition(self, condition: torch.Tensor) -> torch.Tensor:
        return self.activity_head(condition)

    def forward(
        self,
        x_t: torch.Tensor,
        x_r: torch.Tensor,
        x_p: torch.Tensor,
        p_k: torch.Tensor,
        t: torch.Tensor,
        self_condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        condition = self.encode_condition(x_r, x_p, p_k)
        return self.flow_from_condition(x_t, t, condition, self_condition=self_condition)

    @staticmethod
    def _path_map(t: torch.Tensor, path: str, bias: float = 0.0, scale: float = 2.0) -> Tuple[torch.Tensor, torch.Tensor]:
        if path == "linear":
            alpha = t
            alpha_prime = torch.ones_like(t)
        elif path == "cosine":
            alpha = 0.5 - 0.5 * torch.cos(math.pi * t)
            alpha_prime = 0.5 * math.pi * torch.sin(math.pi * t)
        elif path == "exp":
            denom = 1.0 - math.exp(-scale)
            alpha = (1.0 - torch.exp(-scale * t)) / denom
            alpha_prime = scale * torch.exp(-scale * t) / denom
        elif path == "sigmoid":
            # A smooth path with adjustable midpoint bias; scale controls steepness.
            centered = (t - 0.5) * scale + bias
            alpha = torch.sigmoid(centered)
            alpha_prime = alpha * (1.0 - alpha) * scale
        else:
            raise ValueError(f"Unsupported flow path: {path}")
        return alpha, alpha_prime

    def sample_single(
        self,
        x_r: torch.Tensor,
        x_p: torch.Tensor,
        p_k: torch.Tensor,
        num_steps: int,
        stochastic: bool = True,
        solver: str = "euler",
        guidance_scale: float = 1.0,
        self_condition: bool = False,
        flow_path: str = "linear",
        flow_sigmoid_bias: float = 0.0,
        flow_sigmoid_scale: float = 2.0,
    ) -> torch.Tensor:
        condition = self.encode_condition(x_r, x_p, p_k)
        use_guidance = guidance_scale is not None and guidance_scale != 1.0
        unconditional_condition = None
        if use_guidance:
            unconditional_condition = self.encode_condition(x_r, x_p, torch.zeros_like(p_k))
        if stochastic:
            x_t = torch.randn(x_r.size(0), x_r.size(1), 1, device=x_r.device, dtype=x_r.dtype)
        else:
            x_t = torch.zeros(x_r.size(0), x_r.size(1), 1, device=x_r.device, dtype=x_r.dtype)
        self_cond = None

        dt = 1.0 / float(num_steps)
        for step in range(num_steps):
            t = torch.full((x_r.size(0), 1), step / float(num_steps), device=x_r.device, dtype=x_r.dtype)
            alpha, alpha_prime = self._path_map(t, flow_path, bias=flow_sigmoid_bias, scale=flow_sigmoid_scale)
            path_coeff = (1.0 - alpha) / alpha_prime.clamp_min(1e-4)
            current_self_condition = self_cond if self_condition else None
            if solver == "euler":
                if use_guidance:
                    v_cond = self.flow_from_condition(x_t, t, condition, self_condition=current_self_condition)
                    v_uncond = self.flow_from_condition(
                        x_t, t, unconditional_condition, self_condition=current_self_condition
                    )
                    v_theta = v_uncond + guidance_scale * (v_cond - v_uncond)
                else:
                    v_theta = self.flow_from_condition(x_t, t, condition, self_condition=current_self_condition)
                x_t = x_t + dt * v_theta
            elif solver == "heun":
                if use_guidance:
                    v_cond = self.flow_from_condition(x_t, t, condition, self_condition=current_self_condition)
                    v_uncond = self.flow_from_condition(
                        x_t, t, unconditional_condition, self_condition=current_self_condition
                    )
                    v_theta = v_uncond + guidance_scale * (v_cond - v_uncond)
                else:
                    v_theta = self.flow_from_condition(x_t, t, condition, self_condition=current_self_condition)
                x_euler = x_t + dt * v_theta
                t_next = torch.full(
                    (x_r.size(0), 1),
                    min((step + 1) / float(num_steps), 1.0),
                    device=x_r.device,
                    dtype=x_r.dtype,
                )
                alpha_next, alpha_prime_next = self._path_map(
                    t_next, flow_path, bias=flow_sigmoid_bias, scale=flow_sigmoid_scale
                )
                path_coeff_next = (1.0 - alpha_next) / alpha_prime_next.clamp_min(1e-4)
                if use_guidance:
                    v_cond_next = self.flow_from_condition(
                        x_euler, t_next, condition, self_condition=current_self_condition
                    )
                    v_uncond_next = self.flow_from_condition(
                        x_euler, t_next, unconditional_condition, self_condition=current_self_condition
                    )
                    v_theta_next = v_uncond_next + guidance_scale * (v_cond_next - v_uncond_next)
                else:
                    v_theta_next = self.flow_from_condition(
                        x_euler, t_next, condition, self_condition=current_self_condition
                    )
                x_t = x_t + 0.5 * dt * (v_theta + v_theta_next)
            else:
                raise ValueError(f"Unsupported ODE solver: {solver}")
            if self_condition:
                if solver == "heun":
                    self_cond = x_t + path_coeff_next.unsqueeze(-1) * v_theta_next
                else:
                    self_cond = x_t + path_coeff.unsqueeze(-1) * v_theta
        x_out = x_t.squeeze(-1)
        if self.use_activity_gate:
            gate = torch.sigmoid(self.activity_logits_from_condition(condition)).squeeze(-1)
            x_out = x_out * gate
        return x_out

    def sample(
        self,
        x_r: torch.Tensor,
        x_p: torch.Tensor,
        p_k: torch.Tensor,
        num_steps: int,
        num_samples: int = 1,
        stochastic: bool = True,
        enable_dropout: bool = False,
        solver: str = "euler",
        guidance_scale: float = 1.0,
        self_condition: bool = False,
        flow_path: str = "linear",
        flow_sigmoid_bias: float = 0.0,
        flow_sigmoid_scale: float = 2.0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        previous_mode = self.training
        if enable_dropout:
            self.train()
        else:
            self.eval()

        samples = []
        with torch.no_grad():
            for _ in range(num_samples):
                samples.append(
                    self.sample_single(
                        x_r,
                        x_p,
                        p_k,
                        num_steps=num_steps,
                        stochastic=stochastic,
                        solver=solver,
                        guidance_scale=guidance_scale,
                        self_condition=self_condition,
                        flow_path=flow_path,
                        flow_sigmoid_bias=flow_sigmoid_bias,
                        flow_sigmoid_scale=flow_sigmoid_scale,
                    )
                )
        stacked = torch.stack(samples, dim=0)
        mean = stacked.mean(dim=0)
        variance = stacked.var(dim=0, unbiased=False) if num_samples > 1 else None

        if previous_mode:
            self.train()
        else:
            self.eval()

        return mean, variance, stacked

    def predict_all_branches(
        self,
        x_r: torch.Tensor,
        x_p: torch.Tensor,
        branch_powers: torch.Tensor,
        num_steps: int,
        stochastic: bool = False,
        solver: str = "euler",
        guidance_scale: float = 1.0,
        self_condition: bool = False,
        flow_path: str = "linear",
        flow_sigmoid_bias: float = 0.0,
        flow_sigmoid_scale: float = 2.0,
    ) -> torch.Tensor:
        predictions = []
        for branch_index in range(branch_powers.size(1)):
            p_k = branch_powers[:, branch_index, :]
            prediction = self.sample_single(
                x_r,
                x_p,
                p_k,
                num_steps=num_steps,
                stochastic=stochastic,
                solver=solver,
                guidance_scale=guidance_scale,
                self_condition=self_condition,
                flow_path=flow_path,
                flow_sigmoid_bias=flow_sigmoid_bias,
                flow_sigmoid_scale=flow_sigmoid_scale,
            )
            predictions.append(prediction)
        return torch.stack(predictions, dim=1)
