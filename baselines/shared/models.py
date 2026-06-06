import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.shared.bootstrap import REPO_ROOT  # noqa: F401
from gfrc_full_impl.model import MultiModalFeatureEncoder


def build_feature_tensor(x_r: torch.Tensor, x_p: torch.Tensor, p_k: torch.Tensor) -> torch.Tensor:
    return torch.stack([x_r, x_p, p_k], dim=-1)


class BaseBaselineModel(nn.Module):
    probabilistic = False

    def loss_terms(
        self,
        x_r: torch.Tensor,
        x_p: torch.Tensor,
        p_k: torch.Tensor,
        target_y: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        pred = self.predict(x_r, x_p, p_k)
        loss = F.mse_loss(pred, target_y)
        return {
            "loss": loss,
            "recon_loss": loss,
        }

    def predict(self, x_r: torch.Tensor, x_p: torch.Tensor, p_k: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sample(
        self,
        x_r: torch.Tensor,
        x_p: torch.Tensor,
        p_k: torch.Tensor,
        num_samples: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.predict(x_r, x_p, p_k)
        samples = mean.unsqueeze(0).repeat(num_samples, 1, 1)
        return mean, samples

    def predict_all_branches(
        self,
        x_r: torch.Tensor,
        x_p: torch.Tensor,
        branch_powers: torch.Tensor,
    ) -> torch.Tensor:
        predictions = []
        for branch_index in range(branch_powers.size(1)):
            predictions.append(self.predict(x_r, x_p, branch_powers[:, branch_index, :]))
        return torch.stack(predictions, dim=1)


class SequenceContextEncoder(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_proj = nn.Linear(3, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden_dim * 2)

    def forward(
        self,
        x_r: torch.Tensor,
        x_p: torch.Tensor,
        p_k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features = build_feature_tensor(x_r, x_p, p_k)
        hidden = F.gelu(self.input_proj(features))
        sequence, _ = self.gru(hidden)
        sequence = self.norm(sequence)
        pooled = sequence.mean(dim=1)
        return sequence, pooled


class CNNBiLSTMModel(BaseBaselineModel):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.rnn = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def predict(self, x_r: torch.Tensor, x_p: torch.Tensor, p_k: torch.Tensor) -> torch.Tensor:
        features = build_feature_tensor(x_r, x_p, p_k).transpose(1, 2)
        hidden = self.conv(features).transpose(1, 2)
        hidden, _ = self.rnn(hidden)
        return self.head(hidden).squeeze(-1)


class DeterministicPhysicsModel(BaseBaselineModel):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = MultiModalFeatureEncoder(
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=4,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def predict(self, x_r: torch.Tensor, x_p: torch.Tensor, p_k: torch.Tensor) -> torch.Tensor:
        condition = self.encoder(x_r, x_p, p_k)
        return self.head(condition).squeeze(-1)

    def predict_all_branches(
        self,
        x_r: torch.Tensor,
        x_p: torch.Tensor,
        branch_powers: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_branches, seq_len = branch_powers.shape
        x_r_expand = x_r.unsqueeze(1).expand(-1, num_branches, -1).reshape(batch_size * num_branches, seq_len)
        x_p_expand = x_p.unsqueeze(1).expand(-1, num_branches, -1).reshape(batch_size * num_branches, seq_len)
        p_k = branch_powers.reshape(batch_size * num_branches, seq_len)
        pred = self.predict(x_r_expand, x_p_expand, p_k)
        return pred.reshape(batch_size, num_branches, seq_len)


class GatedResidualNetwork(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        hidden = F.gelu(self.fc1(x))
        hidden = self.dropout(hidden)
        hidden = self.fc2(hidden)
        gated = torch.sigmoid(self.gate(x))
        return self.norm(residual + gated * hidden)


class TFTLikeModel(BaseBaselineModel):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_proj = nn.Linear(3, hidden_dim)
        self.grn = GatedResidualNetwork(hidden_dim, dropout)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def predict(self, x_r: torch.Tensor, x_p: torch.Tensor, p_k: torch.Tensor) -> torch.Tensor:
        features = build_feature_tensor(x_r, x_p, p_k)
        hidden = self.grn(F.gelu(self.input_proj(features)))
        lstm_out, _ = self.lstm(hidden)
        trans_out = self.transformer(lstm_out)
        gate = torch.sigmoid(self.gate(torch.cat([lstm_out, trans_out], dim=-1)))
        fused = gate * trans_out + (1.0 - gate) * lstm_out
        return self.head(fused).squeeze(-1)


class TimeXerLikeModel(BaseBaselineModel):
    def __init__(self, hidden_dim: int, dropout: float, patch_size: int, seq_len: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.seq_len = seq_len
        self.max_patches = math.ceil(seq_len / patch_size)
        self.patch_proj = nn.Linear(patch_size * 3, hidden_dim)
        self.position = nn.Parameter(torch.zeros(1, self.max_patches, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, patch_size),
        )

    def predict(self, x_r: torch.Tensor, x_p: torch.Tensor, p_k: torch.Tensor) -> torch.Tensor:
        features = build_feature_tensor(x_r, x_p, p_k)
        batch_size, seq_len, _ = features.shape
        padded_len = self.max_patches * self.patch_size
        if padded_len != seq_len:
            pad = padded_len - seq_len
            features = F.pad(features, (0, 0, 0, pad))

        patches = features.reshape(batch_size, self.max_patches, self.patch_size * 3)
        hidden = self.patch_proj(patches) + self.position[:, : self.max_patches]
        hidden = self.transformer(hidden)
        patch_values = self.decoder(hidden)
        output = patch_values.reshape(batch_size, padded_len)
        return output[:, :seq_len]


class CVAEModel(BaseBaselineModel):
    probabilistic = True

    def __init__(self, hidden_dim: int, dropout: float, latent_dim: int) -> None:
        super().__init__()
        self.context_encoder = SequenceContextEncoder(hidden_dim, dropout)
        self.target_encoder = nn.GRU(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.posterior = nn.Linear(hidden_dim * 4, latent_dim * 2)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2 + latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.latent_dim = latent_dim

    def _decode(self, context_sequence: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        latent_expand = latent.unsqueeze(1).expand(-1, context_sequence.size(1), -1)
        return self.decoder(torch.cat([context_sequence, latent_expand], dim=-1)).squeeze(-1)

    def loss_terms(
        self,
        x_r: torch.Tensor,
        x_p: torch.Tensor,
        p_k: torch.Tensor,
        target_y: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        context_sequence, context_pool = self.context_encoder(x_r, x_p, p_k)
        target_sequence, _ = self.target_encoder(target_y.unsqueeze(-1))
        target_pool = target_sequence.mean(dim=1)
        posterior_stats = self.posterior(torch.cat([context_pool, target_pool], dim=-1))
        mu, logvar = posterior_stats.chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)
        latent = mu + torch.randn_like(std) * std

        pred = self._decode(context_sequence, latent)
        recon = F.mse_loss(pred, target_y)
        kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        total = recon + 0.01 * kl
        return {
            "loss": total,
            "recon_loss": recon,
            "kl_loss": kl,
        }

    def predict(self, x_r: torch.Tensor, x_p: torch.Tensor, p_k: torch.Tensor) -> torch.Tensor:
        mean, _ = self.sample(x_r, x_p, p_k, num_samples=8)
        return mean

    def sample(
        self,
        x_r: torch.Tensor,
        x_p: torch.Tensor,
        p_k: torch.Tensor,
        num_samples: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        context_sequence, _ = self.context_encoder(x_r, x_p, p_k)
        samples = []
        for _ in range(num_samples):
            latent = torch.randn(
                x_r.size(0),
                self.latent_dim,
                device=x_r.device,
                dtype=x_r.dtype,
            )
            samples.append(self._decode(context_sequence, latent))
        stacked = torch.stack(samples, dim=0)
        return stacked.mean(dim=0), stacked


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.hidden_dim // 2
        scale = math.log(10000) / max(half_dim - 1, 1)
        freq = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -scale)
        angles = timesteps.float().unsqueeze(-1) * freq.unsqueeze(0)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if emb.size(-1) < self.hidden_dim:
            emb = F.pad(emb, (0, self.hidden_dim - emb.size(-1)))
        return self.proj(emb)


class DiffusionBaselineModel(BaseBaselineModel):
    probabilistic = True

    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        diffusion_steps: int,
        beta_start: float,
        beta_end: float,
    ) -> None:
        super().__init__()
        self.context_encoder = SequenceContextEncoder(hidden_dim, dropout)
        self.time_embedding = SinusoidalTimeEmbedding(hidden_dim)
        self.input_proj = nn.Linear(hidden_dim * 2 + hidden_dim + 1, hidden_dim)
        self.denoiser = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        betas = torch.linspace(beta_start, beta_end, diffusion_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.diffusion_steps = diffusion_steps

    def _predict_noise(
        self,
        noisy_target: torch.Tensor,
        context_sequence: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        time_embed = self.time_embedding(timesteps).unsqueeze(1).expand(-1, noisy_target.size(1), -1)
        fused = torch.cat([context_sequence, time_embed, noisy_target.unsqueeze(-1)], dim=-1)
        hidden = F.gelu(self.input_proj(fused))
        hidden, _ = self.denoiser(hidden)
        return self.output_head(hidden).squeeze(-1)

    def loss_terms(
        self,
        x_r: torch.Tensor,
        x_p: torch.Tensor,
        p_k: torch.Tensor,
        target_y: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        context_sequence, _ = self.context_encoder(x_r, x_p, p_k)
        batch_size = target_y.size(0)
        timesteps = torch.randint(0, self.diffusion_steps, (batch_size,), device=target_y.device)
        noise = torch.randn_like(target_y)
        alpha_bar = self.alpha_bars[timesteps].unsqueeze(-1)
        noisy_target = alpha_bar.sqrt() * target_y + (1.0 - alpha_bar).sqrt() * noise
        pred_noise = self._predict_noise(noisy_target, context_sequence, timesteps)
        diffusion_loss = F.mse_loss(pred_noise, noise)
        return {
            "loss": diffusion_loss,
            "diffusion_loss": diffusion_loss,
        }

    def predict(self, x_r: torch.Tensor, x_p: torch.Tensor, p_k: torch.Tensor) -> torch.Tensor:
        mean, _ = self.sample(x_r, x_p, p_k, num_samples=8)
        return mean

    def sample(
        self,
        x_r: torch.Tensor,
        x_p: torch.Tensor,
        p_k: torch.Tensor,
        num_samples: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        context_sequence, _ = self.context_encoder(x_r, x_p, p_k)
        samples = []
        batch_size, seq_len = x_r.shape
        for _ in range(num_samples):
            current = torch.randn(batch_size, seq_len, device=x_r.device, dtype=x_r.dtype)
            for step in reversed(range(self.diffusion_steps)):
                timesteps = torch.full((batch_size,), step, device=x_r.device, dtype=torch.long)
                pred_noise = self._predict_noise(current, context_sequence, timesteps)
                beta_t = self.betas[step]
                alpha_t = self.alphas[step]
                alpha_bar_t = self.alpha_bars[step]
                mean = (current - (beta_t / (1.0 - alpha_bar_t).sqrt()) * pred_noise) / alpha_t.sqrt()
                if step > 0:
                    current = mean + beta_t.sqrt() * torch.randn_like(current)
                else:
                    current = mean
            samples.append(current)
        stacked = torch.stack(samples, dim=0)
        return stacked.mean(dim=0), stacked


class AffineCoupling(nn.Module):
    def __init__(self, seq_len: int, hidden_dim: int, mask: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = nn.Sequential(
            nn.Linear(seq_len + hidden_dim * 2, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, seq_len * 2),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        masked = x * self.mask
        stats = self.net(torch.cat([masked, context], dim=-1))
        shift, scale = stats.chunk(2, dim=-1)
        scale = 0.8 * torch.tanh(scale)
        inv_mask = 1.0 - self.mask
        z = masked + inv_mask * (x * torch.exp(scale) + shift)
        logdet = (inv_mask * scale).sum(dim=-1)
        return z, logdet

    def inverse(self, z: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        masked = z * self.mask
        stats = self.net(torch.cat([masked, context], dim=-1))
        shift, scale = stats.chunk(2, dim=-1)
        scale = 0.8 * torch.tanh(scale)
        inv_mask = 1.0 - self.mask
        x = masked + inv_mask * ((z - shift) * torch.exp(-scale))
        return x


class CINNModel(BaseBaselineModel):
    probabilistic = True

    def __init__(self, hidden_dim: int, dropout: float, seq_len: int, flow_layers: int) -> None:
        super().__init__()
        self.context_encoder = SequenceContextEncoder(hidden_dim, dropout)
        self.layers = nn.ModuleList()
        for index in range(flow_layers):
            mask_values = torch.tensor(
                [(index + position) % 2 for position in range(seq_len)],
                dtype=torch.float32,
            )
            self.layers.append(AffineCoupling(seq_len, hidden_dim, mask_values))
        self.seq_len = seq_len

    def loss_terms(
        self,
        x_r: torch.Tensor,
        x_p: torch.Tensor,
        p_k: torch.Tensor,
        target_y: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        _, context = self.context_encoder(x_r, x_p, p_k)
        latent = target_y
        logdet = torch.zeros(target_y.size(0), device=target_y.device, dtype=target_y.dtype)
        for layer in self.layers:
            latent, layer_logdet = layer(latent, context)
            logdet = logdet + layer_logdet

        nll = 0.5 * (latent.pow(2) + math.log(2.0 * math.pi)).sum(dim=-1) - logdet
        loss = nll.mean()
        return {
            "loss": loss,
            "nll_loss": loss,
        }

    def predict(self, x_r: torch.Tensor, x_p: torch.Tensor, p_k: torch.Tensor) -> torch.Tensor:
        mean, _ = self.sample(x_r, x_p, p_k, num_samples=8)
        return mean

    def sample(
        self,
        x_r: torch.Tensor,
        x_p: torch.Tensor,
        p_k: torch.Tensor,
        num_samples: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, context = self.context_encoder(x_r, x_p, p_k)
        samples = []
        for _ in range(num_samples):
            latent = torch.randn(
                x_r.size(0),
                self.seq_len,
                device=x_r.device,
                dtype=x_r.dtype,
            )
            current = latent
            for layer in reversed(self.layers):
                current = layer.inverse(current, context)
            samples.append(current)
        stacked = torch.stack(samples, dim=0)
        return stacked.mean(dim=0), stacked


def build_model(method_name: str, args) -> BaseBaselineModel:
    if method_name == "cnn_bilstm":
        return CNNBiLSTMModel(hidden_dim=args.hidden_dim, dropout=args.dropout)
    if method_name == "deterministic_physics":
        return DeterministicPhysicsModel(hidden_dim=args.hidden_dim, dropout=args.dropout)
    if method_name == "tft":
        return TFTLikeModel(hidden_dim=args.hidden_dim, dropout=args.dropout)
    if method_name == "timexer":
        return TimeXerLikeModel(
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            patch_size=args.patch_size,
            seq_len=args.seq_len,
        )
    if method_name == "cvae":
        return CVAEModel(hidden_dim=args.hidden_dim, dropout=args.dropout, latent_dim=args.latent_dim)
    if method_name == "diffusion":
        return DiffusionBaselineModel(
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            diffusion_steps=args.diffusion_steps,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
        )
    if method_name == "cinn":
        return CINNModel(
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            seq_len=args.seq_len,
            flow_layers=args.flow_layers,
        )
    raise ValueError(f"Unsupported method: {method_name}")
