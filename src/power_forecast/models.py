from __future__ import annotations

import math

import torch
from torch import nn


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        horizon: int,
        hidden_size: int = 48,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(output[:, -1])


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerForecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        horizon: int,
        d_model: int = 48,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position = SinusoidalPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.position(self.input_projection(x)))
        pooled = encoded.mean(dim=1)
        return self.head(pooled)


class PatchChannelMixerForecaster(nn.Module):
    """Patch-time, inverted-channel, and multiscale-stat fusion forecaster."""

    def __init__(
        self,
        input_dim: int,
        horizon: int,
        input_days: int = 90,
        d_model: int = 48,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
        patch_len: int = 14,
        patch_stride: int = 7,
        target_index: int = 0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.horizon = horizon
        self.input_days = input_days
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.target_index = target_index

        self.patch_projection = nn.Linear(input_dim * patch_len, d_model)
        max_patches = max(1, 1 + (input_days - patch_len) // patch_stride)
        self.patch_position = nn.Parameter(torch.randn(max_patches, d_model) * 0.02)
        patch_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.patch_encoder = nn.TransformerEncoder(patch_layer, num_layers=num_layers)

        self.channel_projection = nn.Linear(input_days, d_model)
        self.channel_identity = nn.Parameter(torch.randn(input_dim, d_model) * 0.02)
        channel_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.channel_encoder = nn.TransformerEncoder(channel_layer, num_layers=num_layers)

        self.stat_projection = nn.Sequential(
            nn.Linear(12, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
        )
        self.forecast_head = nn.Linear(d_model, horizon)
        self.stat_head = nn.Linear(d_model, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patch_repr = self._patch_branch(x)
        channel_repr = self._channel_branch(x)
        stat_repr = self.stat_projection(self._multiscale_stats(x))
        fused = self.fusion(torch.cat([patch_repr, channel_repr, stat_repr], dim=-1))
        return self.forecast_head(fused) + 0.2 * self.stat_head(stat_repr)

    def _patch_branch(self, x: torch.Tensor) -> torch.Tensor:
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.patch_stride)
        patches = patches.permute(0, 1, 3, 2).contiguous()
        tokens = patches.view(x.size(0), patches.size(1), self.patch_len * self.input_dim)
        tokens = self.patch_projection(tokens)
        tokens = tokens + self.patch_position[: tokens.size(1)].unsqueeze(0)
        encoded = self.patch_encoder(tokens)
        return encoded.mean(dim=1)

    def _channel_branch(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.channel_projection(x.transpose(1, 2))
        tokens = tokens + self.channel_identity.unsqueeze(0)
        encoded = self.channel_encoder(tokens)
        return encoded.mean(dim=1)

    def _multiscale_stats(self, x: torch.Tensor) -> torch.Tensor:
        target = x[:, :, self.target_index]
        stats: list[torch.Tensor] = []
        for window in (7, 30, 90):
            recent = target[:, -min(window, target.size(1)) :]
            last = recent[:, -1]
            mean = recent.mean(dim=1)
            std = recent.std(dim=1, unbiased=False)
            if recent.size(1) > 1:
                slope = torch.diff(recent, dim=1).mean(dim=1)
            else:
                slope = torch.zeros_like(last)
            stats.extend([last, mean, std, slope])
        return torch.stack(stats, dim=1)


def build_model(
    model_name: str,
    input_dim: int,
    horizon: int,
    input_days: int = 90,
    d_model: int = 48,
    hidden_size: int = 48,
    num_layers: int = 2,
    num_heads: int = 4,
    dropout: float = 0.1,
) -> nn.Module:
    normalized = model_name.lower().replace("-", "_")
    if normalized == "lstm":
        return LSTMForecaster(
            input_dim=input_dim,
            horizon=horizon,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
    if normalized == "transformer":
        return TransformerForecaster(
            input_dim=input_dim,
            horizon=horizon,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
    if normalized in {"trend_conv_transformer", "proposed", "improved", "patch_channel_mixer", "patchchannelmixer"}:
        return PatchChannelMixerForecaster(
            input_dim=input_dim,
            horizon=horizon,
            input_days=input_days,
            d_model=d_model,
            num_layers=max(1, num_layers),
            num_heads=num_heads,
            dropout=dropout,
        )
    raise ValueError(f"Unknown model_name: {model_name}")
