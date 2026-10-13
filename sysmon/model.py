"""LSTM autoencoder.

The model is trained only on *normal* behaviour and learns to reconstruct a
window of metrics. At inference time, windows it reconstructs badly are ones
that look unlike anything it saw in training -> anomaly.

    [B, T, F] --encoder LSTM--> h_T --Linear--> z [B, latent]
    z --Linear, repeat T times--> decoder LSTM --Linear--> [B, T, F]
"""

from __future__ import annotations

import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, latent: int = 16, layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.config = dict(n_features=n_features, hidden=hidden, latent=latent, layers=layers, dropout=dropout)
        lstm_dropout = dropout if layers > 1 else 0.0
        self.encoder = nn.LSTM(n_features, hidden, layers, batch_first=True, dropout=lstm_dropout)
        self.to_latent = nn.Linear(hidden, latent)
        self.from_latent = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(hidden, hidden, layers, batch_first=True, dropout=lstm_dropout)
        self.head = nn.Linear(hidden, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.encoder(x)
        z = self.to_latent(h[-1])  # bottleneck: [B, latent]
        seed = self.from_latent(z).unsqueeze(1).expand(-1, x.size(1), -1)
        out, _ = self.decoder(seed)
        return self.head(out)

    @torch.no_grad()
    def feature_errors(self, x: torch.Tensor) -> torch.Tensor:
        """Per-window, per-feature mean squared reconstruction error: [B, F]."""
        return ((self(x) - x) ** 2).mean(dim=1)
