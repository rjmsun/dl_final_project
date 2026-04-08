from __future__ import annotations

import torch
from torch import nn


class MLPAutoencoder(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        bottleneck_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length

        self.encoder = nn.Sequential(
            nn.Linear(sequence_length, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, sequence_length),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction.unsqueeze(1)
