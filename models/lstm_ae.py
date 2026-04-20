from __future__ import annotations

import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
    """Denoising autoencoder with a bidirectional LSTM encoder and global latent.

    The encoder maps the whole noisy sequence into recurrent features, compresses
    them into one vector of size ``bottleneck_dim``, then a feed-forward decoder
    expands that temporal summary back into a full-length clean signal.
    """

    def __init__(
        self,
        sequence_length: int,
        bottleneck_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.to_latent = nn.Linear(sequence_length * hidden_dim * 2, bottleneck_dim)

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, sequence_length),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("Expected input shape [batch, channels, length].")

        sequence = x.transpose(1, 2)              # [B, L, 1]
        enc_out, _ = self.encoder(sequence)        # [B, L, 2*hidden_dim]
        latent = self.to_latent(enc_out.flatten(start_dim=1))  # [B, bottleneck_dim]
        reconstruction = self.decoder(latent)       # [B, L]
        return reconstruction.unsqueeze(1)          # [B, 1, L]
