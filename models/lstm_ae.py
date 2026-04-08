from __future__ import annotations

import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        bottleneck_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.bottleneck_dim = bottleneck_dim

        self.encoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.to_latent = nn.Linear(hidden_dim, bottleneck_dim)
        self.decoder = nn.LSTM(
            input_size=bottleneck_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("Expected input shape [batch, channels, length].")

        sequence = x.transpose(1, 2)
        _, (hidden_state, _) = self.encoder(sequence)
        latent = self.to_latent(hidden_state[-1])
        repeated_latent = latent.unsqueeze(1).repeat(1, self.sequence_length, 1)
        decoded_sequence, _ = self.decoder(repeated_latent)
        reconstruction = self.output_layer(decoded_sequence)
        return reconstruction.transpose(1, 2)
