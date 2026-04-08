from __future__ import annotations

import torch
from torch import nn


class CNNAutoencoder(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        bottleneck_dim: int,
        base_channels: int = 16,
    ) -> None:
        super().__init__()
        if sequence_length % 4 != 0:
            raise ValueError("sequence_length must be divisible by 4 for the CNN autoencoder.")

        self.sequence_length = sequence_length
        self.encoded_length = sequence_length // 4
        self.final_channels = base_channels * 4

        self.encoder_conv = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(base_channels * 2, self.final_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.to_latent = nn.Linear(self.final_channels * self.encoded_length, bottleneck_dim)
        self.from_latent = nn.Linear(bottleneck_dim, self.final_channels * self.encoded_length)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(
                self.final_channels,
                base_channels * 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                base_channels * 2,
                base_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv1d(base_channels, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder_conv(x)
        latent = self.to_latent(encoded.flatten(start_dim=1))
        decoded = self.from_latent(latent).view(-1, self.final_channels, self.encoded_length)
        return self.decoder_conv(decoded)
