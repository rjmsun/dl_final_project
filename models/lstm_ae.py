from __future__ import annotations

import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
    """Denoising autoencoder with a bidirectional LSTM encoder and a unidirectional
    LSTM decoder, connected through a per-timestep feature bottleneck.

    The encoder reads the noisy sequence in both directions, then a linear
    projection compresses each timestep's features to ``bottleneck_dim``.  The
    decoder LSTM reads those compressed features and produces one clean sample per
    timestep.  Because every output step has a direct gradient path through the
    bottleneck to the corresponding encoder step, gradients flow well and the
    model converges quickly.
    """

    def __init__(
        self,
        sequence_length: int,
        bottleneck_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
    ) -> None:
        super().__init__()

        # Bidirectional encoder — each timestep gets 2 * hidden_dim features.
        self.encoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        encoder_out_dim = hidden_dim * 2
        # Per-timestep compression to the bottleneck.
        self.to_latent = nn.Linear(encoder_out_dim, bottleneck_dim)

        # Decoder reads the compressed sequence and reconstructs the signal.
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

        sequence = x.transpose(1, 2)              # [B, L, 1]
        enc_out, _ = self.encoder(sequence)        # [B, L, 2*hidden_dim]
        latent_seq = self.to_latent(enc_out)       # [B, L, bottleneck_dim]
        dec_out, _ = self.decoder(latent_seq)      # [B, L, hidden_dim]
        reconstruction = self.output_layer(dec_out)  # [B, L, 1]
        return reconstruction.transpose(1, 2)      # [B, 1, L]
