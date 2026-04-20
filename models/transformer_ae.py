from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Classic sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)          # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)                   # [1, max_len, d_model]
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d_model]
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerAutoencoder(nn.Module):
    """Transformer-based denoising autoencoder for 1-D signals.

    Architecture overview
    ---------------------
    Encoder
      1. Linear projection: 1 → d_model
      2. Positional encoding
      3. N × TransformerEncoderLayer (self-attention over timesteps)
      4. Flatten + linear projection to one global bottleneck vector

    Bottleneck
      A single compressed latent representation of shape [B, bottleneck_dim].

    Decoder
      1. Linear projection: bottleneck_dim → L × d_model
      2. Positional encoding
      3. N × TransformerEncoderLayer (self-attention — acts as the "decoder" over
         the latent sequence; a full cross-attention decoder is not needed here
         because we are reconstructing the same-length sequence)
      4. Linear projection: d_model → 1

    Input / output shape: [B, 1, L]  (same as CNN and LSTM autoencoders).
    """

    def __init__(
        self,
        sequence_length: int,
        bottleneck_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # d_model must be divisible by num_heads
        d_model = hidden_dim
        if d_model % num_heads != 0:
            # Round up to the nearest multiple of num_heads
            d_model = math.ceil(d_model / num_heads) * num_heads

        self.sequence_length = sequence_length
        self.d_model = d_model

        # ---- Encoder ----
        self.enc_input_proj = nn.Linear(1, d_model)
        self.enc_pos_enc = PositionalEncoding(d_model, max_len=sequence_length + 1, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,          # Pre-norm: more stable training
        )
        self.encoder_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.to_latent = nn.Linear(sequence_length * d_model, bottleneck_dim)

        # ---- Decoder ----
        self.from_latent = nn.Linear(bottleneck_dim, sequence_length * d_model)
        self.dec_pos_enc = PositionalEncoding(d_model, max_len=sequence_length + 1, dropout=dropout)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder_transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape [B, 1, L]

        Returns
        -------
        Tensor, shape [B, 1, L]
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3-D input [B, 1, L], got {x.dim()}-D.")

        # [B, 1, L] → [B, L, 1]
        seq = x.transpose(1, 2)

        # Encoder
        enc = self.enc_pos_enc(self.enc_input_proj(seq))   # [B, L, d_model]
        enc = self.encoder_transformer(enc)                # [B, L, d_model]
        latent = self.to_latent(enc.flatten(start_dim=1))  # [B, bottleneck_dim]

        # Decoder
        dec = self.from_latent(latent).view(-1, self.sequence_length, self.d_model)
        dec = self.dec_pos_enc(dec)                        # [B, L, d_model]
        dec = self.decoder_transformer(dec)                # [B, L, d_model]
        out = self.output_proj(dec)                        # [B, L, 1]

        return out.transpose(1, 2)                         # [B, 1, L]
