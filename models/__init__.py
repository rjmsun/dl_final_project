from .cnn_ae import CNNAutoencoder
from .lstm_ae import LSTMAutoencoder
from .mlp_ae import MLPAutoencoder
from .transformer_ae import TransformerAutoencoder


def build_model(
    model_name: str,
    sequence_length: int,
    bottleneck_dim: int,
    hidden_dim: int = 128,
    num_layers: int = 1,
):
    if model_name == "mlp":
        return MLPAutoencoder(
            sequence_length=sequence_length,
            bottleneck_dim=bottleneck_dim,
            hidden_dim=hidden_dim,
        )
    if model_name == "cnn":
        return CNNAutoencoder(
            sequence_length=sequence_length,
            bottleneck_dim=bottleneck_dim,
            base_channels=16,
        )
    if model_name == "lstm":
        return LSTMAutoencoder(
            sequence_length=sequence_length,
            bottleneck_dim=bottleneck_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
    if model_name == "transformer":
        return TransformerAutoencoder(
            sequence_length=sequence_length,
            bottleneck_dim=bottleneck_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
    raise ValueError(f"Unsupported model '{model_name}'. Choose from: mlp, cnn, lstm, transformer.")


__all__ = ["MLPAutoencoder", "CNNAutoencoder", "LSTMAutoencoder", "TransformerAutoencoder", "build_model"]
