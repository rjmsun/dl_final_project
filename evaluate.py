"""
Evaluate a trained denoising autoencoder on synthetic 1D signals.
Calculates metrics like MSE and SNR improvement.
I used AI here to help with syntax and also understand the format of this file.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_BOTTLENECK_DIM,
    DEFAULT_GAUSSIAN_STD,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_IMPULSE_AMPLITUDE,
    DEFAULT_IMPULSE_RATE,
    DEFAULT_INTERFERENCE_AMPLITUDE,
    DEFAULT_INTERFERENCE_FREQ,
    DEFAULT_MASK_MAX_LENGTH,
    DEFAULT_MASK_MIN_LENGTH,
    DEFAULT_MASK_RATIO,
    DEFAULT_NUM_LAYERS,
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_TEST_SAMPLES,
    DEFAULT_SEED,
)
from data.dataset import NoiseConfig, SyntheticDenoisingDataset
from models import build_model


def snr_db(reference: torch.Tensor, estimate: torch.Tensor) -> torch.Tensor:
    signal_power = torch.sum(reference**2, dim=(-1, -2))
    noise_power = torch.sum((reference - estimate) ** 2, dim=(-1, -2)).clamp_min(1e-8)
    return 10.0 * torch.log10(signal_power.clamp_min(1e-8) / noise_power)


def evaluate_loader(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> dict[str, float]:
    model.eval()
    criterion = criterion or nn.MSELoss()

    total_loss = 0.0
    total_noisy_snr = 0.0
    total_recon_snr = 0.0
    total_samples = 0

    with torch.no_grad():
        for noisy, clean in data_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            reconstructed = model(noisy)
            batch_size = clean.size(0)

            total_loss += criterion(reconstructed, clean).item() * batch_size
            total_noisy_snr += snr_db(clean, noisy).sum().item()
            total_recon_snr += snr_db(clean, reconstructed).sum().item()
            total_samples += batch_size

    mse = total_loss / total_samples
    noisy_snr = total_noisy_snr / total_samples
    recon_snr = total_recon_snr / total_samples
    return {
        "mse": mse,
        "noisy_snr_db": noisy_snr,
        "recon_snr_db": recon_snr,
        "snr_improvement_db": recon_snr - noisy_snr,
    }


def load_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = build_model(
        model_name=config["model"],
        sequence_length=config["sequence_length"],
        bottleneck_dim=config["bottleneck_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a denoising autoencoder checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a saved checkpoint.")
    parser.add_argument("--num-samples", type=int, default=DEFAULT_TEST_SAMPLES)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--noise-mode", choices=["gaussian", "masking", "both", "impulse", "sinusoidal", "all"], default="both")
    parser.add_argument("--gaussian-std", type=float, default=DEFAULT_GAUSSIAN_STD)
    parser.add_argument("--mask-ratio", type=float, default=DEFAULT_MASK_RATIO)
    parser.add_argument("--mask-min-length", type=int, default=DEFAULT_MASK_MIN_LENGTH)
    parser.add_argument("--mask-max-length", type=int, default=DEFAULT_MASK_MAX_LENGTH)
    parser.add_argument("--impulse-rate", type=float, default=DEFAULT_IMPULSE_RATE)
    parser.add_argument("--impulse-amplitude", type=float, default=DEFAULT_IMPULSE_AMPLITUDE)
    parser.add_argument("--interference-freq", type=float, default=DEFAULT_INTERFERENCE_FREQ)
    parser.add_argument("--interference-amplitude", type=float, default=DEFAULT_INTERFERENCE_AMPLITUDE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    device = torch.device(args.device)

    model, train_config = load_checkpoint(args.checkpoint, device=device)
    sequence_length = args.sequence_length or train_config["sequence_length"]

    noise_config = NoiseConfig(
        noise_mode=args.noise_mode,
        gaussian_std=args.gaussian_std,
        mask_ratio=args.mask_ratio,
        mask_min_length=args.mask_min_length,
        mask_max_length=args.mask_max_length,
        impulse_rate=args.impulse_rate,
        impulse_amplitude=args.impulse_amplitude,
        interference_freq=args.interference_freq,
        interference_amplitude=args.interference_amplitude,
    )
    dataset = SyntheticDenoisingDataset(
        num_samples=args.num_samples,
        sequence_length=sequence_length,
        noise_config=noise_config,
        seed=args.seed,
    )
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    metrics = evaluate_loader(model=model, data_loader=data_loader, device=device)

    print(f"Checkpoint: {args.checkpoint}")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")


if __name__ == "__main__":
    main()
