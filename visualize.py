from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    DEFAULT_GAUSSIAN_STD,
    DEFAULT_IMPULSE_AMPLITUDE,
    DEFAULT_IMPULSE_RATE,
    DEFAULT_INTERFERENCE_AMPLITUDE,
    DEFAULT_INTERFERENCE_FREQ,
    DEFAULT_MASK_MAX_LENGTH,
    DEFAULT_MASK_MIN_LENGTH,
    DEFAULT_MASK_RATIO,
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_SEED,
)
from data.dataset import NoiseConfig, SyntheticDenoisingDataset
from evaluate import load_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize denoising model reconstructions.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a saved checkpoint.")
    parser.add_argument("--num-examples", type=int, default=3)
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
    parser.add_argument("--output", type=str, default="artifacts/reconstructions.png")
    parser.add_argument(
        "--plot-history",
        action="store_true",
        help="Plot train vs val MSE over epochs and save alongside --output.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def plot_training_history(checkpoint_path: str, output_path: str) -> None:
    """Load training history from a checkpoint and plot train vs val MSE."""
    import torch as _torch
    ckpt = _torch.load(checkpoint_path, map_location="cpu")
    history = ckpt.get("history", [])
    if not history:
        print("No training history found in checkpoint; skipping history plot.")
        return

    epochs = [h["epoch"] for h in history]
    train_mse = [h["train_mse"] for h in history]
    val_mse = [h["mse"] for h in history]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_mse, label="Train MSE", linewidth=2, color="#4C72B0")
    ax.plot(epochs, val_mse, label="Val MSE", linewidth=2, linestyle="--", color="#DD8452")
    best_epoch = epochs[int(min(range(len(val_mse)), key=lambda i: val_mse[i]))]
    ax.axvline(best_epoch, color="#55A868", linewidth=1.2, linestyle=":", label=f"Best epoch ({best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Training Curve: MSE over Epochs")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = Path(output_path)
    history_out = out.parent / (out.stem + "_training_curve.png")
    history_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(history_out, dpi=200)
    plt.close(fig)
    print(f"Saved training curve to {history_out}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    device = torch.device(args.device)

    model, train_config = load_checkpoint(args.checkpoint, device=device)
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
        num_samples=args.num_examples,
        sequence_length=args.sequence_length or train_config["sequence_length"],
        noise_config=noise_config,
        seed=args.seed,
    )

    noisy = dataset.noisy_signals.to(device)
    clean = dataset.clean_signals.to(device)
    with torch.no_grad():
        reconstructed = model(noisy).cpu()

    time_axis = range(clean.size(-1))
    fig, axes = plt.subplots(args.num_examples, 1, figsize=(10, 3 * args.num_examples), sharex=True)
    if args.num_examples == 1:
        axes = [axes]

    for idx, axis in enumerate(axes):
        axis.plot(time_axis, clean[idx, 0].cpu().numpy(), label="Clean", linewidth=2)
        axis.plot(time_axis, noisy[idx, 0].cpu().numpy(), label="Noisy", alpha=0.75)
        axis.plot(time_axis, reconstructed[idx, 0].numpy(), label="Reconstructed", linestyle="--")
        axis.set_title(f"Example {idx + 1}")
        axis.set_ylabel("Amplitude")
        axis.grid(alpha=0.3)
        axis.legend(loc="upper right")

    axes[-1].set_xlabel("Time Step")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved visualization to {output_path}")

    if args.plot_history:
        plot_training_history(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
