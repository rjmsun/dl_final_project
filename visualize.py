from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    DEFAULT_GAUSSIAN_STD,
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
    parser.add_argument("--noise-mode", choices=["gaussian", "masking", "both"], default="both")
    parser.add_argument("--gaussian-std", type=float, default=DEFAULT_GAUSSIAN_STD)
    parser.add_argument("--mask-ratio", type=float, default=DEFAULT_MASK_RATIO)
    parser.add_argument("--mask-min-length", type=int, default=DEFAULT_MASK_MIN_LENGTH)
    parser.add_argument("--mask-max-length", type=int, default=DEFAULT_MASK_MAX_LENGTH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=str, default="artifacts/reconstructions.png")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


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


if __name__ == "__main__":
    main()
