from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_BOTTLENECK_DIM,
    DEFAULT_EPOCHS,
    DEFAULT_GAUSSIAN_STD,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MASK_MAX_LENGTH,
    DEFAULT_MASK_MIN_LENGTH,
    DEFAULT_MASK_RATIO,
    DEFAULT_NUM_LAYERS,
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_SEED,
    DEFAULT_TRAIN_SAMPLES,
    DEFAULT_VAL_SAMPLES,
)
from data.dataset import NoiseConfig, SyntheticDenoisingDataset
from evaluate import evaluate_loader
from models import build_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a denoising autoencoder on synthetic 1D signals.")
    parser.add_argument("--model", choices=["mlp", "cnn", "lstm"], default="cnn")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--train-samples", type=int, default=DEFAULT_TRAIN_SAMPLES)
    parser.add_argument("--val-samples", type=int, default=DEFAULT_VAL_SAMPLES)
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--bottleneck-dim", type=int, default=DEFAULT_BOTTLENECK_DIM)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--noise-mode", choices=["gaussian", "masking", "both"], default="both")
    parser.add_argument("--gaussian-std", type=float, default=DEFAULT_GAUSSIAN_STD)
    parser.add_argument("--mask-ratio", type=float, default=DEFAULT_MASK_RATIO)
    parser.add_argument("--mask-min-length", type=int, default=DEFAULT_MASK_MIN_LENGTH)
    parser.add_argument("--mask-max-length", type=int, default=DEFAULT_MASK_MAX_LENGTH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def build_noise_config(args: argparse.Namespace) -> NoiseConfig:
    return NoiseConfig(
        noise_mode=args.noise_mode,
        gaussian_std=args.gaussian_std,
        mask_ratio=args.mask_ratio,
        mask_min_length=args.mask_min_length,
        mask_max_length=args.mask_max_length,
    )


def build_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    noise_config = build_noise_config(args)
    train_dataset = SyntheticDenoisingDataset(
        num_samples=args.train_samples,
        sequence_length=args.sequence_length,
        noise_config=noise_config,
        seed=args.seed,
    )
    val_dataset = SyntheticDenoisingDataset(
        num_samples=args.val_samples,
        sequence_length=args.sequence_length,
        noise_config=noise_config,
        seed=args.seed + 1,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    return train_loader, val_loader


def build_training_namespace(**overrides) -> argparse.Namespace:
    defaults = vars(build_parser().parse_args([]))
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    show_progress: bool = True,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    iterator = tqdm(data_loader, desc="Training", leave=False, disable=not show_progress)
    for noisy, clean in iterator:
        noisy = noisy.to(device)
        clean = clean.to(device)

        optimizer.zero_grad()
        reconstructed = model(noisy)
        loss = criterion(reconstructed, clean)
        loss.backward()
        optimizer.step()

        batch_size = clean.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def train_model(args: argparse.Namespace, verbose: bool = True) -> dict[str, object]:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    train_loader, val_loader = build_dataloaders(args)
    model = build_model(
        model_name=args.model,
        sequence_length=args.sequence_length,
        bottleneck_dim=args.bottleneck_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_stem = args.run_name or args.model
    checkpoint_path = save_dir / f"{checkpoint_stem}_best.pt"

    best_val_mse = float("inf")
    best_val_metrics: dict[str, float] | None = None
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        train_mse = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            show_progress=verbose,
        )
        val_metrics = evaluate_loader(model=model, data_loader=val_loader, device=device, criterion=criterion)
        epoch_summary = {
            "epoch": epoch,
            "train_mse": train_mse,
            **val_metrics,
        }
        history.append(epoch_summary)

        if verbose:
            print(
                f"Epoch {epoch:02d} | "
                f"train_mse={train_mse:.4f} | "
                f"val_mse={val_metrics['mse']:.4f} | "
                f"val_snr_improvement_db={val_metrics['snr_improvement_db']:.4f}"
            )

        if val_metrics["mse"] < best_val_mse:
            best_val_mse = val_metrics["mse"]
            best_val_metrics = val_metrics
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": vars(args),
                    "best_val_mse": best_val_mse,
                    "best_val_metrics": best_val_metrics,
                    "history": history,
                },
                checkpoint_path,
            )

    if verbose:
        print(f"Best checkpoint saved to {checkpoint_path}")

    return {
        "checkpoint_path": str(checkpoint_path),
        "best_val_mse": best_val_mse,
        "best_val_metrics": best_val_metrics or {},
        "history": history,
    }


def main() -> None:
    args = build_parser().parse_args()
    train_model(args, verbose=True)


if __name__ == "__main__":
    main()
