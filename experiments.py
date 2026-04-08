from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path

import torch
from torch.utils.data import DataLoader

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
    DEFAULT_TEST_SAMPLES,
    DEFAULT_TRAIN_SAMPLES,
    DEFAULT_VAL_SAMPLES,
)
from data.dataset import NoiseConfig, SyntheticDenoisingDataset
from evaluate import evaluate_loader, load_checkpoint
from train import build_training_namespace, train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run architecture comparison experiments.")
    parser.add_argument("--models", nargs="+", choices=["mlp", "cnn", "lstm"], default=["mlp", "cnn", "lstm"])
    parser.add_argument("--bottleneck-dims", nargs="+", type=int, default=[DEFAULT_BOTTLENECK_DIM])
    parser.add_argument("--noise-modes", nargs="+", choices=["gaussian", "masking", "both"], default=["both"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--train-samples", type=int, default=DEFAULT_TRAIN_SAMPLES)
    parser.add_argument("--val-samples", type=int, default=DEFAULT_VAL_SAMPLES)
    parser.add_argument("--test-samples", type=int, default=DEFAULT_TEST_SAMPLES)
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--gaussian-std", type=float, default=DEFAULT_GAUSSIAN_STD)
    parser.add_argument("--mask-ratio", type=float, default=DEFAULT_MASK_RATIO)
    parser.add_argument("--mask-min-length", type=int, default=DEFAULT_MASK_MIN_LENGTH)
    parser.add_argument("--mask-max-length", type=int, default=DEFAULT_MASK_MAX_LENGTH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=str, default="experiments")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def evaluate_checkpoint(
    checkpoint_path: str,
    sequence_length: int,
    num_samples: int,
    batch_size: int,
    noise_config: NoiseConfig,
    seed: int,
    device: str,
) -> dict[str, float]:
    model, _ = load_checkpoint(checkpoint_path, device=torch.device(device))
    dataset = SyntheticDenoisingDataset(
        num_samples=num_samples,
        sequence_length=sequence_length,
        noise_config=noise_config,
        seed=seed,
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return evaluate_loader(model=model, data_loader=data_loader, device=torch.device(device))


def write_results_csv(results: list[dict[str, object]], output_path: Path) -> None:
    if not results:
        return
    fieldnames = list(results[0].keys())
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    experiment_grid = list(product(args.models, args.bottleneck_dims, args.noise_modes))
    total_runs = len(experiment_grid)

    for run_index, (model_name, bottleneck_dim, noise_mode) in enumerate(experiment_grid, start=1):
        run_name = f"{model_name}_z{bottleneck_dim}_{noise_mode}"
        print(f"[{run_index}/{total_runs}] Running {run_name}")

        train_args = build_training_namespace(
            model=model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            train_samples=args.train_samples,
            val_samples=args.val_samples,
            sequence_length=args.sequence_length,
            bottleneck_dim=bottleneck_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            noise_mode=noise_mode,
            gaussian_std=args.gaussian_std,
            mask_ratio=args.mask_ratio,
            mask_min_length=args.mask_min_length,
            mask_max_length=args.mask_max_length,
            seed=args.seed + run_index - 1,
            save_dir=str(checkpoint_dir),
            run_name=run_name,
            device=args.device,
        )
        train_summary = train_model(train_args, verbose=True)

        noise_config = NoiseConfig(
            noise_mode=noise_mode,
            gaussian_std=args.gaussian_std,
            mask_ratio=args.mask_ratio,
            mask_min_length=args.mask_min_length,
            mask_max_length=args.mask_max_length,
        )
        test_metrics = evaluate_checkpoint(
            checkpoint_path=train_summary["checkpoint_path"],
            sequence_length=args.sequence_length,
            num_samples=args.test_samples,
            batch_size=args.batch_size,
            noise_config=noise_config,
            seed=args.seed + 10_000 + run_index - 1,
            device=args.device,
        )

        result = {
            "run_name": run_name,
            "model": model_name,
            "bottleneck_dim": bottleneck_dim,
            "noise_mode": noise_mode,
            "epochs": args.epochs,
            "train_samples": args.train_samples,
            "val_samples": args.val_samples,
            "test_samples": args.test_samples,
            "checkpoint_path": train_summary["checkpoint_path"],
            "best_val_mse": train_summary["best_val_mse"],
            "test_mse": test_metrics["mse"],
            "test_noisy_snr_db": test_metrics["noisy_snr_db"],
            "test_recon_snr_db": test_metrics["recon_snr_db"],
            "test_snr_improvement_db": test_metrics["snr_improvement_db"],
        }
        results.append(result)
        print(
            f"Completed {run_name} | "
            f"test_mse={result['test_mse']:.4f} | "
            f"test_snr_improvement_db={result['test_snr_improvement_db']:.4f}"
        )

    csv_path = output_dir / "results.csv"
    json_path = output_dir / "results.json"
    write_results_csv(results, csv_path)
    json_path.write_text(json.dumps(results, indent=2))
    print(f"Saved results to {csv_path}")
    print(f"Saved results to {json_path}")


if __name__ == "__main__":
    main()
