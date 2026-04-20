"""experiments.py — architecture comparison + noise-generalisation sweep.

Main sweep
----------
Trains every combination of (model × bottleneck_dim × noise_mode) and writes:
  - experiments/results.csv
  - experiments/results.json

Noise-generalisation sweep (--noise-generalization)
----------------------------------------------------
After the main sweep, takes every trained checkpoint and evaluates it at a
range of Gaussian noise intensities (regardless of which noise type it was
trained on).  This answers the question: "does a model trained at σ=0.15
generalise to σ=0.05 or σ=0.40?"  Results go to:
  - experiments/generalization.csv
  - experiments/generalization.json
"""
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
    DEFAULT_IMPULSE_AMPLITUDE,
    DEFAULT_IMPULSE_RATE,
    DEFAULT_INTERFERENCE_AMPLITUDE,
    DEFAULT_INTERFERENCE_FREQ,
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

# Noise levels used in the generalisation sweep
GENERALISATION_STD_LEVELS = [0.05, 0.10, 0.15, 0.25, 0.40]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run architecture comparison experiments.")
    parser.add_argument(
        "--models", nargs="+",
        choices=["mlp", "cnn", "lstm", "transformer"],
        default=["mlp", "cnn", "lstm"],
    )
    parser.add_argument("--bottleneck-dims", nargs="+", type=int, default=[DEFAULT_BOTTLENECK_DIM])
    parser.add_argument(
        "--noise-modes", nargs="+",
        choices=["gaussian", "masking", "both", "impulse", "sinusoidal", "all"],
        default=["both"],
    )
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
    parser.add_argument("--impulse-rate", type=float, default=DEFAULT_IMPULSE_RATE)
    parser.add_argument("--impulse-amplitude", type=float, default=DEFAULT_IMPULSE_AMPLITUDE)
    parser.add_argument("--interference-freq", type=float, default=DEFAULT_INTERFERENCE_FREQ)
    parser.add_argument("--interference-amplitude", type=float, default=DEFAULT_INTERFERENCE_AMPLITUDE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=str, default="experiments")
    parser.add_argument(
        "--noise-generalization",
        action="store_true",
        help="After the main sweep, evaluate every checkpoint across multiple "
             "Gaussian noise intensities to measure out-of-distribution robustness.",
    )
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


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_main_sweep(args: argparse.Namespace, checkpoint_dir: Path) -> list[dict[str, object]]:
    """Train + evaluate every (model × bottleneck × noise_mode) combination."""
    results: list[dict[str, object]] = []
    experiment_grid = list(product(args.models, args.bottleneck_dims, args.noise_modes))
    total_runs = len(experiment_grid)

    for run_index, (model_name, bottleneck_dim, noise_mode) in enumerate(experiment_grid, start=1):
        run_name = f"{model_name}_z{bottleneck_dim}_{noise_mode}"
        print(f"\n[{run_index}/{total_runs}] Running {run_name}")

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
            impulse_rate=args.impulse_rate,
            impulse_amplitude=args.impulse_amplitude,
            interference_freq=args.interference_freq,
            interference_amplitude=args.interference_amplitude,
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
            impulse_rate=args.impulse_rate,
            impulse_amplitude=args.impulse_amplitude,
            interference_freq=args.interference_freq,
            interference_amplitude=args.interference_amplitude,
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

    return results


def run_noise_generalization_sweep(
    results: list[dict[str, object]],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    """Evaluate every trained checkpoint at several Gaussian noise intensities.

    This tests whether each architecture generalises to noise levels it was not
    trained on — a key experiment mentioned in the project brief.
    """
    gen_rows: list[dict[str, object]] = []
    total = len(results) * len(GENERALISATION_STD_LEVELS)
    count = 0

    print("\n=== Noise Generalisation Sweep ===")
    for row in results:
        for std in GENERALISATION_STD_LEVELS:
            count += 1
            label = f"{row['run_name']} @ σ={std:.2f}"
            print(f"[{count}/{total}] Evaluating {label}")
            noise_config = NoiseConfig(
                noise_mode="gaussian",
                gaussian_std=std,
            )
            metrics = evaluate_checkpoint(
                checkpoint_path=str(row["checkpoint_path"]),
                sequence_length=args.sequence_length,
                num_samples=args.test_samples,
                batch_size=args.batch_size,
                noise_config=noise_config,
                seed=args.seed + 20_000 + count,
                device=args.device,
            )
            gen_rows.append(
                {
                    "run_name": row["run_name"],
                    "model": row["model"],
                    "bottleneck_dim": row["bottleneck_dim"],
                    "trained_noise_mode": row["noise_mode"],
                    "eval_gaussian_std": std,
                    "test_mse": metrics["mse"],
                    "test_noisy_snr_db": metrics["noisy_snr_db"],
                    "test_recon_snr_db": metrics["recon_snr_db"],
                    "test_snr_improvement_db": metrics["snr_improvement_db"],
                }
            )
            print(
                f"  mse={metrics['mse']:.4f} | "
                f"snr_improvement={metrics['snr_improvement_db']:.4f} dB"
            )

    return gen_rows


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ---- Main sweep ----
    results = run_main_sweep(args, checkpoint_dir)
    csv_path = output_dir / "results.csv"
    json_path = output_dir / "results.json"
    write_csv(results, csv_path)
    json_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved main results → {csv_path}")
    print(f"Saved main results → {json_path}")

    # ---- Noise generalisation sweep ----
    if args.noise_generalization:
        gen_rows = run_noise_generalization_sweep(results, args)
        gen_csv = output_dir / "generalization.csv"
        gen_json = output_dir / "generalization.json"
        write_csv(gen_rows, gen_csv)
        gen_json.write_text(json.dumps(gen_rows, indent=2))
        print(f"\nSaved generalisation results → {gen_csv}")
        print(f"Saved generalisation results → {gen_json}")


if __name__ == "__main__":
    main()
