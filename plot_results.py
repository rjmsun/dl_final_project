"""plot_results.py — generate publication-quality comparison figures from results.csv.

Usage
-----
    python3 plot_results.py                             # uses experiments/results.csv
    python3 plot_results.py --results experiments/results.csv --output-dir artifacts

Output (saved to --output-dir, default: artifacts/)
------
    fig_arch_comparison.png   — test MSE by architecture, grouped by noise mode
    fig_bottleneck.png        — test MSE vs bottleneck dimension, one line per architecture
    fig_snr_comparison.png    — SNR improvement by architecture × noise mode
    fig_generalization.png    — (optional) SNR improvement vs Gaussian σ per model
                                (only produced if experiments/generalization.csv exists)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Aesthetic settings ────────────────────────────────────────────────────────
PALETTE = {
    "mlp":         "#4C72B0",
    "cnn":         "#DD8452",
    "lstm":        "#55A868",
    "transformer": "#C44E52",
}
NOISE_HATCHES = {
    "gaussian":   "",
    "masking":    "//",
    "both":       "xx",
    "impulse":    "..",
    "sinusoidal": "--",
    "all":        "oo",
}
DPI = 200
FIG_W = 10


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#F7F7F7",
            "axes.edgecolor": "#CCCCCC",
            "axes.grid": True,
            "grid.color": "white",
            "grid.linewidth": 1.4,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


# ── Individual plot functions ─────────────────────────────────────────────────

def plot_arch_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart: mean test MSE per architecture, grouped by noise mode."""
    noise_modes = df["noise_mode"].unique().tolist()
    models = df["model"].unique().tolist()
    n_modes = len(noise_modes)
    n_models = len(models)
    x = np.arange(n_modes)
    width = 0.75 / n_models

    fig, ax = plt.subplots(figsize=(FIG_W, 5))
    for i, model in enumerate(models):
        means = [
            df[(df["model"] == model) & (df["noise_mode"] == nm)]["test_mse"].mean()
            for nm in noise_modes
        ]
        bars = ax.bar(
            x + (i - n_models / 2 + 0.5) * width,
            means,
            width=width * 0.92,
            label=model.upper(),
            color=PALETTE.get(model, "#888888"),
            edgecolor="white",
            linewidth=0.6,
        )
        for bar, val in zip(bars, means):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.0005,
                    f"{val:.4f}",
                    ha="center", va="bottom", fontsize=7, rotation=45,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([nm.capitalize() for nm in noise_modes])
    ax.set_xlabel("Noise Mode")
    ax.set_ylabel("Test MSE (lower is better)")
    ax.set_title("Architecture Comparison: Test MSE by Noise Mode")
    ax.legend(title="Architecture", loc="upper right")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    fig.tight_layout()
    out = output_dir / "fig_arch_comparison.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_bottleneck(df: pd.DataFrame, output_dir: Path) -> None:
    """Line chart: test MSE vs bottleneck dimension, one line per architecture."""
    if df["bottleneck_dim"].nunique() < 2:
        print("  Skipping fig_bottleneck.png — only one bottleneck_dim found (need ≥ 2).")
        return

    models = df["model"].unique().tolist()
    fig, ax = plt.subplots(figsize=(FIG_W, 5))
    for model in models:
        sub = df[df["model"] == model].groupby("bottleneck_dim")["test_mse"].mean().reset_index()
        ax.plot(
            sub["bottleneck_dim"], sub["test_mse"],
            marker="o", linewidth=2, markersize=7,
            label=model.upper(),
            color=PALETTE.get(model, "#888888"),
        )
        for _, row in sub.iterrows():
            ax.annotate(
                f"{row['test_mse']:.4f}",
                (row["bottleneck_dim"], row["test_mse"]),
                textcoords="offset points", xytext=(4, 4), fontsize=7,
            )

    ax.set_xlabel("Bottleneck Dimension")
    ax.set_ylabel("Test MSE (lower is better)")
    ax.set_title("Effect of Bottleneck Dimension on Reconstruction MSE")
    ax.legend(title="Architecture")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    out = output_dir / "fig_bottleneck.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_snr_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart: SNR improvement by architecture × noise mode."""
    noise_modes = df["noise_mode"].unique().tolist()
    models = df["model"].unique().tolist()
    n_modes = len(noise_modes)
    n_models = len(models)
    x = np.arange(n_modes)
    width = 0.75 / n_models

    fig, ax = plt.subplots(figsize=(FIG_W, 5))
    for i, model in enumerate(models):
        means = [
            df[(df["model"] == model) & (df["noise_mode"] == nm)]["test_snr_improvement_db"].mean()
            for nm in noise_modes
        ]
        ax.bar(
            x + (i - n_models / 2 + 0.5) * width,
            means,
            width=width * 0.92,
            label=model.upper(),
            color=PALETTE.get(model, "#888888"),
            hatch=NOISE_HATCHES.get(noise_modes[0] if noise_modes else "", ""),
            edgecolor="white",
            linewidth=0.6,
        )

    ax.axhline(0, color="#333333", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([nm.capitalize() for nm in noise_modes])
    ax.set_xlabel("Noise Mode")
    ax.set_ylabel("SNR Improvement (dB, higher is better)")
    ax.set_title("Architecture Comparison: SNR Improvement by Noise Mode")
    ax.legend(title="Architecture", loc="upper right")
    fig.tight_layout()
    out = output_dir / "fig_snr_comparison.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_generalization(gen_csv: Path, output_dir: Path) -> None:
    """Line chart: SNR improvement vs Gaussian σ for each model (generalisation sweep)."""
    if not gen_csv.exists():
        print(f"  Skipping fig_generalization.png — {gen_csv} not found.")
        return

    df = pd.read_csv(gen_csv)
    models = df["model"].unique().tolist()

    fig, ax = plt.subplots(figsize=(FIG_W, 5))
    for model in models:
        sub = (
            df[df["model"] == model]
            .groupby("eval_gaussian_std")["test_snr_improvement_db"]
            .mean()
            .reset_index()
        )
        ax.plot(
            sub["eval_gaussian_std"], sub["test_snr_improvement_db"],
            marker="o", linewidth=2, markersize=7,
            label=model.upper(),
            color=PALETTE.get(model, "#888888"),
        )

    ax.set_xlabel("Evaluation Gaussian σ (noise intensity)")
    ax.set_ylabel("SNR Improvement (dB)")
    ax.set_title("Noise Generalisation: SNR Improvement vs. Gaussian Noise Intensity")
    ax.legend(title="Architecture")
    fig.tight_layout()
    out = output_dir / "fig_generalization.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  Saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot experiment results from results.csv.")
    parser.add_argument("--results", type=str, default="experiments/results.csv",
                        help="Path to the main results CSV.")
    parser.add_argument("--generalization", type=str, default="experiments/generalization.csv",
                        help="Path to the generalization CSV (optional).")
    parser.add_argument("--output-dir", type=str, default="artifacts",
                        help="Directory to save figures.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results_path = Path(args.results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_path.exists():
        raise FileNotFoundError(
            f"Results file not found: {results_path}\n"
            "Run experiments.py first to generate it."
        )

    _style()
    df = pd.read_csv(results_path)
    print(f"Loaded {len(df)} rows from {results_path}")
    print("Generating figures …")

    plot_arch_comparison(df, output_dir)
    plot_bottleneck(df, output_dir)
    plot_snr_comparison(df, output_dir)
    plot_generalization(Path(args.generalization), output_dir)

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
