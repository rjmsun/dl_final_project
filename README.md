# Project 2: Denoising Autoencoders for 1D Signals

Starter PyTorch code for the ECE 685D Project 2 brief:
- synthetic 1D signal generation
- **four noise models**: Gaussian, random masking, impulse (salt-and-pepper), sinusoidal interference
- **four architectures**: MLP, 1D CNN, LSTM, and **Transformer** denoising autoencoders
- training with MSE loss
- evaluation with MSE and SNR improvement
- **training-curve visualisation** (`--plot-history`)
- **publication-quality result plots** (`plot_results.py`)
- **noise-generalisation sweep** (`--noise-generalization`)

## Structure

```text
.
├── config.py
├── data/
│   └── dataset.py          # signal generation + all noise models
├── models/
│   ├── mlp_ae.py
│   ├── cnn_ae.py
│   ├── lstm_ae.py
│   └── transformer_ae.py   # NEW: Transformer autoencoder
├── train.py
├── experiments.py           # architecture × bottleneck × noise sweep + generalisation
├── evaluate.py
├── visualize.py             # reconstruction plots + training-curve plots
├── plot_results.py          # NEW: generates figures from results.csv
├── requirements.txt
└── README.md
```

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
# Original noise modes
python3 train.py --model cnn         --epochs 20 --noise-mode both
python3 train.py --model mlp         --epochs 20 --noise-mode gaussian
python3 train.py --model lstm        --epochs 20 --noise-mode masking

# New noise modes
python3 train.py --model transformer --epochs 20 --noise-mode impulse
python3 train.py --model cnn         --epochs 20 --noise-mode sinusoidal
python3 train.py --model lstm        --epochs 20 --noise-mode all
```

### All `--noise-mode` options

| Mode | Description |
|------|-------------|
| `gaussian` | Additive Gaussian noise (σ controlled by `--gaussian-std`) |
| `masking` | Random contiguous spans zeroed out |
| `both` | Gaussian + masking combined |
| `impulse` | Random large-amplitude spikes (salt-and-pepper) |
| `sinusoidal` | Single-frequency sinusoidal interference tone |
| `all` | All four noise types combined |

### All `--model` options

| Model | Description |
|-------|-------------|
| `mlp` | Fully-connected autoencoder (treats full sequence as flat vector) |
| `cnn` | 1D convolutional autoencoder (extracts local patterns) |
| `lstm` | Bidirectional LSTM encoder with a global bottleneck and MLP decoder |
| `transformer` | Multi-head self-attention encoder + decoder with positional encoding |

## Evaluate

```bash
python3 evaluate.py --checkpoint checkpoints/cnn_best.pt --noise-mode both
```

## Run Comparison Experiments

```bash
# Basic sweep: architecture × bottleneck × noise mode
python3 experiments.py \
  --models mlp cnn lstm transformer \
  --bottleneck-dims 16 32 64 \
  --noise-modes gaussian masking both impulse sinusoidal \
  --epochs 20

# Add noise-generalisation sweep (evaluates every checkpoint across σ levels)
python3 experiments.py \
  --models mlp cnn lstm transformer \
  --bottleneck-dims 16 32 64 \
  --noise-modes gaussian masking both \
  --epochs 20 \
  --noise-generalization
```

Outputs written to `experiments/`:
- `results.csv` / `results.json` — main sweep
- `generalization.csv` / `generalization.json` — generalisation sweep (if enabled)
- per-run checkpoints in `experiments/checkpoints/`

## Plot Results

```bash
# After running experiments.py, generate all comparison figures
python3 plot_results.py

# Specify custom paths
python3 plot_results.py \
  --results experiments/results.csv \
  --generalization experiments/generalization.csv \
  --output-dir artifacts
```

Figures saved to `artifacts/`:
- `fig_arch_comparison.png` — test MSE by architecture, grouped by noise mode
- `fig_bottleneck.png` — MSE vs bottleneck dimension per architecture
- `fig_snr_comparison.png` — SNR improvement by architecture × noise mode
- `fig_generalization.png` — SNR improvement vs Gaussian σ (generalisation sweep)

## Visualize Reconstructions + Training Curves

```bash
# Reconstruction plots only
python3 visualize.py --checkpoint checkpoints/cnn_best.pt --num-examples 3

# Reconstruction + training curve
python3 visualize.py --checkpoint checkpoints/cnn_best.pt --num-examples 3 --plot-history
```

`--plot-history` adds a `_training_curve.png` next to the reconstruction output showing
train vs. val MSE over epochs, with the best epoch marked.

## Notes

- The CNN model expects `sequence_length` to be divisible by 4.
- The Transformer model requires `hidden_dim` to be divisible by `--num-heads` (default 4);
  the model auto-rounds up if needed.
- A good first experiment: compare `mlp`, `cnn`, `lstm`, `transformer` with the same
  bottleneck dimension and noise mode, then report MSE + SNR improvement.
- The recommended baseline table: architecture × bottleneck size × noise type →
  test MSE + test SNR improvement. `plot_results.py` generates this automatically.
- The noise-generalisation sweep answers: does a model trained at σ=0.15 still work
  at σ=0.05 or σ=0.40? Run with `--noise-generalization` and inspect
  `fig_generalization.png`.
