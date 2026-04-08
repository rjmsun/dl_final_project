# Project 2: Denoising Autoencoders for 1D Signals

Starter PyTorch code for the ECE 685D Project 2 brief:
- synthetic 1D signal generation
- Gaussian noise and random masking
- MLP, 1D CNN, and LSTM denoising autoencoders
- training with MSE loss
- evaluation with MSE and SNR improvement
- matplotlib reconstruction plots

## Structure

```text
.
├── config.py
├── data/
│   └── dataset.py
├── models/
│   ├── mlp_ae.py
│   ├── cnn_ae.py
│   └── lstm_ae.py
├── train.py
├── experiments.py
├── evaluate.py
├── visualize.py
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
python3 train.py --model cnn --epochs 20 --noise-mode both
python3 train.py --model mlp --epochs 20 --noise-mode gaussian
python3 train.py --model lstm --epochs 20 --noise-mode masking
```

## Evaluate

```bash
python3 evaluate.py --checkpoint checkpoints/cnn_best.pt --noise-mode both
```

## Run Comparison Experiments

```bash
python3 experiments.py \
  --models mlp cnn lstm \
  --bottleneck-dims 16 32 64 \
  --noise-modes gaussian masking both \
  --epochs 20
```

This writes:
- `experiments/results.csv`
- `experiments/results.json`
- per-run checkpoints in `experiments/checkpoints/`

## Visualize

```bash
python3 visualize.py --checkpoint checkpoints/cnn_best.pt --num-examples 3
```

## Notes

- The CNN model expects `sequence_length` to be divisible by 4.
- The dataset currently uses synthetic mixtures of sine waves, with an optional chirp component for extra variation.
- A good first experiment is to compare `mlp`, `cnn`, and `lstm` with the same bottleneck dimension and report MSE plus SNR improvement.
- For the report, a clean baseline table is: architecture vs. bottleneck size vs. noise type, using test MSE and test SNR improvement.
