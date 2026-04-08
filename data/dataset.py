from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

NoiseMode = Literal["gaussian", "masking", "both"]


@dataclass
class NoiseConfig:
    noise_mode: NoiseMode = "both"
    gaussian_std: float = 0.15
    mask_ratio: float = 0.15
    mask_min_length: int = 4
    mask_max_length: int = 16
    mask_value: float = 0.0


def generate_sine_signal(
    sequence_length: int,
    rng: np.random.Generator,
    min_components: int = 2,
    max_components: int = 5,
    include_chirp_probability: float = 0.35,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, sequence_length, dtype=np.float32)
    signal = np.zeros(sequence_length, dtype=np.float32)

    num_components = int(rng.integers(min_components, max_components + 1))
    for _ in range(num_components):
        amplitude = float(rng.uniform(0.3, 1.0))
        frequency = float(rng.uniform(1.0, 12.0))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        signal += amplitude * np.sin(2.0 * np.pi * frequency * t + phase)

    if rng.random() < include_chirp_probability:
        amplitude = float(rng.uniform(0.2, 0.8))
        start_frequency = float(rng.uniform(1.0, 4.0))
        sweep_rate = float(rng.uniform(6.0, 18.0))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        chirp_phase = 2.0 * np.pi * (start_frequency * t + 0.5 * sweep_rate * t**2)
        signal += amplitude * np.sin(chirp_phase + phase)

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak
    return signal.astype(np.float32)


def add_gaussian_noise(
    signal: np.ndarray,
    std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    noise = rng.normal(0.0, std, size=signal.shape).astype(np.float32)
    return signal + noise


def add_random_masking(
    signal: np.ndarray,
    mask_ratio: float,
    mask_min_length: int,
    mask_max_length: int,
    mask_value: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if mask_ratio <= 0:
        return signal.copy()

    masked_signal = signal.copy()
    target_masked_points = max(1, int(mask_ratio * signal.shape[0]))
    masked_points = 0

    while masked_points < target_masked_points:
        span_length = int(rng.integers(mask_min_length, mask_max_length + 1))
        span_length = min(span_length, signal.shape[0])
        start = int(rng.integers(0, signal.shape[0] - span_length + 1))
        end = start + span_length
        masked_signal[start:end] = mask_value
        masked_points += span_length

    return masked_signal


def corrupt_signal(
    clean_signal: np.ndarray,
    noise_config: NoiseConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    noisy_signal = clean_signal.copy()

    if noise_config.noise_mode in {"gaussian", "both"}:
        noisy_signal = add_gaussian_noise(
            noisy_signal,
            std=noise_config.gaussian_std,
            rng=rng,
        )

    if noise_config.noise_mode in {"masking", "both"}:
        noisy_signal = add_random_masking(
            noisy_signal,
            mask_ratio=noise_config.mask_ratio,
            mask_min_length=noise_config.mask_min_length,
            mask_max_length=noise_config.mask_max_length,
            mask_value=noise_config.mask_value,
            rng=rng,
        )

    return noisy_signal.astype(np.float32)


class SyntheticDenoisingDataset(Dataset):
    def __init__(
        self,
        num_samples: int,
        sequence_length: int,
        noise_config: NoiseConfig | None = None,
        seed: int = 42,
    ) -> None:
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.noise_config = noise_config or NoiseConfig()
        rng = np.random.default_rng(seed)

        clean_signals = []
        noisy_signals = []
        for _ in range(num_samples):
            clean = generate_sine_signal(sequence_length=sequence_length, rng=rng)
            noisy = corrupt_signal(clean, noise_config=self.noise_config, rng=rng)
            clean_signals.append(clean)
            noisy_signals.append(noisy)

        self.clean_signals = torch.from_numpy(np.stack(clean_signals)).unsqueeze(1)
        self.noisy_signals = torch.from_numpy(np.stack(noisy_signals)).unsqueeze(1)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.noisy_signals[index], self.clean_signals[index]
