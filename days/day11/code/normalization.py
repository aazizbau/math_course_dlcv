"""Day 11 normalization utilities (BatchNorm, LayerNorm, drift simulation)."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def batchnorm_forward(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """BatchNorm along batch dimension (expects x shape (batch, features))."""
    mu = x.mean(axis=0, keepdims=True)
    var = x.var(axis=0, keepdims=True)
    xhat = (x - mu) / np.sqrt(var + eps)
    gamma = np.ones((1, x.shape[1]))
    beta = np.zeros((1, x.shape[1]))
    return gamma * xhat + beta


def layernorm_forward(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """LayerNorm along feature dimension (expects x shape (batch, features))."""
    mu = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)
    xhat = (x - mu) / np.sqrt(var + eps)
    gamma = np.ones((1, x.shape[1]))
    beta = np.zeros((1, x.shape[1]))
    return gamma * xhat + beta


@dataclass
class DriftSimulation:
    """Simulates drifting distributions that BatchNorm recenters."""

    samples: int = 1000
    scale_per_frame: float = 0.05
    shift_per_frame: float = 0.1

    def step(self, frame: int) -> tuple[np.ndarray, np.ndarray]:
        base = np.random.randn(self.samples)
        scale = 1 + frame * self.scale_per_frame
        shift = frame * self.shift_per_frame
        noisy = base * scale + shift
        bn = (noisy - noisy.mean()) / noisy.std()
        return noisy, bn


def main() -> None:
    x = np.random.randn(4, 5)
    print("BatchNorm result:\n", batchnorm_forward(x))
    print("LayerNorm result:\n", layernorm_forward(x))


if __name__ == "__main__":
    main()
