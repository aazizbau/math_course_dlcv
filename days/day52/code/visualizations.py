"""Day 52 visualizations: noise vs sharp/flat minima, dropout masks."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_noise_flattening() -> Path:
    x = np.linspace(-2, 2, 400)
    sharp = x**4
    flat = 0.5 * x**2

    rng = np.random.default_rng(0)
    noise = 0.1 * rng.standard_normal(len(x))

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(x, sharp, label="Sharp minimum", color="tab:red")
    ax.plot(x, sharp + noise, color="tab:red", alpha=0.4)
    ax.plot(x, flat, label="Flat minimum", color="tab:blue")
    ax.plot(x, flat + noise, color="tab:blue", alpha=0.4)
    ax.set_title("Noise Disrupts Sharp Minima")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = OUT_DIR / "00_noise_flattening.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_dropout_mask() -> Path:
    rng = np.random.default_rng(1)
    mask = (rng.random((10, 10)) < 0.6).astype(int)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(mask, cmap="gray", interpolation="nearest")
    ax.set_title("Dropout Mask (1=keep)")
    ax.axis("off")

    path = OUT_DIR / "01_dropout_mask.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    path1 = plot_noise_flattening()
    path2 = plot_dropout_mask()
    print("Saved plots →", path1, path2)


if __name__ == "__main__":
    main()
