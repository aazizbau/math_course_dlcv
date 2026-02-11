"""Day 63 visualizations: sensitivity and flat-vs-sharp minima."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_weight_sensitivity() -> Path:
    x = 1.0
    ws = np.linspace(0.2, 8.0, 300)
    dw = 0.1
    y = np.tanh(ws * x)
    y_shift = np.tanh((ws + dw) * x)
    sens = np.abs(y_shift - y)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(ws, sens)
    ax.set_title("Output Sensitivity vs Weight Magnitude")
    ax.set_xlabel("Weight w")
    ax.set_ylabel("|f(w+dw)-f(w)|")
    ax.grid(True, alpha=0.3)

    path = OUT_DIR / "00_weight_sensitivity.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_sharp_vs_flat() -> Path:
    x = np.linspace(-2.5, 2.5, 500)
    sharp = 0.8 * x**4 + 0.15 * x**2
    flat = 0.08 * x**4 + 0.15 * x**2

    rng = np.random.default_rng(0)
    noise = 0.03 * rng.normal(size=x.shape)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(x, sharp, label='Sharper basin')
    ax.plot(x, sharp + noise, alpha=0.35)
    ax.plot(x, flat, label='Flatter basin')
    ax.plot(x, flat + noise, alpha=0.35)
    ax.set_title('Noise Impact: Sharp vs Flat Minima')
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = OUT_DIR / "01_sharp_vs_flat.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    p1 = plot_weight_sensitivity()
    p2 = plot_sharp_vs_flat()
    print("Saved plots ->", p1, p2)


if __name__ == "__main__":
    main()
