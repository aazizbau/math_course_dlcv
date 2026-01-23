"""Day 45 visualizations: variance propagation vs init std."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .initialization import propagate

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_variance_propagation() -> Path:
    stds = np.linspace(0.001, 0.05, 20)
    vars_ = [propagate(3, std) for std in stds]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(stds, vars_, marker="o")
    ax.set_title("Variance After 3 Layers")
    ax.set_xlabel("Init std")
    ax.set_ylabel("Output variance")
    ax.grid(True, alpha=0.3)

    path = OUT_DIR / "00_variance_propagation.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    path = plot_variance_propagation()
    print("Saved variance propagation plot →", path)


if __name__ == "__main__":
    main()
