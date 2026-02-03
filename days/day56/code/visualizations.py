"""Day 56 visualizations: aleatoric vs epistemic intuition plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_uncertainty_bars() -> Path:
    labels = ["Aleatoric", "Epistemic"]
    values = [0.35, 0.15]

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(labels, values, color=["tab:blue", "tab:orange"])
    ax.set_title("Uncertainty Sources")
    ax.set_ylabel("Magnitude")
    ax.grid(True, axis='y', alpha=0.3)

    path = OUT_DIR / "00_uncertainty_bars.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_ensemble_spread() -> Path:
    rng = np.random.default_rng(0)
    preds = rng.normal(0.6, 0.08, size=(30,))

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(preds, bins=12, alpha=0.7)
    ax.axvline(preds.mean(), color='black', linestyle='--', label='Mean')
    ax.set_title("Ensemble Prediction Spread")
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Count")
    ax.legend()

    path = OUT_DIR / "01_ensemble_spread.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    path1 = plot_uncertainty_bars()
    path2 = plot_ensemble_spread()
    print("Saved plots →", path1, path2)


if __name__ == "__main__":
    main()
