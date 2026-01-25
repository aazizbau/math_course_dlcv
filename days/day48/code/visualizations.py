"""Day 48 visualizations: negative log-likelihood curves."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_nll_curves() -> Path:
    ps = np.linspace(1e-4, 1 - 1e-4, 400)
    nll_y1 = -np.log(ps)
    nll_y0 = -np.log(1 - ps)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(ps, nll_y1, label="NLL (y=1)")
    ax.plot(ps, nll_y0, label="NLL (y=0)")
    ax.set_xlabel("Predicted probability p")
    ax.set_ylabel("Negative log-likelihood")
    ax.set_title("Bernoulli NLL Curves")
    ax.grid(True, alpha=0.3)
    ax.legend()

    path = OUT_DIR / "00_bernoulli_nll.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_mse_vs_nll() -> Path:
    y = 1.0
    preds = np.linspace(-1, 3, 400)
    mse = (preds - y) ** 2
    nll = 0.5 * (preds - y) ** 2

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(preds, mse, label="MSE")
    ax.plot(preds, nll, label="Gaussian NLL (sigma=1)")
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Loss")
    ax.set_title("MSE vs Gaussian NLL")
    ax.grid(True, alpha=0.3)
    ax.legend()

    path = OUT_DIR / "01_mse_vs_nll.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    path1 = plot_nll_curves()
    path2 = plot_mse_vs_nll()
    print("Saved plots →", path1, path2)


if __name__ == "__main__":
    main()
