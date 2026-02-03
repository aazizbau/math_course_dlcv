"""Day 57 visualizations: prior vs posterior intuition."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_prior_posterior() -> Path:
    x = np.linspace(-3, 3, 400)
    prior = np.exp(-0.5 * x**2)
    likelihood = np.exp(-0.5 * (x - 1.2) ** 2 / 0.5**2)
    posterior = prior * likelihood
    posterior = posterior / posterior.max()

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(x, prior, label="Prior")
    ax.plot(x, likelihood, label="Likelihood")
    ax.plot(x, posterior, label="Posterior")
    ax.set_title("Bayesian Updating")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = OUT_DIR / "00_prior_posterior.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_ensemble_spread() -> Path:
    rng = np.random.default_rng(1)
    preds = rng.normal(0.6, 0.1, size=(30,))

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
    path1 = plot_prior_posterior()
    path2 = plot_ensemble_spread()
    print("Saved plots →", path1, path2)


if __name__ == "__main__":
    main()
