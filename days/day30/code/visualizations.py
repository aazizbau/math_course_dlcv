"""Day 30 visualizations: reliability diagram and uncertainty map."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .uncertainty_calibration import ece, mc_dropout_predict

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _reliability_data(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    bins = np.linspace(0, 1, n_bins + 1)
    accs = []
    confs = []
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            accs.append(np.nan)
            confs.append((bins[i] + bins[i + 1]) / 2)
        else:
            accs.append(labels[mask].mean())
            confs.append(probs[mask].mean())
    return np.array(confs), np.array(accs)


def plot_reliability_diagram() -> Path:
    rng = np.random.default_rng(2)
    logits = rng.normal(0, 1.0, size=(2000,))
    probs = 1 / (1 + np.exp(-logits))
    labels = (rng.random(2000) < probs).astype(np.float32)

    confs, accs = _reliability_data(probs, labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.plot(confs, accs, marker="o", label="Model")
    ax.set_title(f"Reliability Diagram (ECE={ece(probs, labels):.3f})")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    path = OUT_DIR / "00_reliability_diagram.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_uncertainty_map() -> Path:
    rng = np.random.default_rng(3)
    logits = rng.normal(0, 1.0, size=(96, 96))
    mean_pred, var_pred = mc_dropout_predict(logits, dropout_p=0.25, samples=25)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    axes[0].imshow(mean_pred, cmap="viridis")
    axes[0].set_title("Prediction")
    axes[0].axis("off")

    axes[1].imshow(var_pred, cmap="magma")
    axes[1].set_title("Uncertainty (var)")
    axes[1].axis("off")

    path = OUT_DIR / "01_uncertainty_map.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    print("Saved reliability diagram →", plot_reliability_diagram())
    print("Saved uncertainty map →", plot_uncertainty_map())


if __name__ == "__main__":
    main()
