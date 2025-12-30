"""Day 30: uncertainty and calibration utilities (NumPy demo)."""
from __future__ import annotations

import numpy as np


def mc_dropout_predict(logits: np.ndarray, dropout_p: float = 0.2, samples: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Simulate MC dropout by masking logits with Bernoulli noise."""

    rng = np.random.default_rng(0)
    preds = []
    for _ in range(samples):
        mask = rng.random(logits.shape) > dropout_p
        noisy_logits = logits * mask
        probs = 1 / (1 + np.exp(-noisy_logits))
        preds.append(probs)
    preds = np.stack(preds, axis=0)
    return preds.mean(axis=0), preds.var(axis=0)


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error for binary probabilities."""

    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        ece_val += (mask.sum() / len(probs)) * abs(acc - conf)
    return float(ece_val)


def temperature_scale(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to logits."""

    return logits / temperature


def main() -> None:
    rng = np.random.default_rng(1)
    logits = rng.normal(0, 1.2, size=(256, 256))
    labels = (rng.random((256, 256)) > 0.8).astype(np.float32)

    mean_pred, var_pred = mc_dropout_predict(logits)
    print("Mean prediction:", mean_pred.mean())
    print("Uncertainty (var) mean:", var_pred.mean())

    probs = 1 / (1 + np.exp(-logits))
    print("ECE (raw):", ece(probs.flatten(), labels.flatten()))
    scaled = 1 / (1 + np.exp(-temperature_scale(logits, 1.5)))
    print("ECE (temp-scaled):", ece(scaled.flatten(), labels.flatten()))


if __name__ == "__main__":
    main()
