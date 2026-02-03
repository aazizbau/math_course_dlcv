"""Day 56: aleatoric vs epistemic uncertainty demo."""
from __future__ import annotations

import numpy as np


def decompose_uncertainty(preds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean_pred = preds.mean(axis=0)
    epistemic = preds.var(axis=0)
    return mean_pred, epistemic


def main() -> None:
    preds = np.array([
        [0.7, 0.2, 0.1],
        [0.6, 0.25, 0.15],
        [0.8, 0.15, 0.05],
    ])
    mean_pred, epistemic = decompose_uncertainty(preds)
    print("Mean prediction:", mean_pred)
    print("Epistemic uncertainty:", epistemic)


if __name__ == "__main__":
    main()
