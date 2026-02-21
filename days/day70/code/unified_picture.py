"""Day 70: unified synthesis of optimization, geometry, information, and statistics."""
from __future__ import annotations

import numpy as np


def toy_training_objective(theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    preds = x @ theta
    return float(np.mean((preds - y) ** 2))


def flatness_proxy(theta: np.ndarray, eps: float = 1e-2) -> float:
    # Simple proxy: norm-scaled perturbation sensitivity
    return float(np.linalg.norm(theta) * eps)


def main() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(256, 4))
    true_theta = np.array([0.8, -0.3, 0.4, 0.1])
    y = x @ true_theta + 0.05 * rng.normal(size=256)

    theta = np.array([0.7, -0.2, 0.35, 0.15])
    print('Objective:', toy_training_objective(theta, x, y))
    print('Flatness proxy:', flatness_proxy(theta))


if __name__ == '__main__':
    main()
