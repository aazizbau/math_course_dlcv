"""Day 4 computations: vanilla GD, momentum, and Nesterov on a tilted bowl."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Bowl:
    """Quadratic bowl L(x, y) = 0.5 * (3x^2 + 0.8xy + y^2)."""

    def grad(self, w: np.ndarray) -> np.ndarray:
        x, y = w
        return np.array([3 * x + 0.4 * y, 0.4 * x + y])


@dataclass
class OptimizerConfig:
    lr: float = 0.15
    beta: float = 0.9
    steps: int = 40


def gradient_descent(init: Iterable[float], bowl: Bowl, lr: float, steps: int) -> np.ndarray:
    w = np.array(init, dtype=float)
    path = [w.copy()]
    for _ in range(steps):
        w -= lr * bowl.grad(w)
        path.append(w.copy())
    return np.stack(path)


def momentum(init: Iterable[float], bowl: Bowl, config: OptimizerConfig) -> np.ndarray:
    w = np.array(init, dtype=float)
    v = np.zeros_like(w)
    path = [w.copy()]
    for _ in range(config.steps):
        v = config.beta * v - config.lr * bowl.grad(w)
        w += v
        path.append(w.copy())
    return np.stack(path)


def nesterov(init: Iterable[float], bowl: Bowl, config: OptimizerConfig) -> np.ndarray:
    w = np.array(init, dtype=float)
    v = np.zeros_like(w)
    path = [w.copy()]
    for _ in range(config.steps):
        lookahead = w + config.beta * v
        grad = bowl.grad(lookahead)
        v = config.beta * v - config.lr * grad
        w += v
        path.append(w.copy())
    return np.stack(path)


def main() -> None:
    bowl = Bowl()
    init = [2.5, -2.0]
    config = OptimizerConfig(lr=0.15, beta=0.9, steps=12)

    gd_path = gradient_descent(init, bowl, lr=config.lr, steps=config.steps)
    mom_path = momentum(init, bowl, config)
    nag_path = nesterov(init, bowl, config)

    print("Gradient Descent tail:", gd_path[-3:])
    print("Momentum tail:", mom_path[-3:])
    print("Nesterov tail:", nag_path[-3:])


if __name__ == "__main__":
    main()
