"""Day 6 surfaces and helper gradients for convex/non-convex landscapes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def convex_bowl(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 0.5 * (x**2 + y**2)


def banana(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x**2 + 5 * (y - x**2) ** 2


def waves(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sin(x) * np.cos(y) + 0.1 * (x**2 + y**2)


def banana_grad(w: np.ndarray) -> np.ndarray:
    x, y = w
    dx = 2 * x - 20 * x * (y - x**2)
    dy = 10 * (y - x**2)
    return np.array([dx, dy])


@dataclass
class GDConfig:
    lr: float = 1e-3
    steps: int = 2000


def gd_path(init: Iterable[float], grad_fn, config: GDConfig) -> np.ndarray:
    w = np.array(init, dtype=float)
    path = [w.copy()]
    for _ in range(config.steps):
        w -= config.lr * grad_fn(w)
        path.append(w.copy())
    return np.stack(path)


def demo() -> None:
    cfg = GDConfig(lr=1e-3, steps=100)
    path = gd_path([-1.5, 1.5], banana_grad, cfg)
    print("Tail of GD path:", path[-3:])


if __name__ == "__main__":
    demo()
