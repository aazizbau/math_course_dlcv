"""Day 3 computations: vanilla gradient descent on a quadratic bowl."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np


@dataclass(frozen=True)
class QuadraticBowl:
    """Represents L(x, y) = 0.5 * (3x^2 + 0.8xy + y^2)."""

    def loss(self, w: np.ndarray) -> float:
        x, y = w
        return 0.5 * (3 * x**2 + 0.8 * x * y + y**2)

    def grad(self, w: np.ndarray) -> np.ndarray:
        x, y = w
        return np.array([3 * x + 0.4 * y, 0.4 * x + y])


@dataclass
class GradientDescentRunner:
    bowl: QuadraticBowl
    lr: float = 0.15
    steps: int = 30

    def run(self, init: Iterable[float]) -> np.ndarray:
        w = np.array(init, dtype=float)
        path = [w.copy()]
        for _ in range(self.steps):
            w -= self.lr * self.bowl.grad(w)
            path.append(w.copy())
        return np.stack(path)


def demo() -> None:
    runner = GradientDescentRunner(bowl=QuadraticBowl(), lr=0.15, steps=12)
    path = runner.run([2.5, -2.0])
    print("Optimization path:\n", path)


if __name__ == "__main__":
    demo()
