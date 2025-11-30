"""Day 9 simulations for vanishing/exploding gradients."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np


def simulate_scalar(a: float, steps: int = 30, g0: float = 1.0) -> np.ndarray:
    g = g0
    vals = []
    for _ in range(steps):
        g *= a
        vals.append(g)
    return np.array(vals)


def jacobian_singular_values(layer_size: int = 50, num_layers: int = 20, seed: int | None = None) -> List[float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(num_layers):
        W = rng.normal(size=(layer_size, layer_size)) * np.sqrt(2 / layer_size)
        s = np.linalg.svd(W, compute_uv=False)
        values.append(float(np.max(s)))
    return values


@dataclass
class GradientEvolution:
    factors: Iterable[float]
    steps: int = 30
    g0: float = 1.0

    def simulate(self) -> dict[float, np.ndarray]:
        return {a: simulate_scalar(a, self.steps, self.g0) for a in self.factors}


def main() -> None:
    evolution = GradientEvolution(factors=[0.7, 1.0, 1.3], steps=30)
    paths = evolution.simulate()
    for a, vals in paths.items():
        print(f"a={a}: final gradient = {vals[-1]:.4e}")

    norms = jacobian_singular_values()
    print("Average singular value from He init layers:", np.mean(norms))


if __name__ == "__main__":
    main()
