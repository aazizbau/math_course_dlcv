"""Day 12 weight initialization utilities for variance propagation simulations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

InitType = Literal["xavier", "he", "bad"]


def sample_weights(shape: tuple[int, int], init: InitType, rng: np.random.Generator) -> np.ndarray:
    fan_in, fan_out = shape
    if init == "xavier":
        scale = np.sqrt(2.0 / (fan_in + fan_out))
    elif init == "he":
        scale = np.sqrt(2.0 / fan_in)
    else:  # intentionally poor init
        scale = 0.6
    return rng.normal(scale=scale, size=shape)


@dataclass
class VarianceSimulator:
    layers: int = 50
    width: int = 200
    seed: int = 0

    def run(self, init: InitType) -> list[float]:
        rng = np.random.default_rng(self.seed)
        x = rng.normal(size=self.width)
        variances: list[float] = []
        for _ in range(self.layers):
            W = sample_weights((self.width, self.width), init, rng)
            x = W @ x
            if init == "xavier":
                x = np.tanh(x)
            elif init == "he":
                x = np.maximum(0, x)
            variances.append(float(np.var(x)))
        return variances


def main() -> None:
    sim = VarianceSimulator(layers=10, width=128)
    for init in ("xavier", "he", "bad"):
        vars_ = sim.run(init)
        print(init, vars_[-1])


if __name__ == "__main__":
    main()
