"""Day 7: Jacobian computation for a tiny 2-layer network."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class TinyNetwork:
    W1: np.ndarray
    W2: np.ndarray

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = np.tanh(self.W1 @ x)
        y = self.W2 @ h
        return h, y

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        h, _ = self.forward(x)
        diag = np.diag(1 - h**2)
        return self.W2 @ diag @ self.W1


def build_default_network() -> TinyNetwork:
    W1 = np.array([[1.2, -0.8], [0.5, 1.0]])
    W2 = np.array([[1.0, 0.3], [-0.6, 0.8]])
    return TinyNetwork(W1=W1, W2=W2)


def main() -> None:
    net = build_default_network()
    x = np.array([0.4, -0.2])
    h, y = net.forward(x)
    J = net.jacobian(x)
    print("Input x:", x)
    print("Hidden activation h:", h)
    print("Output y:", y)
    print("Jacobian:\n", J)


if __name__ == "__main__":
    main()
