"""Day 10 activation functions and derivative utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np


@dataclass(frozen=True)
class Activation:
    name: str
    fn: Callable[[np.ndarray], np.ndarray]
    derivative: Callable[[np.ndarray], np.ndarray]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def dsigmoid(x: np.ndarray) -> np.ndarray:
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def dtanh(x: np.ndarray) -> np.ndarray:
    return 1 - np.tanh(x) ** 2


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def drelu(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


def leaky_relu(x: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    return np.where(x > 0, x, alpha * x)


def dleaky_relu(x: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    grad = np.ones_like(x)
    grad[x <= 0] = alpha
    return grad


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def dgelu(x: np.ndarray) -> np.ndarray:
    eps = 1e-4
    return (gelu(x + eps) - gelu(x - eps)) / (2 * eps)


def build_activations() -> Dict[str, Activation]:
    return {
        "Sigmoid": Activation("Sigmoid", sigmoid, dsigmoid),
        "Tanh": Activation("Tanh", tanh, dtanh),
        "ReLU": Activation("ReLU", relu, drelu),
        "LeakyReLU": Activation("LeakyReLU", leaky_relu, dleaky_relu),
        "GELU": Activation("GELU", gelu, dgelu),
    }


def main() -> None:
    acts = build_activations()
    x = np.linspace(-3, 3, 5)
    for name, activation in acts.items():
        print(name, activation.fn(x))


if __name__ == "__main__":
    main()
