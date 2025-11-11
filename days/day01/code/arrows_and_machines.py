"""Day 1 core linear-algebra computations for the 45-day DL/CV math course."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ArrowMachineExample:
    """Encapsulates two feature arrows and a transformation matrix."""

    a: np.ndarray
    b: np.ndarray
    A: np.ndarray

    def dot(self) -> float:
        return float(np.dot(self.a, self.b))

    def cosine_similarity(self) -> float:
        return self.dot() / (np.linalg.norm(self.a) * np.linalg.norm(self.b))

    def projection_of_b_on_a(self) -> np.ndarray:
        scale = (self.b @ self.a) / (np.linalg.norm(self.a) ** 2)
        return scale * self.a

    def transform(self) -> np.ndarray:
        return self.A @ self.a


def main() -> None:
    example = ArrowMachineExample(
        a=np.array([3.0, 1.2]),
        b=np.array([1.4, 3.3]),
        A=np.array([[1.0, 0.5], [0.0, 1.2]]),
    )

    dot_ab = example.dot()
    cos_sim = example.cosine_similarity()
    proj_b_on_a = example.projection_of_b_on_a()
    transformed = example.transform()

    print("Dot(a, b) =", dot_ab)
    print("Cosine similarity =", cos_sim)
    print("Projection of b on a =", proj_b_on_a)
    print("Matrix transform A·a =", transformed)


if __name__ == "__main__":
    main()
