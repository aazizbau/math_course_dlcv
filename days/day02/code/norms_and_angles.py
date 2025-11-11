"""Day 2 computations: vector norms, angles, and cosine similarity."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class NormsAndAngles:
    """Encapsulates two vectors for norm/angle analysis."""

    a: np.ndarray
    b: np.ndarray

    def norm_a(self) -> float:
        return float(np.linalg.norm(self.a))

    def norm_b(self) -> float:
        return float(np.linalg.norm(self.b))

    def dot(self) -> float:
        return float(np.dot(self.a, self.b))

    def cosine_similarity(self) -> float:
        return self.dot() / (self.norm_a() * self.norm_b())

    def unit_a(self) -> np.ndarray:
        return self.a / self.norm_a()

    def unit_b(self) -> np.ndarray:
        return self.b / self.norm_b()

    def summary(self) -> str:
        return (
            f"‖a‖={self.norm_a():.3f}, ‖b‖={self.norm_b():.3f}\n"
            f"Dot(a,b)={self.dot():.3f}, Cosine Similarity={self.cosine_similarity():.3f}\n"
            f"Unit a={self.unit_a()}, Unit b={self.unit_b()}"
        )


def main() -> None:
    example = NormsAndAngles(
        a=np.array([2.0, 3.0]),
        b=np.array([3.0, -1.0]),
    )
    print(example.summary())


if __name__ == "__main__":
    main()
