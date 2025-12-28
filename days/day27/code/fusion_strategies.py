"""Day 27: multi-modal fusion strategies (NumPy demo)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FusionSummary:
    name: str
    description: str
    pros: str
    cons: str


FUSIONS = (
    FusionSummary(
        "Early",
        "Concatenate modalities at input and feed one encoder",
        "Simple and fast",
        "Noise statistics collide; clouds can dominate gradients",
    ),
    FusionSummary(
        "Mid-level",
        "Separate encoders, then fuse feature maps",
        "Modality-specific learning; robust default",
        "More parameters and memory",
    ),
    FusionSummary(
        "Late",
        "Fuse predictions from separate models",
        "Robust to missing sensors",
        "Weak cross-modal interaction",
    ),
    FusionSummary(
        "Attention",
        "Learn weights per modality and spatial location",
        "Dynamic trust across sensors",
        "Heavier compute and tuning",
    ),
)


def attention_fusion(f_opt: np.ndarray, f_sar: np.ndarray, f_dem: np.ndarray) -> np.ndarray:
    """Simple attention fusion using softmax weights from global means."""

    scores = np.array([f_opt.mean(), f_sar.mean(), f_dem.mean()])
    weights = np.exp(scores - scores.max())
    weights = weights / weights.sum()
    fused = weights[0] * f_opt + weights[1] * f_sar + weights[2] * f_dem
    return fused


def main() -> None:
    for fusion in FUSIONS:
        print(f"{fusion.name}: {fusion.description} | pros={fusion.pros} | cons={fusion.cons}")

    rng = np.random.default_rng(0)
    f_opt = rng.normal(0.2, 0.1, size=(8, 8))
    f_sar = rng.normal(0.4, 0.2, size=(8, 8))
    f_dem = rng.normal(0.1, 0.05, size=(8, 8))
    fused = attention_fusion(f_opt, f_sar, f_dem)
    print("Fused feature mean:", fused.mean())


if __name__ == "__main__":
    main()
