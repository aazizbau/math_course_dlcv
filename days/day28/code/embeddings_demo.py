"""Day 28: foundation model embeddings demo (NumPy-based)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FoundationModel:
    name: str
    focus: str
    best_for: str


MODELS = (
    FoundationModel("Prithvi", "Spatio-temporal masked modeling", "Time-series tasks"),
    FoundationModel("AlphaEarth", "Global EO embeddings", "Fast transfer & search"),
    FoundationModel("SatMAE", "Masked autoencoding", "Spatial representation learning"),
)


def embed_patch(patch: np.ndarray, dim: int = 64, seed: int = 0) -> np.ndarray:
    """Create a deterministic embedding from a patch (toy example)."""

    rng = np.random.default_rng(seed)
    proj = rng.normal(0, 1, size=(patch.size, dim))
    flat = patch.reshape(-1)
    emb = flat @ proj
    emb = emb / (np.linalg.norm(emb) + 1e-6)
    return emb.astype(np.float32)


def embedding_distance(e1: np.ndarray, e2: np.ndarray) -> float:
    return float(np.linalg.norm(e1 - e2))


def main() -> None:
    for model in MODELS:
        print(f"{model.name}: {model.focus} | best for {model.best_for}")

    rng = np.random.default_rng(1)
    patch_t1 = rng.random((16, 16, 4)).astype(np.float32)
    patch_t2 = patch_t1.copy()
    patch_t2[4:8, 6:10, :] += 0.4
    patch_t2 = np.clip(patch_t2, 0, 1)

    emb1 = embed_patch(patch_t1)
    emb2 = embed_patch(patch_t2)
    print("Embedding distance:", embedding_distance(emb1, emb2))


if __name__ == "__main__":
    main()
