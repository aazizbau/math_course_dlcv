"""Day 32: PCA from scratch (NumPy demo)."""
from __future__ import annotations

import numpy as np


def pca_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mean, eigenvalues, eigenvectors (sorted by variance)."""

    mean = X.mean(axis=0)
    Xc = X - mean
    cov = np.cov(Xc, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    return mean, eigvals[idx], eigvecs[:, idx]


def project(X: np.ndarray, eigvecs: np.ndarray, k: int) -> np.ndarray:
    return X @ eigvecs[:, :k]


def reconstruct(Z: np.ndarray, eigvecs: np.ndarray, k: int) -> np.ndarray:
    return Z @ eigvecs[:, :k].T


def explained_variance_ratio(eigvals: np.ndarray) -> np.ndarray:
    return eigvals / eigvals.sum()


def main() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(300, 2))
    X[:, 1] = 0.7 * X[:, 0] + 0.3 * rng.normal(0, 1, size=300)

    mean, eigvals, eigvecs = pca_fit(X)
    evr = explained_variance_ratio(eigvals)
    print("Mean:", mean)
    print("Eigenvalues:", eigvals)
    print("Explained variance ratio:", evr)
    Z = project(X - mean, eigvecs, k=1)
    X_recon = reconstruct(Z, eigvecs, k=1) + mean
    print("Reconstruction mean error:", np.mean(np.linalg.norm(X - X_recon, axis=1)))


if __name__ == "__main__":
    main()
