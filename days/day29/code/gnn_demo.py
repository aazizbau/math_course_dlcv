"""Day 29: toy message passing for EO graphs (NumPy-based)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Graph:
    node_features: np.ndarray
    edge_index: np.ndarray


def message_passing(graph: Graph, steps: int = 2) -> np.ndarray:
    """Simple message passing: average neighbor features."""

    x = graph.node_features.copy()
    n_nodes = x.shape[0]
    neighbors = [[] for _ in range(n_nodes)]
    for src, dst in graph.edge_index.T:
        neighbors[dst].append(src)

    for _ in range(steps):
        x_new = x.copy()
        for i in range(n_nodes):
            if neighbors[i]:
                neigh_feats = np.stack([x[j] for j in neighbors[i]], axis=0)
                x_new[i] = 0.6 * x[i] + 0.4 * neigh_feats.mean(axis=0)
        x = x_new
    return x


def build_grid_graph(size: int = 6) -> Graph:
    """Create a grid graph with 4-neighborhood connectivity."""

    n = size * size
    node_features = np.random.randn(n, 4).astype(np.float32)
    edges = []

    def idx(r: int, c: int) -> int:
        return r * size + c

    for r in range(size):
        for c in range(size):
            if r + 1 < size:
                edges.append((idx(r, c), idx(r + 1, c)))
                edges.append((idx(r + 1, c), idx(r, c)))
            if c + 1 < size:
                edges.append((idx(r, c), idx(r, c + 1)))
                edges.append((idx(r, c + 1), idx(r, c)))

    edge_index = np.array(edges, dtype=np.int64).T
    return Graph(node_features=node_features, edge_index=edge_index)


def main() -> None:
    graph = build_grid_graph()
    print("Nodes:", graph.node_features.shape[0])
    print("Edges:", graph.edge_index.shape[1])
    out = message_passing(graph, steps=3)
    print("Feature mean before:", graph.node_features.mean())
    print("Feature mean after:", out.mean())


if __name__ == "__main__":
    main()
