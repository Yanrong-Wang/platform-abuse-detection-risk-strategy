"""YelpChi ingestion and transductive graph feature construction."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import loadmat

RELATIONS = {"rur": "net_rur", "rsr": "net_rsr", "rtr": "net_rtr"}


@dataclass
class YelpData:
    x: np.ndarray
    y: np.ndarray
    graphs: dict[str, sparse.csr_matrix]


def load_yelp(path: str | Path) -> YelpData:
    """Read the CARE-GNN/DGL YelpChi Matlab file without requiring DGL."""
    mat = loadmat(path)
    x = np.asarray(mat["features"].toarray(), dtype=np.float32)
    y = np.asarray(mat["label"], dtype=np.int8).ravel()
    if x.ndim != 2 or x.shape[0] != len(y) or x.shape[1] != 32:
        raise ValueError("Unexpected YelpChi feature/label dimensions")
    if not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Nonfinite features or nonbinary labels")
    graphs = {}
    for name, key in RELATIONS.items():
        graph = mat[key].tocsr().astype(np.float32)
        if graph.shape != (len(y), len(y)):
            raise ValueError(f"Invalid {name} adjacency dimensions")
        graph.setdiag(0)
        graph.eliminate_zeros()
        graph.data[:] = 1
        graphs[name] = graph
    return YelpData(x, y, graphs)


def make_features(data: YelpData) -> pd.DataFrame:
    """Compute label-free, relation-specific one-hop features for every node.

    This is transductive: the full unlabeled graph and node feature matrix are
    available at score time. No neighbor labels are used.
    """
    cols: dict[str, np.ndarray] = {f"f_{i+1:02d}": data.x[:, i] for i in range(data.x.shape[1])}
    for name, graph in data.graphs.items():
        degree = np.asarray(graph.sum(axis=1)).ravel()
        denom = np.maximum(degree, 1)[:, None]
        mean = np.asarray(graph @ data.x) / denom
        second = np.asarray(graph @ np.square(data.x)) / denom
        std = np.sqrt(np.maximum(second - np.square(mean), 0))
        cols[f"{name}_log_degree"] = np.log1p(degree)
        for i in range(data.x.shape[1]):
            cols[f"{name}_mean_{i+1:02d}"] = mean[:, i]
            cols[f"{name}_std_{i+1:02d}"] = std[:, i]
    frame = pd.DataFrame(cols)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Feature engineering produced nonfinite values")
    return frame


def relation_summary(data: YelpData) -> pd.DataFrame:
    rows = []
    for name, graph in data.graphs.items():
        upper = sparse.triu(graph, k=1).tocoo()
        if len(upper.row):
            same = np.mean(data.y[upper.row] == data.y[upper.col])
            positive_edge = np.mean((data.y[upper.row] == 1) & (data.y[upper.col] == 1))
            positive_endpoints = float(data.y[upper.row].sum() + data.y[upper.col].sum())
            positive_neighbor_rate = float(2 * ((data.y[upper.row] == 1) & (data.y[upper.col] == 1)).sum() / positive_endpoints) if positive_endpoints else float("nan")
        else:
            same = positive_edge = positive_neighbor_rate = float("nan")
        degree = np.asarray(graph.sum(axis=1)).ravel()
        rows.append({"relation": name.upper(), "undirected_edges": len(upper.row),
                     "mean_degree": float(degree.mean()), "median_degree": float(np.median(degree)),
                     "isolated_nodes": int((degree == 0).sum()),
                     "edge_label_agreement": float(same), "both_spam_edge_fraction": float(positive_edge),
                     "positive_neighbor_spam_rate": positive_neighbor_rate})
    return pd.DataFrame(rows)
