"""Existing anchor geometry and sparse exact-distance DBSCAN.

Rows are [seconds, projected x metres, projected y metres, heading radians].
The historical scales (10 seconds, 10000 metres, pi radians) are unchanged.
DBSCAN still counts anchors rather than distinct source flights.
"""
from __future__ import annotations

import numpy as np

def wrap_heading(value: np.ndarray) -> np.ndarray:
    return (value + np.pi) % (2 * np.pi) - np.pi


def anchor_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Paper distance between anchors [absolute seconds, projected metres x/y, heading rad]."""
    delta = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return np.sqrt((delta[..., 0] / 10.0) ** 2 + np.sum((delta[..., 1:3] / 10000.0) ** 2, axis=-1) + (wrap_heading(delta[..., 3]) / np.pi) ** 2)


def cluster_anchors(anchors: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Sparse exact-metric DBSCAN, using chord embedding only to find candidates."""
    from scipy.sparse import coo_matrix
    from scipy.spatial import cKDTree
    from sklearn.cluster import DBSCAN
    if not len(anchors):
        return np.empty(0, dtype=np.int64)
    embedded = np.column_stack((anchors[:, 0] / 10.0, anchors[:, 1:3] / 10000.0, np.cos(anchors[:, 3]) / np.pi, np.sin(anchors[:, 3]) / np.pi))
    pairs = cKDTree(embedded).query_pairs(eps, output_type='ndarray')
    if len(pairs):
        distances = anchor_distance(anchors[pairs[:, 0]], anchors[pairs[:, 1]])
        select = distances <= eps
        pairs, distances = (pairs[select], distances[select])
        distances = np.maximum(distances, np.finfo(np.float64).eps)
        row = np.r_[pairs[:, 0], pairs[:, 1], np.arange(len(anchors))]
        col = np.r_[pairs[:, 1], pairs[:, 0], np.arange(len(anchors))]
        data = np.r_[distances, distances, np.zeros(len(anchors))]
    else:
        row = col = np.arange(len(anchors))
        data = np.zeros(len(anchors))
    graph = coo_matrix((data, (row, col)), shape=(len(anchors), len(anchors))).tocsr()
    try:
        from sklearn.neighbors import sort_graph_by_row_values
        graph = sort_graph_by_row_values(graph, copy=False, warn_when_not_sorted=False)
    except ImportError:
        pass
    return DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit_predict(graph)
