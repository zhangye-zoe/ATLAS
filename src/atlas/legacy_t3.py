"""Legacy-inspired, optional T3 trajectory diagnostics.

The legacy CORAL scripts evaluated multiple *post-hoc* pseudotime candidates,
not only the learned TimeHead. These routines reproduce their basic estimators
without changing ATLAS training or silently replacing reported T3 scores.

The legacy graph_auto/dpt estimators require scanpy; graph_peripheral uses the
current refactored unsupervised graph-time heuristic. Reference labels are never
used to calculate the unsupervised estimators.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import NearestNeighbors

from .temporal import graph_pseudotime, normalize01


def _require_scanpy():
    try:
        import anndata as ad
        import scanpy as sc
    except ImportError as exc:
        raise ImportError(
            "Legacy graph_auto and DPT candidates require scanpy and anndata. "
            "Install scanpy, or run --modes head graph_peripheral."
        ) from exc
    return ad, sc


def _legacy_geodesic_graph(h: np.ndarray, k: int = 30) -> sparse.csr_matrix:
    """Legacy CORAL undirected kNN graph (including duplicate edge sums)."""
    h = np.asarray(h, dtype=np.float32)
    n = h.shape[0]
    if n < 3:
        return sparse.csr_matrix((n, n), dtype=np.float32)
    nn = NearestNeighbors(n_neighbors=min(max(2, k + 1), n), metric="euclidean").fit(h)
    distances, indices = nn.kneighbors(h)
    rows, cols, values = [], [], []
    for i in range(n):
        for j_index in range(1, indices.shape[1]):
            j = int(indices[i, j_index])
            if j == i:
                continue
            value = float(distances[i, j_index])
            rows.extend((i, j))
            cols.extend((j, i))
            values.extend((value, value))
    graph = sparse.csr_matrix(
        (values, (rows, cols)), shape=(n, n), dtype=np.float32
    )
    graph.eliminate_zeros()
    return graph


def _legacy_auto_root(h: np.ndarray) -> Tuple[int, str]:
    """Replicate legacy root selection by minimum diffusion component 1."""
    ad, sc = _require_scanpy()
    x = ad.AnnData(X=np.asarray(h, dtype=np.float32))
    sc.pp.neighbors(x, n_neighbors=min(30, max(3, x.n_obs - 1)), use_rep="X")
    sc.tl.diffmap(x)
    component = np.asarray(x.obsm["X_diffmap"])[:, 0]
    return int(np.nanargmin(component)), "legacy:min_diffmap1"


def _legacy_graph(h: np.ndarray, root_index: int, k: int = 30) -> np.ndarray:
    if len(h) < 3:
        return np.zeros(len(h), dtype=np.float32)
    d = np.asarray(
        shortest_path(_legacy_geodesic_graph(h, k=k), directed=False, indices=root_index),
        dtype=np.float32,
    )
    finite = np.isfinite(d)
    if not finite.any():
        return np.zeros(len(d), dtype=np.float32)
    d[~finite] = float(np.max(d[finite]))
    return normalize01(d)


def get_t3_candidates(
    h: np.ndarray,
    time_head: np.ndarray,
    modes: Sequence[str],
    graph_k: int = 30,
    root_labels: Optional[Sequence[str]] = None,
    cell_labels: Optional[np.ndarray] = None,
) -> Tuple[dict[str, np.ndarray], dict[str, str]]:
    """Generate time candidates. Graph-root labels are diagnostic-only.

    Using labels to choose a graph root is *not* label-free evaluation and
    must never be reported as unsupervised T3.
    """
    h = np.asarray(h, dtype=np.float32)
    head = np.asarray(time_head, dtype=np.float32).reshape(-1)
    if h.ndim != 2 or head.ndim != 1 or h.shape[0] != len(head):
        raise ValueError("h must be N x D; time_head must contain N values")
    if h.shape[0] < 3:
        raise ValueError("T3 candidates require at least 3 cells")
    allowed = {"head", "graph_peripheral", "graph_auto", "dpt", "graph_root"}
    if any(m not in allowed for m in modes):
        raise ValueError(f"Unknown T3 candidate; expected one of {sorted(allowed)}")

    candidates: dict[str, np.ndarray] = {}
    sources: dict[str, str] = {}
    if "head" in modes:
        candidates["head"] = normalize01(head)
        sources["head"] = "trained TimeHead; original refactor evaluation"

    if "graph_peripheral" in modes:
        candidates["graph_peripheral"] = graph_pseudotime(h, k=graph_k)
        sources["graph_peripheral"] = "unsupervised peripheral root; refactored ATLAS teacher heuristic"

    if "dpt" in modes:
        ad, sc = _require_scanpy()
        x = ad.AnnData(X=h.copy())
        sc.pp.neighbors(x, n_neighbors=min(20, max(3, len(h) - 1)), use_rep="X")
        sc.tl.diffmap(x)
        dc1 = np.asarray(x.obsm["X_diffmap"])[:, 0]
        x.uns["iroot"] = int(np.nanargmin(dc1))
        sc.tl.dpt(x)
        candidates["dpt"] = normalize01(np.asarray(x.obs["dpt_pseudotime"]))
        sources["dpt"] = "unsupervised Scanpy DPT; legacy default in some ratio-loop scripts"

    if "graph_auto" in modes:
        root, desc = _legacy_auto_root(h)
        candidates["graph_auto"] = _legacy_graph(h, root, k=graph_k)
        sources["graph_auto"] = f"legacy graph-geodesic auto-root ({desc}; index={root})"

    if "graph_root" in modes:
        if not root_labels or cell_labels is None:
            raise ValueError(
                "graph_root requires --root-labels and reference cell-type labels. "
                "Its result is label-guided and must not be reported as unsupervised."
            )
        labels = np.asarray(cell_labels, dtype=str)
        mask = np.isin(labels, np.asarray(root_labels, dtype=str))
        if not mask.any():
            raise ValueError("None of the --root-labels occurred in the selected cells")
        centroid = h[mask].mean(axis=0)
        root = int(np.argmin(np.sum((h - centroid) ** 2, axis=1)))
        candidates["graph_root"] = _legacy_graph(h, root, k=graph_k)
        sources["graph_root"] = (
            "LABEL-GUIDED graph root: reference cell type used in inference; "
            "diagnostic only, not a label-free T3 benchmark"
        )
    return candidates, sources


def score_ordering_fast(
    time: np.ndarray, stages: np.ndarray, orient: bool = True
) -> dict[str, float]:
    """Exact strict POA and stage-pair POAUC as in current ATLAS metrics.

    Complexity is O(S^2 N log N), avoiding the legacy O(N^2) Python loop.
    Tied predicted times count 0 for POA and 0.5 for AUC.
    """
    x = np.asarray(time, dtype=float).reshape(-1)
    s = np.asarray(stages, dtype=float).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(s)
    x, s = x[valid], s[valid]
    if len(x) < 2 or len(np.unique(s)) < 2:
        raise ValueError("At least two different stages are required")
    def one(t):
        num, den = 0.0, 0.0
        auc_num, auc_den = 0.0, 0.0
        uniq = np.sort(np.unique(s))
        for i, lo in enumerate(uniq):
            x_lo = t[s == lo]
            for hi in uniq[i + 1:]:
                x_hi = np.sort(t[s == hi])
                w = float(hi - lo)
                left = np.searchsorted(x_hi, x_lo, side="left")
                right = np.searchsorted(x_hi, x_lo, side="right")
                wins = float(np.sum(len(x_hi) - right))
                ties = float(np.sum(right - left))
                pairs = len(x_lo) * len(x_hi)
                num += w * wins
                den += w * pairs
                auc_num += w * (wins + 0.5 * ties) / pairs
                auc_den += w
        return num / den, auc_num / auc_den
    poa, poauc = one(x)
    flipped = False
    if orient:
        poa_reverse, poauc_reverse = one(1.0 - x)
        if poa_reverse > poa:
            poa, poauc, flipped = poa_reverse, poauc_reverse, True
    return {"POA": float(poa), "POAUC": float(poauc), "flipped_by_reference": flipped, "n_eval": int(len(x))}
