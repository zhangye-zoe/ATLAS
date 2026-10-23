from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding
from sklearn.neighbors import NearestNeighbors


def normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def rank01(x: np.ndarray) -> np.ndarray:
    if len(x) <= 1:
        return np.zeros(len(x), dtype=np.float32)
    return ((rankdata(x, method="average") - 1.0) / (len(x) - 1.0)).astype(np.float32)


def knn_graph(h: np.ndarray, k: int = 15) -> Tuple[np.ndarray, np.ndarray, sparse.csr_matrix]:
    n = h.shape[0]
    k = min(max(2, k + 1), n)
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(h)
    dist, idx = nn.kneighbors(h)
    # remove self neighbor
    dist, idx = dist[:, 1:], idx[:, 1:]
    rows = np.repeat(np.arange(n), idx.shape[1])
    cols = idx.reshape(-1)
    vals = dist.reshape(-1)
    g = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    g = g.minimum(g.T) + g.maximum(g.T)
    return idx, dist, g


def graph_pseudotime(h: np.ndarray, k: int = 15) -> np.ndarray:
    """Unsupervised geodesic progression with a deterministic peripheral root.

    Static snapshot pseudotime is orientation-ambiguous without an external root.
    We choose a peripheral latent point deterministically and keep reference-based
    orientation strictly in evaluation, never in teacher construction.
    """
    idx, dist, g = knn_graph(h, k)
    centroid = h.mean(axis=0, keepdims=True)
    peripheral = np.linalg.norm(h - centroid, axis=1)
    root = int(np.argmax(peripheral))
    d = dijkstra(g, directed=False, indices=root)
    finite = np.isfinite(d)
    if not finite.all():
        fill = np.nanmax(d[finite]) if finite.any() else 0.0
        d[~finite] = fill
    return normalize01(d)


def diffusion_pseudotime(h: np.ndarray, graph_time: np.ndarray, k: int = 15) -> np.ndarray:
    n = h.shape[0]
    if n < 4:
        return graph_time.copy()
    # SpectralEmbedding is a reproducible diffusion-like coordinate for this refactor.
    # The sign is oriented against graph-time without labels.
    emb = SpectralEmbedding(
        n_components=min(3, max(1, n - 2)),
        n_neighbors=min(k, max(2, n - 1)),
        affinity="nearest_neighbors",
        random_state=0,
    ).fit_transform(h)
    best = emb[:, 0]
    best_corr = -np.inf
    for c in range(emb.shape[1]):
        v = emb[:, c]
        corr = abs(np.corrcoef(v, graph_time)[0, 1]) if np.std(v) > 0 else 0.0
        if np.isfinite(corr) and corr > best_corr:
            best_corr, best = corr, v
    if np.corrcoef(best, graph_time)[0, 1] < 0:
        best = -best
    return normalize01(best)


def attribution_proxy_time(
    h: np.ndarray,
    shared: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
    graph_time: np.ndarray,
) -> np.ndarray:
    """Explicit reproducible proxy for the manuscript's underspecified t_attr.

    The current manuscript names an attribution-aware signal but does not define
    its exact estimator. Legacy CORAL also has no faithful implementation. Rather
    than silently inventing a hidden procedure, this refactor uses a documented
    latent-axis attribution proxy: modality-specific projections onto the leading
    fused progression axis are combined only where each modality is observed.

    This function is deliberately isolated so it can be replaced once the final
    t_attr definition is fixed in the manuscript.
    """
    pca = PCA(n_components=1, random_state=0).fit(h)
    axis = pca.components_[0].astype(np.float32)
    if np.corrcoef(h @ axis, graph_time)[0, 1] < 0:
        axis = -axis
    scores = []
    weights = []
    for m in sorted(shared):
        s = shared[m] @ axis
        mask = masks[m].astype(np.float32)
        scores.append(s * mask)
        weights.append(mask)
    num = np.sum(np.stack(scores, axis=0), axis=0)
    den = np.sum(np.stack(weights, axis=0), axis=0)
    den = np.maximum(den, 1.0)
    return normalize01(num / den)


def _smoothness_risk(t: np.ndarray, knn_idx: np.ndarray) -> float:
    return float(np.mean(np.abs(t[:, None] - t[knn_idx])))


def _modality_risk(
    t: np.ndarray,
    shared: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
) -> float:
    names = sorted(shared)
    if len(names) < 2:
        return 0.0
    vals = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ia = np.where(masks[a] > 0)[0]
            ib = np.where(masks[b] > 0)[0]
            if len(ia) == 0 or len(ib) == 0:
                continue
            za = shared[a][ia]
            zb = shared[b][ib]
            nn = NearestNeighbors(n_neighbors=1).fit(zb)
            _, j = nn.kneighbors(za)
            vals.append(np.mean(np.abs(t[ia] - t[ib[j[:, 0]]])))
    return float(np.mean(vals)) if vals else 0.0


def _spread_risk(t: np.ndarray, target_std: float = 0.20) -> float:
    return float(max(0.0, target_std - float(np.std(t))))


@dataclass
class TemporalTeacher:
    teacher: np.ndarray
    candidates: Dict[str, np.ndarray]
    risks: Dict[str, Dict[str, float]]
    weights: Dict[str, float]
    knn_idx: np.ndarray
    confidence: Optional[np.ndarray] = None
    method: str = "global"


def build_temporal_teacher(
    h: np.ndarray,
    shared: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
    k_neighbors: int = 15,
    tau: float = 0.85,
    lambda_sm: float = 1.0,
    lambda_mv: float = 1.0,
    lambda_sp: float = 1.0,
    target_std: float = 0.20,
    method: str = "global",
    consensus_tau: float = 0.15,
    lambda_agreement: float = 3.0,
    lambda_local_smooth: float = 1.0,
) -> TemporalTeacher:
    graph = graph_pseudotime(h, k_neighbors)
    diff = diffusion_pseudotime(h, graph, k_neighbors)
    attr = attribution_proxy_time(h, shared, masks, graph)
    candidates = {"graph": graph, "diff": diff, "attr": attr}
    knn_idx, _, _ = knn_graph(h, k_neighbors)

    risks: Dict[str, Dict[str, float]] = {}
    total_risk = {}
    for name, t in candidates.items():
        r_sm = _smoothness_risk(t, knn_idx)
        r_mv = _modality_risk(t, shared, masks)
        r_sp = _spread_risk(t, target_std)
        risks[name] = {"smooth": r_sm, "modality": r_mv, "spread": r_sp}
        total_risk[name] = lambda_sm * r_sm + lambda_mv * r_mv + lambda_sp * r_sp

    if method == "global":
        # Preserve the published/refactored baseline exactly.
        keys = list(candidates)
        raw = np.asarray([np.exp(-total_risk[k] / max(tau, 1e-6)) for k in keys], dtype=np.float64)
        raw /= raw.sum() if raw.sum() > 0 else 1.0
        weights = {k: float(w) for k, w in zip(keys, raw)}
        teacher = np.zeros(h.shape[0], dtype=np.float64)
        for k in keys:
            teacher += weights[k] * rank01(candidates[k])
        teacher = normalize01(teacher)
        return TemporalTeacher(teacher, candidates, risks, weights, knn_idx, method=method)
    if method != "local_consensus":
        raise ValueError(f"Unknown temporal fusion method: {method}")

    teacher, confidence, weights = fuse_time_candidates_local(
        candidates,
        knn_idx,
        tau=consensus_tau,
        lambda_agreement=lambda_agreement,
        lambda_local_smooth=lambda_local_smooth,
        candidate_global_risk=total_risk,
        anchor="graph",
    )
    return TemporalTeacher(
        teacher=teacher, candidates=candidates, risks=risks,
        weights=weights, knn_idx=knn_idx, confidence=confidence, method=method,
    )



def fuse_time_candidates_local(
    candidates: Mapping[str, np.ndarray],
    knn_idx: np.ndarray,
    *,
    tau: float = 0.15,
    lambda_agreement: float = 3.0,
    lambda_local_smooth: float = 1.0,
    candidate_global_risk: Optional[Mapping[str, float]] = None,
    anchor: str = "graph",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Reference-label-free, per-cell reliability fusion over *ranked* time candidates.

    Risks combine local graph discontinuity, disagreement with candidate median,
    and optional training-only global risks. Opposite candidate signs are aligned
    to an unsupervised anchor by rank correlation (no cell labels). Returns
    (fused_time[N], confidence[N], mean_candidate_weights).

    This is a proposed ablation, not a verified increase in POA/POAUC.
    """
    names = list(candidates)
    if not names or anchor not in candidates:
        raise ValueError("Fusion needs >=1 candidate including the unsupervised anchor")
    kidx = np.asarray(knn_idx, dtype=np.int64)
    n = np.asarray(candidates[anchor]).size
    if n < 3 or kidx.ndim != 2 or kidx.shape[0] != n or kidx.shape[1] < 1:
        raise ValueError("Need >=3 cells and at least one kNN per cell")
    if kidx.min() < 0 or kidx.max() >= n:
        raise ValueError("kNN indices out of bounds")
    if tau <= 0 or lambda_agreement < 0 or lambda_local_smooth < 0:
        raise ValueError("tau must be positive and risk weights nonnegative")

    anchor_rank = rank01(np.asarray(candidates[anchor], dtype=float))
    ranks, inactive = [], []
    for name in names:
        t = np.asarray(candidates[name], dtype=float).reshape(-1)
        if len(t) != n or not np.isfinite(t).all():
            raise ValueError(f"Invalid time candidate {name}: must be finite length {n}")
        inactive.append(float(np.std(t)) < 1e-8)
        r = rank01(t).astype(np.float64)
        if name != anchor and np.std(r) > 1e-8 and np.std(anchor_rank) > 1e-8:
            if np.corrcoef(r, anchor_rank)[0, 1] < 0:
                r = 1.0 - r
        ranks.append(r)
    arr = np.stack(ranks, axis=0)  # [C,N]
    median = np.median(arr, axis=0)
    agreement_risk = np.abs(arr - median[None, :])
    local_risk = np.abs(arr[:, :, None] - arr[:, kidx]).mean(axis=2)
    global_risk = candidate_global_risk or {}
    g = np.asarray([float(global_risk.get(name, 0.0)) for name in names])[:, None]
    score = (
        lambda_agreement * agreement_risk
        + lambda_local_smooth * local_risk
        + g
    )
    for i, bad in enumerate(inactive):
        if bad and names[i] != anchor:
            score[i, :] = 1e6
    # Numerically stable local softmax across candidate methods.
    logits = -score / tau
    logits -= np.max(logits, axis=0, keepdims=True)
    w = np.exp(logits)
    w /= np.maximum(w.sum(axis=0, keepdims=True), 1e-12)
    fused = np.sum(w * arr, axis=0)
    # Rank-space fusion retains the relative ordering (critical for T3).
    fused = normalize01(fused)
    # Agreement-derived reliability, never based on ground-truth stage labels.
    disagreement = np.sqrt(np.sum(w * (arr - fused[None, :]) ** 2, axis=0))
    confidence = np.clip(np.exp(-4.0 * disagreement), 0.25, 1.0).astype(np.float32)
    mean_weights = {name: float(w[i].mean()) for i, name in enumerate(names)}
    return fused.astype(np.float32), confidence, mean_weights
