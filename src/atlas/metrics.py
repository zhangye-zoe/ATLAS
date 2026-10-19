from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, roc_auc_score


def safe_cor(x: np.ndarray, y: np.ndarray) -> float:
    """Exact correlation helper used by the retained CORAL Table-2 script."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size <= 1 or y.size <= 1:
        return float("nan")
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def hit_at_k(cross_dist: np.ndarray, k: int = 5) -> float:
    """Exact hit@k helper from the original CORAL ratio-loop evaluation."""
    nq = cross_dist.shape[0]
    hits = []
    for i in range(nq):
        ord_idx = np.argsort(cross_dist[i, :])[: min(k, cross_dist.shape[1])]
        hits.append(i in ord_idx)
    return float(np.mean(hits)) if len(hits) > 0 else float("nan")


def evaluate_t1_coral_style(
    pred_mat: np.ndarray,
    true_mat: np.ndarray,
    seed: int = 1234,
    max_pcs: int = 30,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Reproduce the original CORAL/scMRDR-style T1 protocol exactly.

    IMPORTANT: FOSCTTM and PED in the original CORAL Table-2 code were NOT
    computed between ATAC and RNA latent embeddings.  They were computed as:

      1) ATAC -> predicted RNA using the source-only prediction head;
      2) stack [true RNA; predicted RNA] in the SAME RNA feature space;
      3) fit one joint PCA (up to 30 PCs);
      4) compute cross distances between true-RNA PCA and predicted-RNA PCA;
      5) FOSCTTM_i = mean_j[d(i,j) < d(i,i)] (denominator is N, exactly as CORAL);
      6) PED_i = d(i,i), averaged over cells.

    This function intentionally preserves that historical protocol so numbers
    are directly comparable to the retained CORAL results.
    """
    true_cells_by_features = np.asarray(true_mat, dtype=float)
    pred_cells_by_features = np.asarray(pred_mat, dtype=float)
    if true_cells_by_features.shape != pred_cells_by_features.shape:
        raise ValueError(
            "CORAL-style T1 requires predicted and true RNA matrices with the same shape; "
            f"got {pred_cells_by_features.shape} vs {true_cells_by_features.shape}."
        )
    if true_cells_by_features.ndim != 2:
        raise ValueError("CORAL-style T1 expects 2D cell-by-feature matrices.")

    mix = np.vstack([true_cells_by_features, pred_cells_by_features])
    n_components = min(int(max_pcs), mix.shape[0] - 1, mix.shape[1])
    if n_components < 2:
        raise ValueError("Too few dimensions to run PCA for T1 evaluation.")

    pca = PCA(n_components=n_components, random_state=int(seed))
    emb = pca.fit_transform(mix)
    n_q = true_cells_by_features.shape[0]
    true_emb = emb[:n_q]
    pred_emb = emb[n_q:]
    cross_dist = pairwise_distances(true_emb, pred_emb)

    fos, ped = [], []
    for i in range(cross_dist.shape[0]):
        true_d = cross_dist[i, i]
        # Deliberately identical to the original CORAL implementation.
        # The denominator is all N targets; the diagonal is false under strict <.
        fos.append(np.mean(cross_dist[i] < true_d))
        ped.append(true_d)

    metrics = {
        "FOSCTTM": float(np.mean(fos)),
        "PED": float(np.mean(ped)),
        "Top1_ACC": hit_at_k(cross_dist, 1),
        "Top5_ACC": hit_at_k(cross_dist, 5),
        "Top10_ACC": hit_at_k(cross_dist, 10),
        "PCA_components": int(n_components),
    }
    return metrics, true_emb, pred_emb


def evaluate_t2(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    """Reproduce the original CORAL T2 true-vs-predicted RNA metrics."""
    pred_mat = np.asarray(pred, dtype=float)
    true_mat = np.asarray(true, dtype=float)
    if pred_mat.shape != true_mat.shape:
        raise ValueError(f"T2 shape mismatch: {pred_mat.shape} vs {true_mat.shape}")
    cell_corr = [safe_cor(pred_mat[i], true_mat[i]) for i in range(pred_mat.shape[0])]
    gene_corr = [safe_cor(pred_mat[:, j], true_mat[:, j]) for j in range(pred_mat.shape[1])]
    value_rmse = float(np.sqrt(np.mean((pred_mat - true_mat) ** 2)))
    cell_mean = float(np.nanmean(cell_corr))
    gene_mean = float(np.nanmean(gene_corr))
    return {
        "Pearson": cell_mean,  # compatibility alias used by existing Table-2 summary code
        "Pearson_cell": cell_mean,
        "Pearson_gene": gene_mean,
        "RMSE": value_rmse,
    }


def labels_to_stages(labels: Sequence[str], stage_map: Mapping[str, int]) -> np.ndarray:
    return np.asarray([stage_map.get(str(x), np.nan) for x in labels], dtype=np.float64)



def pairwise_ordering_accuracy(time: np.ndarray, stages: np.ndarray) -> float:
    """Exact POA from the ATLAS paper.

    For every pair of cells with different reference stages:
        |s_i-s_j| * I[(t_i-t_j)(s_i-s_j)>0]
    is accumulated and normalized by the total stage-distance weight.
    """
    t = np.asarray(time, dtype=np.float64)
    s = np.asarray(stages, dtype=np.float64)
    valid = np.isfinite(t) & np.isfinite(s)
    t, s = t[valid], s[valid]

    n = len(t)
    if n < 2:
        return float("nan")

    num, den = 0.0, 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if s[i] == s[j]:
                continue
            w = abs(float(s[i] - s[j]))
            den += w
            if (t[i] - t[j]) * (s[i] - s[j]) > 0:
                num += w

    return float(num / den) if den > 0 else float("nan")


def pairwise_ordering_auc(time: np.ndarray, stages: np.ndarray) -> float:
    """Exact weighted POAUC from the ATLAS paper."""
    t = np.asarray(time, dtype=np.float64)
    s = np.asarray(stages, dtype=np.float64)
    valid = np.isfinite(t) & np.isfinite(s)
    t, s = t[valid], s[valid]

    unique = np.sort(np.unique(s))
    num, den = 0.0, 0.0

    for ia, a in enumerate(unique):
        for b in unique[ia + 1:]:
            mask = (s == a) | (s == b)
            y = (s[mask] == b).astype(np.int64)
            if len(np.unique(y)) < 2:
                continue
            auc = roc_auc_score(y, t[mask])
            w = abs(float(a-b))
            num += w * auc
            den += w

    return float(num / den) if den > 0 else float("nan")

def orient_time_for_reference(time: np.ndarray, stages: np.ndarray) -> np.ndarray:
    """Evaluation-only orientation for the sign ambiguity of unsupervised pseudotime."""
    t = np.asarray(time, dtype=np.float32)
    direct = pairwise_ordering_accuracy(t, stages)
    reverse = pairwise_ordering_accuracy(1.0 - t, stages)
    return (1.0 - t) if np.isfinite(reverse) and reverse > direct else t


def evaluate_t3(time: np.ndarray, stages: np.ndarray, orient: bool = True) -> Dict[str, float]:
    t = orient_time_for_reference(time, stages) if orient else np.asarray(time)
    return {
        "POA": pairwise_ordering_accuracy(t, stages),
        "POAUC": pairwise_ordering_auc(t, stages),
    }
