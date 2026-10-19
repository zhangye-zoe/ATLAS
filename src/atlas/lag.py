from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import numpy as np


@dataclass
class LagBank:
    bank: np.ndarray      # [N, K, D]
    valid: np.ndarray     # [N, K]


def build_source_lag_bank(
    source_shared: np.ndarray,
    source_mask: np.ndarray,
    time: np.ndarray,
    n_bins: int = 4,
) -> LagBank:
    """Construct Eq.-14 source-modality lag bank by temporal interval pooling.

    Critical fix over legacy CORAL: bank entries pool source-specific shared
    latent states z_s^(a), never the fused h representation.
    """
    z = np.asarray(source_shared, dtype=np.float32)
    t = np.asarray(time, dtype=np.float32).reshape(-1)
    mask = np.asarray(source_mask).astype(bool)
    n, d = z.shape
    src_idx = np.where(mask)[0]
    if len(src_idx) == 0:
        return LagBank(np.zeros((n, n_bins, d), np.float32), np.zeros((n, n_bins), bool))

    order = src_idx[np.argsort(t[src_idx])]
    ts = t[order]
    zs = z[order]
    csum = np.vstack([np.zeros((1, d), dtype=np.float64), np.cumsum(zs, axis=0, dtype=np.float64)])

    bank = np.zeros((n, n_bins, d), dtype=np.float32)
    valid = np.zeros((n, n_bins), dtype=bool)
    width = 1.0 / max(n_bins, 1)
    for i in range(n):
        ti = float(t[i])
        for k in range(n_bins):
            # B_1 is the most recent earlier interval; larger k means older history.
            lower = max(0.0, ti - (k + 1) * width)
            upper = ti - k * width
            if upper <= 0:
                continue
            lo = int(np.searchsorted(ts, lower, side="left"))
            hi = int(np.searchsorted(ts, upper, side="left"))  # strict tj < ti at k=0
            if hi <= lo:
                continue
            bank[i, k] = ((csum[hi] - csum[lo]) / float(hi - lo)).astype(np.float32)
            valid[i, k] = True
    return LagBank(bank, valid)


def build_lag_banks(
    shared: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
    time: np.ndarray,
    edges=(('atac', 'rna'),),
    n_bins: int = 4,
) -> Dict[str, LagBank]:
    out = {}
    for src, tgt in edges:
        out[f"{src}__to__{tgt}"] = build_source_lag_bank(
            shared[src], masks[src], time, n_bins=n_bins
        )
    return out


def build_query_lag_bank(
    source_shared: np.ndarray,
    source_time: np.ndarray,
    query_time: np.ndarray,
    n_bins: int = 4,
) -> LagBank:
    """Pool training source states into lag bins for arbitrary query cells."""
    z = np.asarray(source_shared, dtype=np.float32)
    ts = np.asarray(source_time, dtype=np.float32).reshape(-1)
    tq = np.asarray(query_time, dtype=np.float32).reshape(-1)
    if z.shape[0] != ts.shape[0]:
        raise ValueError("source_shared/source_time length mismatch")
    order = np.argsort(ts)
    ts = ts[order]
    zs = z[order]
    d = z.shape[1]
    csum = np.vstack([np.zeros((1, d), dtype=np.float64), np.cumsum(zs, axis=0, dtype=np.float64)])
    bank = np.zeros((len(tq), n_bins, d), dtype=np.float32)
    valid = np.zeros((len(tq), n_bins), dtype=bool)
    width = 1.0 / max(n_bins, 1)
    for i, ti in enumerate(tq):
        for k in range(n_bins):
            lower = max(0.0, float(ti) - (k + 1) * width)
            upper = float(ti) - k * width
            if upper <= 0:
                continue
            lo = int(np.searchsorted(ts, lower, side="left"))
            hi = int(np.searchsorted(ts, upper, side="left"))
            if hi > lo:
                bank[i, k] = ((csum[hi] - csum[lo]) / float(hi - lo)).astype(np.float32)
                valid[i, k] = True
    return LagBank(bank, valid)
