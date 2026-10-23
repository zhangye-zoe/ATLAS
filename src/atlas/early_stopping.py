"""Leakage-aware stage-stratified T3 selection/held-out evaluation partition.

Only validation reference stages are used for the partition and monitoring; they
never enter ATLAS reconstruction/alignment/temporal teacher losses. Selection
by stage labels makes this a *supervised checkpoint-selection protocol*, not
label-free model selection. Final T3 must be reported on held-out stage labels.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, Tuple

import numpy as np

from .data import Table2Data
from .metrics import labels_to_stages


def subset_validation(data: Table2Data, indices: np.ndarray) -> Table2Data:
    """View a cell-aligned validation subset while keeping training arrays intact."""
    ix = np.asarray(indices, dtype=np.int64)
    n = len(data.val_cell_ids)
    if ix.ndim != 1 or np.any(ix < 0) or np.any(ix >= n) or len(np.unique(ix)) != len(ix):
        raise ValueError("Invalid validation subset indices")
    return replace(
        data,
        val_cell_ids=data.val_cell_ids[ix].copy(),
        val_modalities={k: x[ix].copy() for k, x in data.val_modalities.items()},
        val_modalities_raw={k: x[ix].copy() for k, x in data.val_modalities_raw.items()},
        labels=None if data.labels is None else np.asarray(data.labels)[ix].copy(),
    )


def split_monitor_report(
    data: Table2Data, monitor_fraction: float = 0.30, seed: int = 1234
) -> Tuple[Table2Data, Table2Data, Dict]:
    """Stratify reference-labeled validation cells into selection vs report.

    Every stage must have at least two cells, so both partitions contain it.
    Unknown stage labels are retained in the report subset, never in selection.
    The same seed and data split produce identical partitions across arms.
    """
    if data.labels is None:
        raise ValueError("T3 early stopping requires independent reference stage labels")
    if not 0.05 <= monitor_fraction <= 0.80:
        raise ValueError("monitor_fraction must be between 0.05 and 0.80")
    stages = labels_to_stages(data.labels, data.stage_map)
    valid = np.flatnonzero(np.isfinite(stages))
    if len(valid) < 20:
        raise ValueError("At least 20 stage-annotated validation cells are required")
    unique = np.unique(stages[valid])
    if len(unique) < 2:
        raise ValueError("Need at least 2 known reference stages for T3 POAUC")

    rng = np.random.default_rng(seed)
    monitored = []
    stage_counts = {}
    for stage in unique:
        group = valid[stages[valid] == stage].copy()
        if len(group) < 2:
            raise ValueError(
                f"Stage {stage} has only {len(group)} cell(s); "
                "cannot create selection and held-out subsets without leakage"
            )
        rng.shuffle(group)
        n_select = int(np.clip(round(monitor_fraction * len(group)), 1, len(group) - 1))
        monitored.extend(group[:n_select].tolist())
        stage_counts[str(int(stage))] = {
            "all": int(len(group)), "monitor": n_select,
            "report": int(len(group) - n_select)
        }

    monitor_idx = np.sort(np.asarray(monitored, dtype=np.int64))
    report_idx = np.setdiff1d(np.arange(len(stages)), monitor_idx)
    if len(np.intersect1d(monitor_idx, report_idx)):
        raise AssertionError("Checkpoint-selection/report cell leakage detected")

    # IDs, not only row positions, are checked for overlap.
    monitor_ids = data.val_cell_ids[monitor_idx].astype(str)
    report_ids = data.val_cell_ids[report_idx].astype(str)
    if len(set(monitor_ids) & set(report_ids)):
        raise AssertionError("Duplicate cell IDs occur across evaluation partitions")

    info = {
        "policy": "stage-stratified fixed selection/report holdout",
        "checkpoint_selection_uses_reference_labels": True,
        "training_losses_use_reference_labels": False,
        "monitor_fraction": float(monitor_fraction),
        "split_seed": int(seed),
        "all_validation_cells": len(stages),
        "monitor_cells": len(monitor_idx),
        "report_cells": len(report_idx),
        "stage_counts": stage_counts,
        "monitor_cell_ids": monitor_ids.tolist(),
        "report_cell_ids": report_ids.tolist(),
    }
    return (
        subset_validation(data, monitor_idx),
        subset_validation(data, report_idx),
        info,
    )
