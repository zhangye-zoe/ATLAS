"""Stage-stratified, label-masked T3 selection and held-out evaluation.

Only reference-stage *labels* are partitioned, not unlabeled validation cells.
Both subsets use the same full cohort for transductive graph-time inference,
but disjoint monitor vs reporting annotations. Selection by stage labels is a
supervised checkpoint-selection protocol, not label-free model selection.
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
    """Create disjoint label masks with the same full transductive cell cohort.

    Singleton reference stages are reserved for reporting only; all cells'
    UNLABELED modalities remain available to both inferential views, and
    only stage labels are withheld on the complementary subset.
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
        rng.shuffle(group)
        # A singleton cannot be assigned to both partitions. Never put it
        # into selection; reserve it for an unbiased report instead.
        n_select = (
            int(np.clip(round(monitor_fraction * len(group)), 1, len(group) - 1))
            if len(group) >= 2 else 0
        )
        monitored.extend(group[:n_select].tolist())
        stage_counts[str(int(stage))] = {
            "all": int(len(group)), "monitor": n_select,
            "report": int(len(group) - n_select)
        }

    monitor_idx = np.sort(np.asarray(monitored, dtype=np.int64))
    report_idx = np.setdiff1d(np.arange(len(stages)), monitor_idx)
    if (len(np.unique(stages[monitor_idx])) < 2
            or len(np.unique(stages[np.intersect1d(report_idx, valid)])) < 2):
        raise ValueError(
            "Both reference-label partitions must contain >=2 known stages; "
            "inspect stage counts or increase monitor_fraction"
        )
    if len(np.intersect1d(monitor_idx, report_idx)):
        raise AssertionError("Checkpoint-selection/report cell leakage detected")

    # IDs, not only row positions, are checked for overlap.
    monitor_ids = data.val_cell_ids[monitor_idx].astype(str)
    report_ids = data.val_cell_ids[report_idx].astype(str)
    if len(set(monitor_ids) & set(report_ids)):
        raise AssertionError("Duplicate cell IDs occur across evaluation partitions")

    hidden_label = "__t3_heldout_label__"
    if hidden_label in data.stage_map:
        raise ValueError("Reserved label sentinel conflicts with stage_map")
    monitor_labels = np.asarray(data.labels, dtype=object).copy()
    report_labels = np.asarray(data.labels, dtype=object).copy()
    monitor_labels[report_idx] = hidden_label
    report_labels[monitor_idx] = hidden_label

    info = {
        "policy": "stage-stratified disjoint labels; common transductive cell cohort",
        "unlabeled_validation_inputs_shared_across_partitions": True,
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
        replace(data, labels=monitor_labels),
        replace(data, labels=report_labels),
        info,
    )
