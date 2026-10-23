"""Unit tests for proposed label-free local time-consensus training variant."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from atlas.temporal import build_temporal_teacher, fuse_time_candidates_local
from atlas.losses import time_distillation_loss


def test_local_fusion_is_deterministic_and_uses_no_labels():
    n = 48
    graph = np.linspace(0, 1, n, dtype=np.float32)
    diff = 1.0 - graph  # opposite sign; must be oriented to unsupervised graph
    attr = graph.copy()
    attr[16:25] = attr[16:25][::-1]  # local inconsistency
    nearest = np.stack([np.maximum(0, np.arange(n)-1),
                        np.minimum(n-1, np.arange(n)+1)], axis=1)
    times = {"graph": graph, "diff": diff, "attr": attr}
    args = dict(tau=0.15, lambda_agreement=3.0,
                lambda_local_smooth=1.0, anchor="graph")
    a, confidence, weights = fuse_time_candidates_local(times, nearest, **args)
    b, confidence2, weights2 = fuse_time_candidates_local(times, nearest, **args)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(confidence, confidence2)
    assert weights == weights2
    assert len(a) == n and np.isfinite(a).all()
    assert np.all((confidence >= 0.25) & (confidence <= 1.0))
    assert pytest.approx(sum(weights.values()), abs=1e-10) == 1.0
    assert np.corrcoef(a, graph)[0, 1] > 0.9
    assert np.mean(confidence[17:24]) < np.mean(confidence[:10])


def test_fusion_rejects_invalid_candidates_without_silent_fallback():
    x = np.linspace(0, 1, 20)
    knn = np.tile(np.arange(20), (1, 1)).reshape(20, 1)
    with pytest.raises(ValueError, match="Invalid time candidate"):
        fuse_time_candidates_local({"graph": x, "attr": x[:-1]}, knn)
    with pytest.raises(ValueError, match="must be positive"):
        fuse_time_candidates_local({"graph": x}, knn, tau=0.0)


def test_teacher_global_compatibility_and_local_confidence():
    rng = np.random.default_rng(7)
    h = np.stack(
        [np.linspace(0, 4, 60),
         np.sin(np.linspace(0, 4, 60)),
         rng.normal(0, .01, 60)], axis=1).astype(np.float32)
    shared = {"atac": h + 0.02, "rna": h - 0.02}
    masks = {"atac": np.ones(60), "rna": np.ones(60)}
    baseline = build_temporal_teacher(h, shared, masks, k_neighbors=8)
    baseline_explicit = build_temporal_teacher(
        h, shared, masks, k_neighbors=8, method="global"
    )
    np.testing.assert_array_equal(baseline.teacher, baseline_explicit.teacher)
    assert baseline.confidence is None
    updated = build_temporal_teacher(
        h, shared, masks, k_neighbors=8, method="local_consensus"
    )
    assert updated.method == "local_consensus"
    assert len(updated.teacher) == 60
    assert len(updated.confidence) == 60
    assert np.isfinite(updated.teacher).all()
    assert sum(updated.weights.values()) == pytest.approx(1.0)


def test_confidence_weighted_time_distillation_has_finite_gradient():
    pred = torch.tensor([0.2, 0.7, 0.8], requires_grad=True)
    teacher = torch.tensor([0.1, 0.4, 0.9])
    h = torch.ones((3, 2))
    confidence = torch.tensor([1.0, 0.25, 0.75])
    distill, reg = time_distillation_loss(
        pred, teacher, h, confidence=confidence, target_std=0.2
    )
    expected = (
        (confidence * (pred - teacher).square()).sum() / confidence.sum()
    )
    assert distill.item() == pytest.approx(expected.item(), abs=1e-7)
    (distill + reg).backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    old_loss, _ = time_distillation_loss(pred.detach(), teacher, h)
    assert old_loss.item() == pytest.approx(
        torch.nn.functional.mse_loss(pred.detach(), teacher).item()
    )
