"""Regression tests for legacy T3 diagnostic parity with current ATLAS scoring."""
import numpy as np
import pytest

from atlas.legacy_t3 import get_t3_candidates, score_ordering_fast
from atlas.metrics import evaluate_t3


@pytest.mark.parametrize("n", [24, 120])
@pytest.mark.parametrize("with_ties", [False, True])
def test_fast_ordering_matches_reference(n, with_ties):
    rng = np.random.default_rng(1234 + n)
    t = rng.uniform(size=n)
    stages = np.repeat(np.arange(4), n // 4)
    if with_ties:
        t = (t * 4).astype(int) / 4.0
    for orient in (False, True):
        fast = score_ordering_fast(t, stages, orient=orient)
        baseline = evaluate_t3(t, stages, orient=orient)
        assert fast["POA"] == pytest.approx(baseline["POA"], abs=1e-12)
        assert fast["POAUC"] == pytest.approx(baseline["POAUC"], abs=1e-12)


def test_label_free_time_candidates_and_label_guided_guard():
    rng = np.random.default_rng(5)
    h = rng.normal(size=(55, 5)).astype(np.float32)
    head = rng.uniform(size=55).astype(np.float32)
    times, metadata = get_t3_candidates(
        h, head, modes=["head", "graph_peripheral"], graph_k=10
    )
    assert set(times) == {"head", "graph_peripheral"}
    assert all(v.shape == (55,) and np.isfinite(v).all() for v in times.values())
    with pytest.raises(ValueError, match="requires --root-labels"):
        get_t3_candidates(h, head, modes=["graph_root"])


def test_no_cherry_picking_implicit_mode():
    rng = np.random.default_rng(33)
    h = rng.normal(size=(25, 4))
    t = np.linspace(0, 1, 25)
    modes = ["head", "graph_peripheral"]
    candidate, _ = get_t3_candidates(h, t, modes=modes, graph_k=5)
    assert list(candidate) == modes
