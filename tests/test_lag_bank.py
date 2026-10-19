import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atlas.lag import build_source_lag_bank


def test_lag_bank_pools_only_observed_source_shared_states():
    # Source values are deliberately easy to recognize. Cell 1 is unobserved and
    # has an extreme value that must never enter any bank average.
    z = np.asarray([[1.0], [999.0], [3.0], [5.0]], dtype=np.float32)
    mask = np.asarray([1, 0, 1, 1], dtype=np.float32)
    t = np.asarray([0.10, 0.20, 0.30, 0.70], dtype=np.float32)
    bank = build_source_lag_bank(z, mask, t, n_bins=2)

    # Query at t=.70, recent bin [.20,.70) contains only observed source cell t=.30.
    assert bank.valid[3, 0]
    assert np.allclose(bank.bank[3, 0], [3.0])
    # Extreme unobserved 999 must never contaminate any valid bank entry.
    assert not np.isclose(bank.bank[bank.valid], 999.0).any()
