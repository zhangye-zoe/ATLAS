from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from atlas.losses import lag_objective, multilevel_alignment_loss


def test_alignment_is_active_when_pairing_is_zero():
    torch.manual_seed(0)
    n, d = 8, 4
    za = torch.randn(n, d, requires_grad=True)
    zb = torch.randn(n, d, requires_grad=True) + 1.0
    # No cell has both modalities: first half ATAC only, second half RNA only.
    ma = torch.tensor([1,1,1,1,0,0,0,0], dtype=torch.float32)
    mb = 1.0 - ma
    out = {"shared": {"atac": za, "rna": zb}, "h": torch.randn(n, d)}
    mmd, ot, con = multilevel_alignment_loss(
        out, {"atac": ma, "rna": mb}, epsilon=0.1, n_iter=10, max_cells=64
    )
    assert torch.isfinite(mmd) and torch.isfinite(ot)
    assert (mmd + ot).item() > 0
    assert con.item() == 0.0


def test_unpaired_lag_loss_remains_active_and_backpropagates():
    torch.manual_seed(1)
    n, g, d = 6, 5, 3
    pred = torch.randn(n, g, requires_grad=True)
    h = torch.randn(n, d)
    za = torch.randn(n, d)
    zb = torch.randn(n, d)
    ma = torch.tensor([1,1,1,0,0,0], dtype=torch.float32)
    mb = 1.0 - ma
    x = {"atac": torch.randn(n, g), "rna": torch.randn(n, g)}
    out = {
        "h": h,
        "shared": {"atac": za, "rna": zb},
        "lag": {"atac__to__rna": {"pred_target": pred}},
    }
    t = torch.linspace(0, 1, n)
    loss, parts = lag_objective(
        out, x, {"atac": ma, "rna": mb}, t,
        temporal_window=1.0, local_k=3, epsilon=0.2, n_iter=20
    )
    assert parts["paired"].item() == 0.0
    assert parts["unpaired"].item() > 0.0
    loss.backward()
    assert pred.grad is not None
    assert pred.grad[:3].abs().sum().item() > 0.0
    # Target-only predictions are not supervised by the source-only OT term.
    assert pred.grad[3:].abs().sum().item() == 0.0


def test_unpaired_source_prediction_loss_is_trainable():
    from atlas.losses import cross_prediction_objective
    torch.manual_seed(3)
    n, g, d = 6, 5, 3
    pred = torch.randn(n, g, requires_grad=True)
    ma = torch.tensor([1,1,1,0,0,0], dtype=torch.float32)
    mb = 1.0 - ma
    out = {
        "h": torch.randn(n, d),
        "shared": {"atac": torch.randn(n, d), "rna": torch.randn(n, d)},
        "cross": {"atac__to__rna": pred},
    }
    x = {"atac": torch.randn(n, g), "rna": torch.randn(n, g)}
    loss, parts = cross_prediction_objective(
        out, x, {"atac": ma, "rna": mb}, teacher_time=torch.linspace(0,1,n),
        temporal_window=1.0, local_k=3, epsilon=0.2, n_iter=20
    )
    assert parts["paired"].item() == 0.0
    assert parts["unpaired"].item() > 0.0
    loss.backward()
    assert pred.grad[:3].abs().sum().item() > 0
    assert pred.grad[3:].abs().sum().item() == 0
