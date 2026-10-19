from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from atlas.model import ATLAS


def test_masked_target_does_not_change_fused_source_representation():
    torch.manual_seed(2)
    model = ATLAS({"atac": 7, "rna": 7}, shared_dim=4, private_dim=3,
                  encoder_hidden=(12, 8), decoder_hidden=(8, 12), dropout=0.0)
    model.eval()
    atac = torch.randn(5, 7)
    rna1 = torch.randn(5, 7)
    rna2 = rna1 + 1000.0
    masks = {"atac": torch.ones(5), "rna": torch.zeros(5)}
    with torch.no_grad():
        o1 = model({"atac": atac, "rna": rna1}, masks, sample=False)
        o2 = model({"atac": atac, "rna": rna2}, masks, sample=False)
    assert torch.allclose(o1["h"], o2["h"], atol=1e-6)
    assert torch.allclose(o1["time"], o2["time"], atol=1e-6)
