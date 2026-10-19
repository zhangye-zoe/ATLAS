from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atlas.utils import paired_ratio_to_split_label


def test_pair_ratio_mapping():
    assert paired_ratio_to_split_label(0) == "single_100"
    assert paired_ratio_to_split_label(20) == "single_080"
    assert paired_ratio_to_split_label(40) == "single_060"
    assert paired_ratio_to_split_label(100) == "single_000"
