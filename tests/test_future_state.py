import numpy as np
from atlas.future_state import evaluate_future_state

def test_future_binary_metrics():
    metrics=evaluate_future_state([0,1,0,1], [0.1,0.9,0.2,0.8])
    assert metrics["AUROC"]==1.0 and metrics["F1"]==1.0
