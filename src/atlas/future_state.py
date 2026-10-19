"""Evaluation metrics for held-out future state or fate scores.

This deliberately does not assume that pseudotime bins are true future labels.
A benchmark-specific ground-truth generation pipeline must provide `y_true`.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score


def evaluate_future_state(y_true, y_score, threshold=0.5):
    """Return AUROC and F1 for binary targets; multiclass scores also supported.

    Parameters
    ----------
    y_true: 1-D array of experimentally justified integer fate/state labels.
    y_score: 1-D positive-class probabilities or N x C multiclass probabilities.
    """
    y_true=np.asarray(y_true)
    score=np.asarray(y_score, dtype=float)
    if y_true.ndim!=1 or score.shape[0]!=len(y_true):
        raise ValueError("y_true must be 1D; scores must have matching rows")
    if not np.isfinite(score).all():
        raise ValueError("Non-finite scores")
    if score.ndim==1 or (score.ndim==2 and score.shape[1]==2):
        if len(np.unique(y_true))!=2:
            raise ValueError("Binary AUROC requires both classes")
        p=score if score.ndim==1 else score[:,1]
        return {"F1":float(f1_score(y_true,p>=threshold)),"AUROC":float(roc_auc_score(y_true,p))}
    if score.ndim==2 and score.shape[1]>2:
        if set(np.unique(y_true))!=set(range(score.shape[1])):
            raise ValueError("All multiclass labels 0..C-1 must be present")
        return {"F1":float(f1_score(y_true,score.argmax(1),average="macro")),
                "AUROC":float(roc_auc_score(y_true,score,multi_class="ovr",average="macro"))}
    raise ValueError("Unsupported score shape")
