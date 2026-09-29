"""Score-fusion utilities for the engineered cascade.

The cascade combines a CHEAP score (the intent signal, ``baselines.
intent_signal``) with an EXPENSIVE learned score (GLM-4.6). This module
provides several fusion strategies and a held-out evaluation harness.

All strategies are reported (no cherry-picking). The headline comparison
is cascade vs GLM-alone at matched FPR, with McNemar + bootstrap CIs.
"""
from __future__ import annotations

from typing import Sequence

# Reuse the conformal non-conformity machinery's spirit but operate on
# arbitrary fused scores via evaluation.stats for thresholds/CIs.


def fuse_max(intent: Sequence[float], glm: Sequence[float]) -> list:
    """OR-style late fusion: the max of the two scores."""
    return [max(a, b) for a, b in zip(intent, glm)]


def fuse_weighted(intent: Sequence[float], glm: Sequence[float],
                  w_intent: float, w_glm: float) -> list:
    """Fixed linear combination (weights chosen a priori, not fit)."""
    return [w_intent * a + w_glm * b for a, b in zip(intent, glm)]


def fit_logistic(intent_cal, glm_cal, labels_cal):
    """Fit a logistic fuser on the CALIBRATION split only.

    Returns a function score(intent, glm) -> P(malicious) in [0,1]. Uses
    sklearn LogisticRegression on the 2-D feature [intent, glm] with
    class_weight balanced (InjecAgent is label-imbalanced).
    """
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    X = np.column_stack([list(intent_cal), list(glm_cal)])
    y = np.array([1 if l else 0 for l in labels_cal])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X, y)

    def fuse(intent, glm):
        Xs = np.column_stack([list(intent), list(glm)])
        return list(clf.predict_proba(Xs)[:, 1])
    return fuse, clf


def abstention_plan(intent_test, glm_test, tau_lo, tau_hi):
    """Cost-aware cascade: decide where the cheap intent score alone is
    trusted, so the GLM call is skipped.

      intent >= tau_hi  -> trust MALICIOUS (cheap layer decides; no GLM)
      intent <= tau_lo  -> trust BENIGN    (cheap layer decides; no GLM)
      otherwise         -> use GLM score

    Returns (fused_scores, glm_call_fraction). Where GLM is skipped we
    still need a score for ROC; we use intent (>=tau_hi region gets the
    high intent value, <=tau_lo gets its low value), which preserves
    ranking within the abstained regions.
    """
    out = []
    n_glm = 0
    for ci, cg in zip(intent_test, glm_test):
        if ci >= tau_hi:
            out.append(max(ci, 0.5 + (ci - tau_hi)))   # trust cheap-MAL
        elif ci <= tau_lo:
            out.append(ci * 0.5)                       # trust cheap-BEN
        else:
            out.append(cg)                             # use GLM
            n_glm += 1
    frac = n_glm / len(intent_test) if intent_test else 0.0
    return out, frac
