"""Statistical utilities for the paper's empirical evaluation.

All functions operate on plain arrays (labels / scores / predictions),
*not* on live verifier objects, so they are usable both on freshly
computed verdicts and on cached per-trajectory prediction files. This
is what lets us report bootstrap CIs and exact McNemar tests without
re-querying any LLM.

Conventions
-----------
- ``labels``   : list[bool] or list[int in {0,1}] — True/1 == malicious.
- ``scores``   : list[float] — *higher == more malicious*.
- ``preds``    : list[bool] — True == flagged/detected (above threshold).

These arrays must be parallel (same length, same ordering, same
trajectories). The caller is responsible for that pairing; these
helpers do not re-order.

Implemented (pure Python, no numpy, for auditability):
  - confusion(labels, preds) -> dict
  - tpr/fpr/precision/f1 from a confusion dict
  - roc(labels, scores) -> list[(fpr, tpr, threshold)]
  - auc(labels, scores) -> float            (rank / Mann-Whitney, tie-safe)
  - threshold_at_fpr(labels, scores, fpr) -> float
  - tpr_at_fpr(labels, scores, fpr) -> float
  - bootstrap_ci(labels, secondary, stat_fn, ...) -> (point, lo, hi)
  - bootstrap_paired_diff(labels, preds_a, preds_b, stat_fn, ...) -> (d, lo, hi)
  - mcnemar(labels, preds_a, preds_b) -> dict(stat, p_value, b, c)
  - q_statistic / disagreement_rate / double_fault_rate (Kuncheva
    ensemble diversity between two detectors' correctness patterns)
  - spearman(x, y) rank correlation (tie-safe, average ranks)
"""
from __future__ import annotations

import math
import random
from typing import Callable, Sequence


# ---------------------------------------------------------------------------
# Confusion-matrix metrics
# ---------------------------------------------------------------------------

def confusion(labels: Sequence, preds: Sequence) -> dict:
    """Confusion counts from parallel label/pred arrays.

    ``preds`` is a boolean "flagged" decision. Returns tp/fp/tn/fn.
    """
    if len(labels) != len(preds):
        raise ValueError(f"label/pred length mismatch: {len(labels)} vs {len(preds)}")
    tp = fp = tn = fn = 0
    for lab, pr in zip(labels, preds):
        flag = bool(pr)
        if lab:                      # malicious
            if flag: tp += 1
            else:    fn += 1
        else:                        # benign
            if flag: fp += 1
            else:    tn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "n_mal": tp + fn, "n_ben": fp + tn, "n": tp + fp + tn + fn}


def _tpr(c: dict) -> float:
    return c["tp"] / c["n_mal"] if c["n_mal"] else 0.0


def _fpr(c: dict) -> float:
    return c["fp"] / c["n_ben"] if c["n_ben"] else 0.0


def _precision(c: dict) -> float:
    return c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0


def _f1(c: dict) -> float:
    p, r = _precision(c), _tpr(c)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def metrics_from_counts(labels: Sequence, preds: Sequence) -> dict:
    """Full metric dict for a (labels, preds) pair."""
    c = confusion(labels, preds)
    return {**c, "tpr": _tpr(c), "fpr": _fpr(c),
            "precision": _precision(c), "f1": _f1(c)}


def apply_threshold(scores: Sequence, threshold: float) -> list:
    """Boolean preds: flag where score >= threshold.

    Convention: the threshold IS a score value, and we flag every item
    at or above it. This matches ``threshold_at_fpr`` (which returns a
    score value) so the two compose without an off-by-one.
    """
    return [s >= threshold for s in scores]


# ---------------------------------------------------------------------------
# Score-based: ROC / AUC / operating points
# ---------------------------------------------------------------------------

def roc(labels: Sequence, scores: Sequence) -> list:
    """ROC points sorted by decreasing threshold.

    Returns list of (fpr, tpr, threshold). The first point is (0,0) at
    threshold=+inf (flag nothing); the last is (1,1) at threshold below
    the min score (flag everything). Ties are handled by stepping the
    threshold to each distinct score value (groups of equal scores move
    together).
    """
    if len(labels) != len(scores):
        raise ValueError("label/score length mismatch")
    n_mal = sum(1 for l in labels if l)
    n_ben = len(labels) - n_mal
    if n_mal == 0 or n_ben == 0:
        raise ValueError("ROC undefined when one class is absent")

    # Sort by score descending; break ties deterministically (label order
    # does not matter because equal-score items move as a group below).
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    pts = [(0.0, 0.0)]  # flag nothing
    tp = fp = 0
    i = 0
    while i < len(order):
        # advance over the whole tie-group at this score
        s = scores[order[i]]
        while i < len(order) and scores[order[i]] == s:
            if labels[order[i]]:
                tp += 1
            else:
                fp += 1
            i += 1
        pts.append((fp / n_ben, tp / n_mal))
    # ensure terminal (1,1)
    if pts[-1] != (1.0, 1.0):
        pts.append((1.0, 1.0))
    return pts


def auc(labels: Sequence, scores: Sequence) -> float:
    """Area under the ROC curve via the rank / Mann-Whitney statistic.

    Tie-safe: a malicious/benign pair with equal scores contributes 0.5.
    Equivalent to trapezoidal AUC of ``roc()`` but O(n_mal * n_ben) and
    exact under ties.
    """
    mal_scores = [scores[i] for i in range(len(labels)) if labels[i]]
    ben_scores = [scores[i] for i in range(len(labels)) if not labels[i]]
    if not mal_scores or not ben_scores:
        raise ValueError("AUC undefined when one class is absent")
    total = 0.0
    for ms in mal_scores:
        for bs in ben_scores:
            if ms > bs: total += 1.0
            elif ms == bs: total += 0.5
    return total / (len(mal_scores) * len(ben_scores))


def threshold_at_fpr(labels: Sequence, scores: Sequence, target_fpr: float) -> float:
    """Highest threshold whose FPR is <= target_fpr (most conservative
    operating point that stays within the FPR budget)."""
    n_ben = sum(1 for l in labels if not l)
    if n_ben == 0:
        raise ValueError("no benign examples to set FPR")
    # candidate thresholds: distinct score values, plus +inf.
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    fp = 0
    i = 0
    # threshold just above the max score => flag nothing, fpr 0.
    best_threshold = math.inf
    while i < len(order):
        s = scores[order[i]]
        # count benign at score >= s (i.e., flagged under threshold=s)
        group_ben = 0
        while i < len(order) and scores[order[i]] == s:
            if not labels[order[i]]:
                group_ben += 1
            i += 1
        fp += group_ben
        fpr = fp / n_ben
        if fpr <= target_fpr:
            best_threshold = s   # lowest score we still flag at; keep tightening
        else:
            break
    return best_threshold


def tpr_at_fpr(labels: Sequence, scores: Sequence, target_fpr: float) -> float:
    """TPR at the operating threshold whose FPR is <= target_fpr."""
    th = threshold_at_fpr(labels, scores, target_fpr)
    if math.isinf(th):   # flag nothing
        return 0.0
    preds = apply_threshold(scores, th)
    return _tpr(confusion(labels, preds))


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(labels: Sequence, secondary: Sequence,
                 stat_fn: Callable, n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 0) -> tuple:
    """Percentile bootstrap CI for ``stat_fn(labels, secondary)``.

    ``stat_fn(labels, secondary) -> float``. Resamples (label_i,
    secondary_i) index pairs with replacement. Returns
    ``(point_estimate, lower, upper)`` at the (1-alpha) level.
    """
    if len(labels) != len(secondary):
        raise ValueError("length mismatch")
    n = len(labels)
    rng = random.Random(seed)
    point = stat_fn(labels, secondary)
    if n == 0:
        return (point, float("nan"), float("nan"))
    boots = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        lb = [labels[i] for i in idx]
        sb = [secondary[i] for i in idx]
        boots.append(stat_fn(lb, sb))
    boots.sort()
    lo = boots[int(math.floor((alpha / 2) * n_boot))]
    hi = boots[int(math.ceil((1 - alpha / 2) * n_boot)) - 1]
    return (point, lo, hi)


def bootstrap_paired_diff(labels: Sequence, preds_a: Sequence, preds_b: Sequence,
                          stat_fn: Callable, n_boot: int = 2000,
                          alpha: float = 0.05, seed: int = 0) -> tuple:
    """Bootstrap CI on stat_fn(B) - stat_fn(A) on paired predictions.

    A and B are two systems evaluated on the SAME trajectories (same
    ordering). Resamples trajectory indices, keeping A/B paired.
    Returns (diff_point, lower, upper).
    """
    n = len(labels)
    if not (len(preds_a) == len(preds_b) == n):
        raise ValueError("paired arrays must be equal length")
    rng = random.Random(seed)
    point = stat_fn(labels, preds_b) - stat_fn(labels, preds_a)
    boots = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        lb = [labels[i] for i in idx]
        ab = [preds_a[i] for i in idx]
        bb = [preds_b[i] for i in idx]
        boots.append(stat_fn(lb, bb) - stat_fn(lb, ab))
    boots.sort()
    lo = boots[int(math.floor((alpha / 2) * n_boot))]
    hi = boots[int(math.ceil((1 - alpha / 2) * n_boot)) - 1]
    return (point, lo, hi)


# ---------------------------------------------------------------------------
# McNemar (paired, exact binomial)
# ---------------------------------------------------------------------------

def _binom_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Two-sided exact binomial p-value for k successes out of n."""
    if n == 0:
        return 1.0
    # pmf
    def log_binom(nn, kk):
        return (math.lgamma(nn + 1) - math.lgamma(kk + 1) - math.lgamma(nn - kk + 1))
    def pmf(kk):
        return math.exp(log_binom(n, kk) + kk * math.log(p) + (n - kk) * math.log(1 - p))
    small = pmf(k)
    # two-sided: sum probabilities <= the observed pmf
    total = sum(pmf(kk) for kk in range(n + 1) if pmf(kk) <= small + 1e-15)
    return min(1.0, total)


def mcnemar(labels: Sequence, preds_a: Sequence, preds_b: Sequence) -> dict:
    """Exact McNemar test on paired detections.

    b = # trajectories B catches that A misses.
    c = # trajectories A catches that B misses.
    Under the null (no difference), b and c are Binomial(b+c, 0.5).
    Returns the two-sided exact p-value plus b, c, and the continuity-
    corrected chi-square statistic for reference.
    """
    b = c = 0
    for lab, pa, pb in zip(labels, preds_a, preds_b):
        if pb and not pa: b += 1     # B-only catch
        elif pa and not pb: c += 1   # A-only catch
    n = b + c
    if n == 0:
        return {"stat_chi2": 0.0, "p_value": 1.0, "b": b, "c": c, "n_disc": 0}
    p = _binom_two_sided(min(b, c), n, 0.5)
    chi2 = (abs(b - c) - 1) ** 2 / n if n else 0.0   # continuity-corrected
    return {"stat_chi2": chi2, "p_value": p, "b": b, "c": c, "n_disc": n}


# ---------------------------------------------------------------------------
# Convenience stat_fns
# ---------------------------------------------------------------------------

stat_tpr = lambda labels, preds: _tpr(confusion(labels, preds))
stat_fpr = lambda labels, preds: _fpr(confusion(labels, preds))
stat_f1  = lambda labels, preds: _f1(confusion(labels, preds))
stat_precision = lambda labels, preds: _precision(confusion(labels, preds))


# ---------------------------------------------------------------------------
# Ensemble-diversity measures (Kuncheva & Whitaker, 2003) and rank
# correlation — for quantifying complementarity between detector pairs.
# Correctness convention: a detector is "correct" on a trajectory when
# its flag decision matches the label (flag malicious, pass benign).
# ---------------------------------------------------------------------------

def _pair_counts(labels: Sequence, preds_a: Sequence, preds_b: Sequence) -> dict:
    """N11/N10/N01/N00 correctness-agreement counts for two detectors.

    N11 = both correct, N10 = only A correct, N01 = only B correct,
    N00 = both wrong.
    """
    if not (len(labels) == len(preds_a) == len(preds_b)):
        raise ValueError("array length mismatch")
    n11 = n10 = n01 = n00 = 0
    for lab, pa, pb in zip(labels, preds_a, preds_b):
        ca, cb = bool(pa) == bool(lab), bool(pb) == bool(lab)
        if ca and cb:    n11 += 1
        elif ca:         n10 += 1
        elif cb:         n01 += 1
        else:            n00 += 1
    return {"n11": n11, "n10": n10, "n01": n01, "n00": n00, "n": n11 + n10 + n01 + n00}


def q_statistic(labels: Sequence, preds_a: Sequence, preds_b: Sequence):
    """Yule's Q over pairwise correctness (Kuncheva's diversity measure).

    Q = (N11*N00 - N01*N10) / (N11*N00 + N01*N10), in [-1, 1].
    Q = 0 under independent errors; Q > 0 means correlated errors (less
    diversity). Returns None when the denominator is 0 (degenerate pair,
    e.g. both detectors perfect or both always wrong in agreement).
    """
    c = _pair_counts(labels, preds_a, preds_b)
    num = c["n11"] * c["n00"] - c["n01"] * c["n10"]
    den = c["n11"] * c["n00"] + c["n01"] * c["n10"]
    if den == 0:
        return None
    return num / den


def disagreement_rate(preds_a: Sequence, preds_b: Sequence) -> float:
    """Fraction of trajectories on which the two detectors' *correctness*
    disagrees (Kuncheva's disagreement measure): (N01 + N10) / N.
    """
    c = _pair_counts([False] * len(preds_a), preds_a, preds_b)
    return (c["n01"] + c["n10"]) / c["n"] if c["n"] else 0.0


def double_fault_rate(labels: Sequence, preds_a: Sequence, preds_b: Sequence) -> float:
    """Fraction of trajectories on which BOTH detectors are wrong: N00 / N."""
    c = _pair_counts(labels, preds_a, preds_b)
    return c["n00"] / c["n"] if c["n"] else 0.0


def _avg_ranks(v: Sequence) -> list:
    """Average ranks (1-based, ties share the mean rank), ascending order."""
    order = sorted(range(len(v)), key=lambda i: v[i])
    ranks = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x: Sequence, y: Sequence) -> float:
    """Spearman rank correlation (ties handled via average ranks).

    For constant inputs (no rank variance) returns 0.0 rather than NaN,
    with the degeneracy visible in the caller's own range checks.
    """
    if len(x) != len(y):
        raise ValueError("length mismatch")
    rx, ry = _avg_ranks(x), _avg_ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    if den == 0:
        return 0.0
    return num / den


# ---------------------------------------------------------------------------
# Multiple-comparison correction
# ---------------------------------------------------------------------------

def holm_bonferroni(p_values: Sequence) -> list:
    """Holm–Bonferroni adjusted p-values (family-wise error control).

    Step-down procedure: sort ascending; the i-th smallest is compared
    to alpha/(m-i) at level alpha=0.05, with the running max enforcing
    monotonicity. Returns adjusted p-values in the ORIGINAL order.
    Inputs may contain None (test not computable) — they pass through
    as None and are excluded from the family size.
    """
    m = sum(1 for p in p_values if p is not None)
    if m == 0:
        return list(p_values)
    present = [(p, i) for i, p in enumerate(p_values) if p is not None]
    present.sort(key=lambda t: t[0])
    adjusted = [None] * len(p_values)
    running = 0.0
    for rank, (p, i) in enumerate(present):
        adj = min(1.0, (m - rank) * p)
        running = max(running, adj)
        adjusted[i] = running
    return adjusted


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Perfect separation -> AUC 1.0
    labels = [True, True, True, False, False, False]
    scores = [0.9, 0.8, 0.7, 0.2, 0.1, 0.05]
    print("AUC (should be 1.0):", auc(labels, scores))
    print("TPR@10% FPR (should be 1.0):", tpr_at_fpr(labels, scores, 0.10))
    print("TPR@0% FPR (should be ~0.67, 2/3 mal caught at fpr0):",
          tpr_at_fpr(labels, scores, 0.0))

    # AUC 0.5 random-ish
    rng = random.Random(1)
    lab2 = [rng.random() < 0.5 for _ in range(200)]
    sc2 = [rng.random() + (0.1 if l else 0.0) for l in lab2]
    print("AUC (weak, ~0.56):", round(auc(lab2, sc2), 3))

    # Bootstrap CI on TPR
    preds = apply_threshold(scores, 0.5)
    pt, lo, hi = bootstrap_ci(labels, preds, stat_tpr, n_boot=2000, seed=0)
    print(f"TPR = {pt:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

    # McNemar: A and B differ on 2 cases (b=2, c=0) -> not significant
    preds_a = [True, True, False, False, False, False]
    preds_b = [True, True, True, False, False, False]   # one extra catch
    print("McNemar (b=1,c=0):", mcnemar(labels, preds_a, preds_b))
    # bigger discordance
    preds_b2 = [True, True, True, True, False, False]   # two extra catches, one wrong
    print("McNemar (b=2,c=1):", mcnemar(labels, preds_a, preds_b2))
