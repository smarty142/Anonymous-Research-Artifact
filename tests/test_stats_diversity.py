"""Tests for ensemble-diversity statistics and Spearman correlation
added to evaluation/stats.py (Kuncheva & Whitaker measures)."""
import math

from evaluation import stats


# --- q_statistic -----------------------------------------------------------

def test_q_perfect_agreement_is_positive_one():
    # Both detectors correct on everything they agree on and wrong together
    # on the rest: perfectly correlated errors -> Q = 1.
    labels = [True, True, False, False]
    a = [True, False, True, False]     # wrong on 1, 3
    b = [True, False, True, False]     # identical
    q = stats.q_statistic(labels, a, b)
    assert q == 1.0


def test_q_anti_correlated_errors_is_negative_one():
    # Each detector is wrong exactly where the other is right.
    labels = [True, True, False, False]
    a = [True, True, True, True]       # wrong on both benign
    b = [False, False, False, False]   # wrong on both malicious
    q = stats.q_statistic(labels, a, b)
    assert q == -1.0


def test_q_degenerate_returns_none():
    # Both detectors perfect: N11*N00 + N01*N10 = 0 -> undefined.
    labels = [True, True, False, False]
    a = [True, True, False, False]
    assert stats.q_statistic(labels, a, a) is None


def test_q_hand_computed():
    # N11=2 (both correct on 0,3), N10=1 (only A on 2), N01=1 (only B on 1),
    # N00=1 (both wrong flagging benign 4)
    # Q = (2*1 - 1*1) / (2*1 + 1*1) = 1/3
    labels = [True, True, True, True, False]
    a = [True, False, True, True, True]    # wrong on 1 and benign 4
    b = [True, True, False, True, True]    # wrong on 2 and benign 4
    q = stats.q_statistic(labels, a, b)
    assert math.isclose(q, 1.0 / 3.0)


# --- disagreement / double fault -------------------------------------------

def test_disagreement_rate():
    labels = [True, True, True, False]
    a = [True, False, True, False]     # correct 0,2,3
    b = [True, True, False, False]     # correct 0,1,3
    # correctness disagrees on 1 vs 2 -> 2/4
    assert math.isclose(stats.disagreement_rate(a, b), 0.5)


def test_disagreement_rate_identical():
    a = [True, False, True]
    assert stats.disagreement_rate(a, list(a)) == 0.0


def test_double_fault_rate():
    labels = [True, True, True, False]
    a = [True, False, True, False]     # wrong only on 1
    b = [True, True, False, False]     # wrong only on 2
    # both wrong nowhere -> 0.0
    assert stats.double_fault_rate(labels, a, b) == 0.0
    # make both wrong on trajectory 1
    b2 = [True, False, True, True]     # wrong on 1 (and benign 3)
    # a wrong on 1; b2 wrong on 1,3 -> both-wrong = {1} -> 1/4
    assert math.isclose(stats.double_fault_rate(labels, a, b2), 0.25)


# --- spearman ---------------------------------------------------------------

def test_spearman_monotonic_is_one():
    x = [0.1, 0.4, 0.9, 1.5]
    y = [2.0, 2.1, 2.7, 3.0]
    assert math.isclose(stats.spearman(x, y), 1.0)


def test_spearmon_reverse_is_negative_one():
    x = [0.1, 0.4, 0.9, 1.5]
    y = [3.0, 2.7, 2.1, 2.0]
    assert math.isclose(stats.spearman(x, y), -1.0)


def test_spearman_ties_average_ranks():
    x = [0.1, 0.1, 0.5, 0.9]
    y = [1.0, 1.0, 2.0, 3.0]
    # ranks x = [1.5, 1.5, 3, 4], y = [1.5, 1.5, 3, 4] -> exactly 1.0
    assert math.isclose(stats.spearman(x, y), 1.0)


def test_spearman_constant_returns_zero():
    assert stats.spearman([0.5, 0.5, 0.5], [0.1, 0.4, 0.9]) == 0.0


# --- holm-bonferroni --------------------------------------------------------

def test_holm_basic():
    # Family of 3: raw .01, .04, .03
    # sorted: .01 -> 3*.01=.03; .03 -> 2*.03=.06; .04 -> 1*.04=.04 -> max-run .06
    adj = stats.holm_bonferroni([0.04, 0.01, 0.03])
    assert math.isclose(adj[1], 0.03)
    assert math.isclose(adj[2], 0.06)
    assert math.isclose(adj[0], 0.06)   # monotonicity: .04 lifted to .06


def test_holm_caps_at_one_and_passes_none():
    adj = stats.holm_bonferroni([0.9, None, 0.5])
    assert adj[1] is None
    assert adj[0] == 1.0 and adj[2] == 1.0
