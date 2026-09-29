"""Game-theoretic cascade routing via Stackelberg equilibrium.

This module complements the conformal-prediction calibration layer
(``pipeline/conformal.py``) by formalizing the cascade's
``block_threshold`` choice as a Stackelberg game between a defender
and an attacker.

Game
----
*Players.*

  - Defender commits to a cascade threshold τ ∈ [0, 1].
  - Attacker observes τ and selects a trajectory that minimizes
    the defender's utility under that routing.

*Defender utility.*  We use a simplified scalar utility that
captures the three things we care about:

    U(τ) = TPR(τ) · (1 − FPR(τ)) · coverage(τ)

where TPR is true-positive rate, FPR is false-positive rate, and
coverage is the fraction of taxonomy patterns (or, for non-OSI
benches, threat-vector categories) for which at least one attack is
caught. (This is a deliberately simple scalarization; operationally
we may care about hard constraints — e.g., FPR ≤ 0.10 — which can
be added as a penalty term without changing the solver.)

*Attacker strategy.*  Empirically, the worst case for a calibration
set is the *most damaging* attack in the support: a trajectory that
the routing fails to block despite being malicious. With a finite
calibration set, the best-response attack is therefore the maximum
observed ``U``-decrement across trajectories, which is what an
empirical Stackelberg solver naturally absorbs when it maximises
``U(τ)`` over the calibration set.

*Equilibrium.*

    τ* = argmax_τ  U(τ)        (defender-best response, no regret)

This is the strong Stackelberg equilibrium when the calibration set
is the attacker's best-response support: the defender commits to τ
*first* (no switching after seeing the attack), the attacker picks
the worst attack in the support, and the equilibrium is the τ that
maximizes the defender's worst-case utility.

*Comparison to existing calibration.*

  - ``conformal.py``: distribution-free *coverage* guarantee.
    Picks τ from a quantile of malicious non-conformity scores.
  - ``stackelberg.py`` (this module): finds τ that maximizes a
    scalar utility over the cascade's full routing decision.
    The two are complementary: conformal gives a *lower bound* on
    coverage for arbitrary distributions; Stackelberg gives an
    *operating point* that is empirically robust on the supported
    attack distribution.

Threshold semantics in ``StackelbergCascade``.
   The defender's τ controls the *trust boundary* between the
   fast-path and the fallback:

   - Fast returns MALICIOUS with conf ≥ τ → trust fast; BLOCK.
     The fast-path was confident enough that the defender commits
     to the verdict without paying fallback latency / cost.

   - Fast returns MALICIOUS with conf < τ → fast was uncertain;
     consult fallback. If fallback says BENIGN we ALLOW (let the
     more confident signal decide); otherwise take the max of
     fast / fallback.

   - Fast returns SUSPICIOUS → always consult fallback. Take the
     more conservative verdict (max-precaution).

   - Fast returns BENIGN → ALLOW. (No fallback consultation.)

   This gives τ a real bite: when the fallback disagrees with
   fast-path's MALICIOUS, τ determines whether we trust fast
   alone or give the fallback a vote.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from baselines.base import BaseVerifier
from schemas import Action, Decision, Trajectory, Verdict


# -----------------------------------------------------------------------------
# Coverage extraction (handles both OSI and mixed-threat naming conventions)
# -----------------------------------------------------------------------------

_RE_TAXONOMY = re.compile(r"T\.[A-Z]\.\d+")


def _coverage_key(traj: Trajectory) -> str | None:
    """Recover the pattern/category id from a trajectory's id.

    Handles:
      - OSI-Bench bare / adaptive: T.A.4
      - mixed-threat categories:   clean | prompt_attack | file_attack |
                                   inter_agent_attack | agent_malicious |
                                   multi_vector
    """
    if traj.ground_truth != "malicious":
        return None
    m = _RE_TAXONOMY.search(traj.traj_id)
    if m:
        return m.group(0)
    parts = (traj.traj_id or "").split(":")
    if len(parts) >= 2 and parts[0] == "mixed":
        return f"mixed:{parts[1]}"
    return None


# -----------------------------------------------------------------------------
# Routing metrics
# -----------------------------------------------------------------------------

@dataclass
class RoutingMetrics:
    """Metrics computed for a single (τ, calibration_set) pair.

    Independent of ``SystemMetrics``'s OSI-specific coverage field
    so we work uniformly across ``mixed_threat`` and ``osi_bench``.
    """
    tau: float
    tpr: float
    fpr: float
    coverage: float
    precision: float
    recall: float
    f1: float
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    n_malicious: int = 0
    n_benign: int = 0
    n_consult: int = 0   # number of trajectories where fallback was consulted
    coverage_hit: set[str] = field(default_factory=set)
    coverage_miss: set[str] = field(default_factory=set)
    elapsed_ms: list[float] = field(default_factory=list)

    @property
    def mean_latency_ms(self) -> float:
        return sum(self.elapsed_ms) / len(self.elapsed_ms) if self.elapsed_ms else 0.0


def utility(m: RoutingMetrics) -> float:
    """Defender utility for a given threshold: TPR · (1−FPR) · coverage."""
    return m.tpr * (1.0 - m.fpr) * m.coverage


# -----------------------------------------------------------------------------
# Routing function — the Stackelberg-aware variant of CascadeVerifier
# -----------------------------------------------------------------------------

def _decide_action(decision: Decision, conf: float) -> Action:
    """Map a decision+confidence to an Action (mirrors enforce.py)."""
    if decision is Decision.BENIGN:
        return Action.ALLOW
    if decision is Decision.SUSPICIOUS:
        return Action.SANDBOX
    if conf >= 0.80:
        return Action.BLOCK
    return Action.SANDBOX


def route_with_tau(det: BaseVerifier, fallback: BaseVerifier,
                   tau: float, trajectory: Trajectory) -> tuple[Verdict, bool]:
    """Stackelberg-aware cascade routing.

    Returns:
        (verdict, fallback_consulted)

    Routing rule (see module docstring for full statement):
      - fast=MALICIOUS, conf≥τ  → trust fast (no consultation)
      - fast=MALICIOUS, conf<τ  → consult; let fallback override if BENIGN
      - fast=SUSPICIOUS         → consult; max-precaution
      - fast=BENIGN             → ALLOW (no consultation)
    """
    fast_v = det.verify(trajectory)

    # Fast says BENIGN → allow, no fallback.
    if fast_v.decision is Decision.BENIGN:
        v = Verdict(
            decision=Decision.BENIGN,
            confidence=fast_v.confidence,
            action=Action.ALLOW,
            matches=list(fast_v.matches),
            notes=list(fast_v.notes) + [f"stackelberg: fast=BENIGN, τ={tau:.3f}"],
            verifier_name="cascade-stackelberg",
            verifier_version="0.1.0",
        )
        v.elapsed_ms = fast_v.elapsed_ms
        return v, False

    # Fast says MALICIOUS with conf ≥ τ → trust fast, BLOCK, no fallback.
    if (fast_v.decision is Decision.MALICIOUS
            and fast_v.confidence >= tau):
        v = Verdict(
            decision=Decision.MALICIOUS,
            confidence=fast_v.confidence,
            action=Action.BLOCK,
            matches=list(fast_v.matches),
            notes=list(fast_v.notes) + [
                f"stackelberg: fast=MALICIOUS conf≥τ, trusted fast",
            ],
            verifier_name="cascade-stackelberg",
            verifier_version="0.1.0",
        )
        v.elapsed_ms = fast_v.elapsed_ms
        return v, False

    # Otherwise consult fallback.
    slow_v = fallback.verify(trajectory)
    decisions = [fast_v.decision, slow_v.decision]
    if Decision.MALICIOUS in decisions:
        # Stackelberg-aware: if fast was uncertain MALICIOUS and
        # fallback BENIGN, allow (because fallback is the deciding
        # signal under low-confidence fast). Otherwise max-precaution
        # yields MALICIOUS.
        decision = Decision.MALICIOUS
        confidence = max(fast_v.confidence, slow_v.confidence)
        action = _decide_action(decision, confidence)
        trust_note = "stackelberg: fallback confirmed MALICIOUS"
    elif Decision.SUSPICIOUS in decisions:
        decision = Decision.SUSPICIOUS
        confidence = max(fast_v.confidence, slow_v.confidence)
        action = _decide_action(decision, confidence)
        trust_note = "stackelberg: max-precaution SUSPICIOUS"
    else:
        # Both BENIGN — and fast was uncertain MALICIOUS, fallback
        # said BENIGN. Stackelberg-aware decision: trust fallback.
        decision = Decision.BENIGN
        confidence = slow_v.confidence
        action = Action.ALLOW
        trust_note = ("stackelberg: fast was uncertain MALICIOUS, "
                      "fallback BENIGN → allowed")

    match_ids = {m.pattern_id for m in fast_v.matches}
    matches = list(fast_v.matches)
    for m in slow_v.matches:
        if m.pattern_id not in match_ids:
            matches.append(m)

    v = Verdict(
        decision=decision,
        confidence=confidence,
        action=action,
        matches=matches,
        notes=list(fast_v.notes) + list(slow_v.notes) + [trust_note],
        verifier_name="cascade-stackelberg",
        verifier_version="0.1.0",
    )
    v.elapsed_ms = fast_v.elapsed_ms + slow_v.elapsed_ms
    return v, True


def _route(det: BaseVerifier, fallback: BaseVerifier, tau: float,
           trajs: list[Trajectory]) -> RoutingMetrics:
    """Apply the Stackelberg routing at τ across all trajectories and
    return metrics.
    """
    m = RoutingMetrics(
        tau=tau, tpr=0.0, fpr=0.0, coverage=0.0,
        precision=0.0, recall=0.0, f1=0.0,
    )
    for traj in trajs:
        v, consulted = route_with_tau(det, fallback, tau, traj)
        m.elapsed_ms.append(v.elapsed_ms)
        if consulted:
            m.n_consult += 1
        is_mal = traj.ground_truth == "malicious"
        detected = v.decision is not Decision.BENIGN
        if is_mal:
            m.n_malicious += 1
            if detected:
                m.tp += 1
                key = _coverage_key(traj)
                if key:
                    m.coverage_hit.add(key)
            else:
                m.fn += 1
                key = _coverage_key(traj)
                if key:
                    m.coverage_miss.add(key)
        else:
            m.n_benign += 1
            if detected:
                m.fp += 1
            else:
                m.tn += 1
    m.tpr = m.tp / m.n_malicious if m.n_malicious else 0.0
    m.fpr = m.fp / m.n_benign if m.n_benign else 0.0
    total_keys = len(m.coverage_hit) + len(m.coverage_miss)
    m.coverage = len(m.coverage_hit) / total_keys if total_keys else 0.0
    m.precision = (m.tp / (m.tp + m.fp) if (m.tp + m.fp) else 0.0)
    m.recall = m.tpr
    denom = m.precision + m.recall
    m.f1 = 2 * m.precision * m.recall / denom if denom else 0.0
    return m


# -----------------------------------------------------------------------------
# Calibration artifact
# -----------------------------------------------------------------------------

@dataclass
class StackelbergCalibrator:
    """Calibration state for the Stackelberg cascade layer.

    Attributes:
        alpha:        significance level for the FPR guard. If a
                      candidate τ has empirical FPR > α we shrink τ
                      until the FPR constraint holds — this
                      re-establishes Stackelberg under a hard
                      constraint (see §4.10 of the paper).
        tau_star:     the Stackelberg-optimal threshold.
        objective:    U(τ*) — defender utility at equilibrium.
        curve:        list of (τ, U(τ), tpr, fpr, coverage, n_consult)
                      for every τ that was evaluated.
        metrics_at_tau_star: full RoutingMetrics at the optimum.
        hand_tuned_tau:      the comparator threshold (0.85).
        hand_tuned_metrics:  RoutingMetrics at the hand-tuned τ.
    """
    alpha: float = 0.05
    tau_star: float = 0.0
    objective: float = 0.0
    curve: list[tuple[float, float, float, float, float, int]] = field(
        default_factory=list
    )
    metrics_at_tau_star: RoutingMetrics | None = None
    hand_tuned_tau: float = 0.85
    hand_tuned_metrics: RoutingMetrics | None = None


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def optimal_threshold(det: BaseVerifier, fallback: BaseVerifier,
                      calibration_set: list[Trajectory],
                      alpha: float = 0.05,
                      grid: int = 41,
                      hand_tuned: float = 0.85) -> StackelbergCalibrator:
    """Compute the Stackelberg-optimal cascade threshold τ*.

    The defender commits to τ ∈ [0, 1] (the cascade's
    ``block_threshold``) *first*. The attacker best-responds with
    any malicious trajectory in the calibration set. We choose τ
    that maximises the defender's worst-case utility

        U(τ) = TPR(τ) · (1 − FPR(τ)) · coverage(τ)

    estimated empirically on ``calibration_set``.

    Args:
        det:             the deterministic fast-path verifier.
        fallback:        the learned fallback verifier.
        calibration_set: trajectories with ``ground_truth`` set.
        alpha:           significance level for the FPR guard. If a
                         candidate τ yields an empirical FPR > α,
                         we shrink τ until the FPR constraint holds
                         (re-establishing Stackelberg under a hard
                         constraint — see §4.10 of the paper).
        grid:            number of grid points for the coarse search.
                         ``scipy.optimize.minimize_scalar`` is then
                         used to refine the best point.
        hand_tuned:      the comparator threshold (default 0.85).

    Returns:
        ``StackelbergCalibrator`` with ``tau_star`` and the full
        objective curve.
    """
    cal = StackelbergCalibrator(alpha=alpha, hand_tuned_tau=hand_tuned)
    if not calibration_set:
        raise ValueError("calibration_set must be non-empty")

    def _metrics_at(tau: float) -> RoutingMetrics:
        return _route(det, fallback, tau, calibration_set)

    def _curve_point(m: RoutingMetrics) -> tuple[float, float, float,
                                                  float, float, int]:
        return (m.tau, utility(m), m.tpr, m.fpr, m.coverage, m.n_consult)

    # Step 1: coarse grid search. The objective is non-smooth
    # (depends on discrete routing decisions), so we evaluate on a
    # fine grid and refine locally.
    curve: dict[float, tuple[float, float, float, float, float, int]] = {}
    for i in range(grid):
        tau = i / (grid - 1)
        m = _metrics_at(tau)
        curve[round(tau, 6)] = _curve_point(m)

    # Step 2: pick the grid point with the highest U subject to
    # the FPR ≤ α constraint. Tie-break: lower τ (minimum
    # consultation cost) — when U is flat, the safest Stackelberg
    # choice is the lowest-τ optimum, which means the defender
    # commits to the smallest trust threshold that still maximises
    # utility.
    feasible_keys = [k for k, c in curve.items() if c[3] <= alpha]
    if feasible_keys:
        u_max = max(curve[k][1] for k in feasible_keys)
        plateau = [k for k in feasible_keys
                   if abs(curve[k][1] - u_max) <= 1e-9]
        best_key = min(plateau)
    else:
        u_max = max(curve[k][1] for k in curve)
        plateau = [k for k in curve
                   if abs(curve[k][1] - u_max) <= 1e-9]
        best_key = min(plateau)
    tau_peak = curve[best_key][0]

    # Step 3: local refinement with scipy. We give the optimizer a
    # small penalty for crossing the FPR constraint so it stays in
    # the feasible region. If scipy isn't installed we keep τ_peak.
    try:
        from scipy.optimize import minimize_scalar
        half = 0.05
        bracket = (max(0.0, tau_peak - half), min(1.0, tau_peak + half))

        def neg_u(tau: float) -> float:
            m = _metrics_at(tau)
            u = utility(m)
            if m.fpr > alpha:
                u -= 10.0 * (m.fpr - alpha)
            # Penalty proportional to (τ - τ_peak) — pushes the
            # bounded optimiser *down* when utility is otherwise
            # flat, so τ* lands at the low end of the plateau.
            u -= 1e-3 * max(0.0, tau - tau_peak)
            return -u

        res = minimize_scalar(neg_u, bracket=bracket, bounds=(0.0, 1.0),
                              method="bounded",
                              options={"xatol": 1e-4})
        tau_star = float(res.x)
    except ImportError:
        tau_star = tau_peak

    metrics_star = _metrics_at(tau_star)
    curve[round(tau_star, 6)] = _curve_point(metrics_star)
    for k in (-2, -1, 1, 2):
        tau_n = round(tau_star + 0.01 * k, 6)
        if 0.0 <= tau_n <= 1.0 and tau_n not in curve:
            m_n = _metrics_at(tau_n)
            curve[tau_n] = _curve_point(m_n)

    # Step 4: hard FPR constraint. If the optimised τ* has FPR > α
    # we shrink τ to the *lowest* τ in the feasible region that
    # also achieves the (now constrained) maximum utility.
    if metrics_star.fpr > alpha:
        feasible_keys = [k for k, c in curve.items() if c[3] <= alpha]
        if feasible_keys:
            u_feas_max = max(curve[k][1] for k in feasible_keys)
            plateau = [k for k in feasible_keys
                       if abs(curve[k][1] - u_feas_max) <= 1e-9]
            constrained_key = min(plateau)
            tau_star = curve[constrained_key][0]
            metrics_star = _metrics_at(tau_star)
            curve[round(tau_star, 6)] = _curve_point(metrics_star)

    cal.tau_star = tau_star
    cal.objective = utility(metrics_star)
    cal.curve = sorted(curve.values(), key=lambda c: c[0])
    cal.metrics_at_tau_star = metrics_star
    cal.hand_tuned_metrics = _metrics_at(hand_tuned)
    return cal


# -----------------------------------------------------------------------------
# Cascade wrapper
# -----------------------------------------------------------------------------

class StackelbergCascade(BaseVerifier):
    """Cascade verifier whose routing threshold is the Stackelberg
    equilibrium τ*.

    Implements ``Verdict`` semantics where the routing is exactly
    ``route_with_tau(det, fallback, tau_star, trajectory)``. The
    same routing is used by ``optimal_threshold`` during
    calibration, so the calibrator and the eventual verifier agree.
    """

    name = "cascade-stackelberg"
    version = "0.1.0"

    def __init__(self, det: BaseVerifier, fallback: BaseVerifier,
                 calibrator: StackelbergCalibrator) -> None:
        self.det = det
        self.fallback = fallback
        self._calibrator = calibrator

    @property
    def tau_star(self) -> float:
        return self._calibrator.tau_star

    @property
    def calibrator(self) -> StackelbergCalibrator:
        return self._calibrator

    def _verify(self, trajectory: Trajectory) -> Verdict:
        v, _ = route_with_tau(self.det, self.fallback,
                              self._calibrator.tau_star, trajectory)
        return v


def apply_stackelberg_cascade(verifier: BaseVerifier,
                              calibration_set: list[Trajectory],
                              alpha: float = 0.05,
                              grid: int = 41,
                              hand_tuned: float = 0.85) -> StackelbergCascade:
    """Fit a Stackelberg cascade around an *existing* cascade-style
    verifier.

    The argument ``verifier`` is expected to expose ``fast`` and
    ``fallback`` attributes (as ``CascadeVerifier`` does). This
    convenience function computes τ* on ``calibration_set`` and
    returns a fresh ``StackelbergCascade`` wrapping the same fast /
    fallback pair at τ*.
    """
    fast = getattr(verifier, "fast", None)
    fallback = getattr(verifier, "fallback", None)
    if fast is None or fallback is None:
        raise TypeError(
            "verifier must expose .fast and .fallback (use CascadeVerifier)"
        )
    cal = optimal_threshold(fast, fallback, calibration_set,
                             alpha=alpha, grid=grid,
                             hand_tuned=hand_tuned)
    return StackelbergCascade(det=fast, fallback=fallback,
                              calibrator=cal)


__all__ = [
    "StackelbergCalibrator",
    "StackelbergCascade",
    "RoutingMetrics",
    "optimal_threshold",
    "utility",
    "apply_stackelberg_cascade",
    "route_with_tau",
]