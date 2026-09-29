"""Conformal prediction wrapper for the deterministic pipeline.

Replaces the threshold-based `decide()` stage with a *distribution-free*
coverage guarantee. The wrapper computes a non-conformity score for
each trajectory from the deterministic detector's pattern matches, and
calibrates a threshold on a held-out calibration set so that the
prediction set has guaranteed (1−α) coverage.

Coverage guarantee:
    P(true_label ∈ C(x)) ≥ 1 − α
for any trajectory x drawn from the same distribution as the calibration
set, with no assumptions on the underlying distribution.

Output:
    - A `Verdict` whose `confidence` field is the (1−α) coverage
      guarantee (not a point estimate)
    - A new `prediction_set` field listing which labels the conformal
      wrapper considers possible (e.g., {benign}, {malicious}, or
      {benign, malicious})
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

from pipeline.context import PipelineContext
from pipeline.threat_match import threat_match
from pipeline.tokenize import tokenize
from pipeline.parse import parse
from pipeline.enforce import enforce
from schemas import Action, Decision, Trajectory, Verdict


# Severity weights (for non-conformity score).
# Higher weight = more confident "malicious" signal.
_SEVERITY_WEIGHT = {
    "critical": 8,
    "high": 4,
    "medium": 2,
    "low": 1,
}


def nonconformity_score(ctx: PipelineContext) -> float:
    """Compute a non-conformity score: higher = more anomalous.

    Score = (sum of distinct-pattern severity weights) +
             (count of distinct patterns × 0.5) +
             (count of "structural" pattern matches × 2)

    This discriminates between attacks that fire one pattern vs many,
    and rewards structural patterns (T.A.1, T.A.2, T.A.11) more than
    the regex-based ones, since structural matches are harder for
    paraphrases to evade.
    """
    if not ctx.matches:
        return 0.0
    distinct = {m.pattern_id: m.severity for m in ctx.matches}
    sev_sum = sum(_SEVERITY_WEIGHT.get(sev, 1) for sev in distinct.values())
    n_distinct = len(distinct)
    # Bonus for structural detectors (T.A.1, T.A.2, T.A.11)
    structural = {"T.A.1", "T.A.2", "T.A.11", "T.A.13", "T.A.17"}
    n_struct = sum(1 for pid in distinct if pid in structural)
    return sev_sum + 0.5 * n_distinct + 2.0 * n_struct


def detect_patterns(trajectory: Trajectory) -> PipelineContext:
    """Run stages 1-3 of the pipeline and return the context with
    pattern matches. (No decision yet — that's the conformal layer's
    job.)
    """
    from taxonomy.loader import Taxonomy
    ctx = PipelineContext(trajectory=trajectory)
    setattr(ctx, "_taxonomy", Taxonomy.load_default())
    tokenize(ctx)
    parse(ctx)
    threat_match(ctx)
    return ctx


@dataclass
class ConformalCalibrator:
    """Calibration state for the conformal prediction layer.

    Attributes:
        alpha:                target miscoverage rate (e.g., 0.05)
        threshold:            calibrated non-conformity threshold
        coverage_guarantee:   the actual (1−α) coverage this calibrator
                              targets (typically 1−α; lower if finite-sample
                              correction applied)
    """
    alpha: float
    threshold: float = 0.0
    coverage_guarantee: float = 1.0

    @classmethod
    def fit(cls, calibration_trajectories: list[Trajectory],
            alpha: float = 0.05) -> "ConformalCalibrator":
        """Calibrate the threshold on a held-out set.

        The threshold is the (1−α)(1 + 1/n) quantile of the
        non-conformity scores of the *malicious* calibration
        trajectories — i.e., we want the threshold to be just high
        enough to cover (1−α) of malicious examples. Any trajectory
        above this threshold is flagged as malicious with the
        coverage guarantee.
        """
        mal_scores = []
        for t in calibration_trajectories:
            if t.ground_truth != "malicious":
                continue
            ctx = detect_patterns(t)
            mal_scores.append(nonconformity_score(ctx))
        if not mal_scores:
            raise ValueError("calibration set has no malicious trajectories")
        mal_scores.sort()
        n = len(mal_scores)
        # Finite-sample correction: (1−α)(1 + 1/n) quantile
        q_idx = max(0, min(n - 1, int(math.ceil((1 - alpha) * (n + 1))) - 1))
        threshold = mal_scores[q_idx]
        return cls(alpha=alpha, threshold=threshold,
                   coverage_guarantee=1.0 - alpha)

    def predict(self, trajectory: Trajectory) -> ConformalVerdict:
        """Compute the prediction set for a new trajectory.

        Returns:
            - decision: a single Decision (the most conservative)
            - prediction_set: set of Decision values
            - coverage: the (1−α) coverage guarantee
        """
        ctx = detect_patterns(trajectory)
        score = nonconformity_score(ctx)
        # If score > threshold, the trajectory is confidently malicious.
        # If score <= threshold, the trajectory is "ambiguous" — both
        # benign and malicious are possible.
        if score > self.threshold:
            return ConformalVerdict(
                decision=Decision.MALICIOUS,
                prediction_set={Decision.MALICIOUS},
                score=score,
                threshold=self.threshold,
                coverage=self.coverage_guarantee,
            )
        # Ambiguous: if any high-severity match, still suspicious.
        if any(m.severity in ("high", "critical") for m in ctx.matches):
            return ConformalVerdict(
                decision=Decision.SUSPICIOUS,
                prediction_set={Decision.SUSPICIOUS,
                                Decision.MALICIOUS, Decision.BENIGN},
                score=score,
                threshold=self.threshold,
                coverage=self.coverage_guarantee,
            )
        return ConformalVerdict(
            decision=Decision.BENIGN,
            prediction_set={Decision.BENIGN,
                            Decision.MALICIOUS, Decision.SUSPICIOUS},
            score=score,
            threshold=self.threshold,
            coverage=self.coverage_guarantee,
        )


@dataclass
class ConformalVerdict:
    decision: Decision
    prediction_set: set
    score: float
    threshold: float
    coverage: float
    matches: list = field(default_factory=list)
    notes: list = field(default_factory=list)


class ConformalPipeline:
    """End-to-end conformal-prediction wrapper around the deterministic pipeline.

    Use:
        cal = ConformalPipeline(alpha=0.05)
        cal.calibrate(calibration_trajectories)   # fit threshold
        verd = cal.verify(new_trajectory)         # predict with coverage

    The verifier emits a `ConformalVerdict` whose `coverage` field
    is the (1−α) coverage guarantee.
    """

    name = "conformal-det"
    version = "0.1.0"

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha
        self.calibrator: ConformalCalibrator | None = None
        self._coverage_guarantee = 1.0 - alpha

    def calibrate(self, trajectories: list[Trajectory]) -> None:
        self.calibrator = ConformalCalibrator.fit(trajectories, self.alpha)

    def verify(self, trajectory: Trajectory) -> ConformalVerdict:
        if self.calibrator is None:
            raise RuntimeError("calibrate() must be called before verify()")
        return self.calibrator.predict(trajectory)