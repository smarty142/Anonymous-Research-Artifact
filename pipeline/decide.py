"""Stage 4: Decide.

Combines all ThreatMatch objects on the context into a single
``Decision`` and a ``confidence`` score.

Heuristic:
  * One critical match         -> malicious (high conf)
  * >=2 distinct high/critical -> malicious
  * 1 high or several mediums  -> suspicious
  * otherwise                  -> benign

This is intentionally a *simple* deterministic rule. The paper's claim is
that the upstream matching is the contribution; the decision combiner
just maps evidence to action.
"""
from __future__ import annotations

from pipeline.context import PipelineContext
from schemas import Decision


_SEVERITY_SCORE = {"critical": 4, "high": 3, "medium": 2, "low": 1}


def decide(ctx: PipelineContext) -> tuple[Decision, float]:
    matches = ctx.matches
    if not matches:
        return Decision.BENIGN, 0.95

    # Aggregate per-pattern evidence
    by_pattern: dict[str, list] = {}
    for m in matches:
        by_pattern.setdefault(m.pattern_id, []).append(m)

    # Sum severities, deduplicated per pattern
    score = 0.0
    max_severity_score = 0
    for pid, ms in by_pattern.items():
        # Use the highest severity per pattern; one occurrence per pattern
        max_sev = max(_SEVERITY_SCORE.get(m.severity, 1) for m in ms)
        max_severity_score = max(max_severity_score, max_sev)
        score += max_sev

    # Decision thresholds
    distinct = len(by_pattern)
    has_critical = any(m.severity == "critical" for m in matches)

    if has_critical and distinct >= 1:
        decision = Decision.MALICIOUS
        confidence = 0.85 + min(distinct, 5) * 0.02
    elif distinct >= 2 and max_severity_score >= 3:
        decision = Decision.MALICIOUS
        confidence = 0.75 + min(distinct, 5) * 0.03
    elif max_severity_score >= 3 or distinct >= 3:
        decision = Decision.SUSPICIOUS
        confidence = 0.55 + min(distinct, 5) * 0.04
    elif max_severity_score >= 2 or distinct >= 1:
        decision = Decision.SUSPICIOUS
        confidence = 0.40 + min(distinct, 5) * 0.04
    else:
        decision = Decision.BENIGN
        confidence = 0.80

    # Clamp confidence
    confidence = max(0.0, min(0.99, confidence))
    return decision, confidence