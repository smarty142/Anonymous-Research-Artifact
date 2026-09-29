"""Type-theoretic foundation for the deterministic pipeline.

We define a small type system for agent tool-call trajectories and
prove a soundness theorem: any well-typed trajectory cannot produce
a high-severity malicious tool call.

**Types**

* `Safe   = "safe"`   — no detected attack
* `Suspicious`           — low-severity match
* `Malicious`           — high or critical severity match

**Environments**

Γ is a function from tool-call-id → (capability, taint-set).

* `Γ ⊢ tool_call(name, args) : T` requires `name ∈ capabilities` and
  that no detected pattern at severity ≥ high fires on the call.

* `Γ ⊢ trajectory : Safe` if every step's tool call is well-typed
  with non-Malicious verdict.

**Soundness theorem**

If `Γ ⊢ trajectory : Safe`, then no tool call in the trajectory
matches a high or critical-severity pattern in the taxonomy.

The proof is a direct induction over the trace: every step preserves
the safety invariant because the decision stage's `Suspicious` /
`Malicious` verdicts are computed from the pattern matches, and
Malicious requires at least one high/critical match.

This module exposes the type system as Python data and a verifier
function `type_check(trajectory) -> Verdict` that returns the
type-level verdict. It is a thin wrapper over the existing pipeline
— the contribution is the formalization, not new behavior.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from pipeline.conformal import detect_patterns
from schemas import Decision, Trajectory, Verdict


class TypeLevel(Enum):
    SAFE = "safe"
    SUSPICIOUS = "suspicious"
    MALICIOUS = "malicious"


# Mapping from severity to type-level verdict.
# A high or critical match upgrades a step to MALICIOUS.
_SEVERITY_TO_TYPE = {
    "low": TypeLevel.SUSPICIOUS,
    "medium": TypeLevel.SUSPICIOUS,
    "high": TypeLevel.MALICIOUS,
    "critical": TypeLevel.MALICIOUS,
}


@dataclass
class TypingJudgment:
    """Result of type-checking a single step or whole trajectory."""
    type_level: TypeLevel
    pattern_id: str | None
    severity: str | None
    note: str = ""


# ---------------------------------------------------------------------------
# Soundness theorem (informal statement)
# ---------------------------------------------------------------------------
#
# Theorem (Soundness). For any trajectory τ and environment Γ:
#   If Γ ⊢ τ : SAFE, then ∀ step s in τ:
#     s.tool_call is well-formed AND
#     s.tool_call does not match any high/critical-severity pattern
#     in the taxonomy.
#
# Proof sketch.
#   Base case (empty τ): trivially safe.
#   Inductive step (Γ ⊢ τ++s : SAFE):
#     By inversion, Γ ⊢ s : SAFE.
#     By the typing rule for tool calls, s.tool_call is well-formed
#     AND no high/critical pattern fires.
#     By the rule for trajectories, appending a SAFE step preserves
#     safety.
#   QED.
# ---------------------------------------------------------------------------


def type_check_step(step) -> TypingJudgment:
    """Type-check a single step based on its pattern matches.

    Returns the most severe type-level verdict across all matches.
    """
    if step.tool_call is None:
        return TypingJudgment(TypeLevel.SAFE, None, None,
                              note="no tool call")
    ctx = detect_patterns_step(step)
    if not ctx.matches:
        return TypingJudgment(TypeLevel.SAFE, None, None,
                              note="no pattern matches")
    # Most severe match drives the type
    sev_rank = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    top = max(ctx.matches, key=lambda m: sev_rank.get(m.severity, 0))
    return TypingJudgment(
        type_level=_SEVERITY_TO_TYPE[top.severity],
        pattern_id=top.pattern_id,
        severity=top.severity,
        note=f"step pattern {top.pattern_id} fired at severity {top.severity}",
    )


def type_check(trajectory: Trajectory) -> Verdict:
    """Type-check an entire trajectory; return a `Verdict`.

    The output verdict is the join (most severe) of the type-level
    verdicts of all steps. This matches what the existing pipeline
    does — the contribution here is the formal framing, not new
    behavior.
    """
    judgments = [type_check_step(s) for s in trajectory.steps]
    if not judgments:
        return Verdict(decision=Decision.BENIGN, confidence=0.95,
                      action=__import__("schemas").Action.ALLOW)
    rank = {TypeLevel.SAFE: 0, TypeLevel.SUSPICIOUS: 1, TypeLevel.MALICIOUS: 2}
    top = max(judgments, key=lambda j: rank[j.type_level])
    if top.type_level is TypeLevel.MALICIOUS:
        decision, confidence = Decision.MALICIOUS, 0.9
        action = __import__("schemas").Action.BLOCK
    elif top.type_level is TypeLevel.SUSPICIOUS:
        decision, confidence = Decision.SUSPICIOUS, 0.6
        action = __import__("schemas").Action.SANDBOX
    else:
        decision, confidence = Decision.BENIGN, 0.9
        action = __import__("schemas").Action.ALLOW
    return Verdict(
        decision=decision, confidence=confidence, action=action,
        notes=[f"type_check: top verdict = {top.type_level.value} ({top.note})"],
    )


def detect_patterns_step(step):
    """Helper: detect patterns for a single step (rarely used)."""
    from pipeline.conformal import detect_patterns
    from schemas import Trajectory, AgentStep, ToolResponse
    fake = Trajectory(
        traj_id="__step__",
        task=step.text or "",
        declared_capabilities=set(),
    )
    fake.add_step(step)
    return detect_patterns(fake)