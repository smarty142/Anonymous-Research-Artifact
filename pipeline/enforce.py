"""Stage 5: Enforce.

Map a :class:`Decision` to an :class:`Action`:

  benign              -> allow
  suspicious          -> sandbox (run in restricted mode + audit log)
  malicious (low conf)-> sandbox + require HITL
  malicious (high conf)-> block; HITL for irreversible upcoming actions

The enforce stage also adds an ``enforcement_note`` describing what the
runtime should do (caller implements).
"""
from __future__ import annotations

from pipeline.context import PipelineContext
from schemas import Action, Decision


def enforce(decision: Decision, confidence: float,
            has_irreversible: bool) -> tuple[Action, str]:
    if decision is Decision.BENIGN:
        return Action.ALLOW, "no action required"

    if decision is Decision.SUSPICIOUS:
        return (Action.HITL if has_irreversible else Action.SANDBOX,
                "sandboxed execution; HITL for irreversible steps")

    # malicious
    if confidence >= 0.80 and has_irreversible:
        return Action.BLOCK, "blocked: high-confidence malicious + irreversible tool pending"
    if confidence >= 0.80:
        return Action.BLOCK, "blocked: high-confidence malicious"
    if has_irreversible:
        return Action.HITL, "HITL review required: malicious + irreversible tool pending"
    return Action.SANDBOX, "sandboxed + audit log"