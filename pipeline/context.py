"""PipelineContext: shared state passed through the 5-stage pipeline.

Stages mutate this context by appending :class:`ThreatMatch` objects to
``ctx.matches`` and recording stage-local artefacts (normalized text,
AST, taint flows). Stage 4 reads ``ctx.matches`` to compute a verdict;
stage 5 reads the verdict to compute an action.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from schemas import ThreatMatch, Trajectory


@dataclass
class TaintFlow:
    """A data-flow edge: source (call_id) -> sink (location) with taint label."""
    source_call_id: str
    source_label: str                     # 'email' | 'web' | 'file' | 'unknown' | ...
    sink_location: str                    # e.g., 'step[3].tool_call.args[to]'
    sink_kind: str                        # 'identifier' | 'control_flow' | 'sink' | ...
    sink_value_snippet: str = ""


@dataclass
class ParsedCall:
    """AST view of one tool call, produced by stage 2."""
    step_id: int
    tool_name: str
    args: dict[str, Any]
    extra_keys: list[str] = field(default_factory=list)
    type_errors: list[str] = field(default_factory=list)


@dataclass
class PipelineContext:
    trajectory: Trajectory
    matches: list[ThreatMatch] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # Stage artefacts
    normalized_text: dict[int, str] = field(default_factory=dict)
    flagged_chars: dict[int, list[tuple[int, str, str]]] = field(default_factory=dict)
    ast: dict[int, ParsedCall] = field(default_factory=dict)
    taint_flows: list[TaintFlow] = field(default_factory=dict if False else list)  # type: ignore
    declared_capabilities: set[str] = field(default_factory=set)

    def add_match(self, m: ThreatMatch) -> None:
        self.matches.append(m)

    def add_note(self, note: str) -> None:
        self.notes.append(note)

    # ------------------------------------------------------------------
    # Detector-name based match: emits one ThreatMatch per pattern sharing
    # this detector name. This fixes the case where multiple taxonomy
    # patterns share a detector (e.g., T.A.4 and T.C.11 both use
    # ``path_traversal``) — every applicable pattern should fire.
    # ------------------------------------------------------------------
    def emit(self, detector_name: str, step_id: int,
             location: str, evidence: str, default_stage: str = "threat_match") -> None:
        tax = getattr(self, "_taxonomy", None)
        patterns = []
        if tax is not None:
            patterns = [p for p in tax.patterns if p.detector == detector_name]
        if not patterns:
            # emit a stub match so evidence is not lost
            self.add_match(ThreatMatch(
                pattern_id="?", pattern_name=detector_name,
                severity="medium", stage=default_stage,
                location=location, evidence=evidence, detector=detector_name,
            ))
            return
        for p in patterns:
            self.add_match(ThreatMatch(
                pattern_id=p.id, pattern_name=p.name,
                severity=p.severity.value, stage=p.stage.value,
                location=location, evidence=evidence, detector=detector_name,
            ))