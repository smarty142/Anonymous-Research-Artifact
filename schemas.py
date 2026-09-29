"""Trajectory and verdict schemas — shared across pipeline, bench, baselines.

We use plain dataclasses (not pydantic) for the core types so the package
imports cleanly without pydantic installed in minimal environments.
Pydantic-style validation is implemented via ``__post_init__`` where useful.
"""
from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


# -----------------------------------------------------------------------------
# Agent trace
# -----------------------------------------------------------------------------

@dataclass
class ToolCall:
    """A single tool call emitted by the agent at one step."""
    name: str
    args: dict[str, Any]
    call_id: str = field(default_factory=lambda: f"c_{uuid.uuid4().hex[:8]}")
    # declared capability tier, if any (e.g., "read_email", "send_email_external")
    declared_capability: Optional[str] = None

    def arg(self, key: str, default: Any = None) -> Any:
        return self.args.get(key, default)


@dataclass
class ToolResponse:
    """Result of a tool call (may be the model-emulated or real one)."""
    call_id: str
    output: Any
    success: bool = True
    error: Optional[str] = None
    # Provenance: was the output tainted? (e.g., from email, web, file)
    source: str = "unknown"        # 'trusted' | 'email' | 'web' | 'file' | 'unknown'
    # Latency in ms (for TOCTOU / side-channel analysis)
    latency_ms: float = 0.0


@dataclass
class AgentStep:
    """One step in an agent trajectory: optionally a tool call + response + text."""
    step_id: int
    text: str = ""                          # any text emitted by the agent
    tool_call: Optional[ToolCall] = None
    tool_response: Optional[ToolResponse] = None
    # was this step preceded by an irreversible-confirmation step?
    had_confirm: bool = False
    timestamp: float = field(default_factory=time.time)

    def is_tool_step(self) -> bool:
        return self.tool_call is not None


@dataclass
class Trajectory:
    """A full agent run: sequence of steps + metadata."""
    task: str                                          # user instruction
    steps: list[AgentStep] = field(default_factory=list)
    # declared capabilities the agent SHOULD restrict itself to
    declared_capabilities: set[str] = field(default_factory=set)
    # trajectory-level metadata
    traj_id: str = field(default_factory=lambda: f"t_{uuid.uuid4().hex[:10]}")
    source_bench: str = "unknown"      # 'agentdojo' | 'injecagent' | 'osi-bench' | ...
    ground_truth: str = "benign"        # 'benign' | 'malicious' — ground-truth label
    # taint map: tool_call_id -> source label, populated for dataflow analysis
    taint_map: dict[str, str] = field(default_factory=dict)

    def add_step(self, step: AgentStep) -> None:
        step.step_id = len(self.steps)
        self.steps.append(step)
        # Auto-populate taint_map when a tool response is recorded
        if step.tool_response is not None and step.tool_call is not None:
            self.taint_map[step.tool_call.call_id] = step.tool_response.source

    def last_step(self) -> Optional[AgentStep]:
        return self.steps[-1] if self.steps else None

    def tool_calls(self) -> list[ToolCall]:
        return [s.tool_call for s in self.steps if s.tool_call is not None]

    def all_text(self) -> str:
        """Concatenate all text emitted by the agent."""
        return "\n".join(s.text for s in self.steps if s.text)


# -----------------------------------------------------------------------------
# Verdict types
# -----------------------------------------------------------------------------

class Decision(str, enum.Enum):
    BENIGN = "benign"
    SUSPICIOUS = "suspicious"
    MALICIOUS = "malicious"


class Action(str, enum.Enum):
    ALLOW = "allow"
    BLOCK = "block"
    SANDBOX = "sandbox"
    HITL = "hitl"               # human-in-the-loop review


@dataclass(frozen=True)
class ThreatMatch:
    """A single pattern that fired against a trajectory."""
    pattern_id: str
    pattern_name: str
    severity: str
    stage: str                         # which pipeline stage caught it
    location: str                      # e.g., 'step[3].tool_call.args[to]' or 'step[0].text'
    evidence: str                      # short snippet that triggered the match
    detector: str                      # name of the detector function


@dataclass
class Verdict:
    """Output of the verifier."""
    decision: Decision
    confidence: float                  # in [0, 1]
    action: Action
    matches: list[ThreatMatch] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    verifier_name: str = "unknown"
    verifier_version: str = "0.1.0"
    elapsed_ms: float = 0.0

    @property
    def is_attack_detected(self) -> bool:
        """A learning-friendly binary: anything non-benign counts as a detection."""
        return self.decision is not Decision.BENIGN

    def matched_pattern_ids(self) -> list[str]:
        return [m.pattern_id for m in self.matches]