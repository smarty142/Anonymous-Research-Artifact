"""The layered agent protection system — composes the four defense layers
into one fail-closed pipeline.

Request path (secure-by-design):

   user prompt ─▶ [1] INPUT FILTER   (sync)  injection pre-scan at ingress
                       │
                       ▼
                 [2] CONTEXT GUARD   (sync)  secret/PII redaction + prompt
                       │                      minimization (privacy by design)
                       ▼
                   LLM AGENT
                       │  (each tool call)
                       ▼
                 [3] TOOL GATEWAY    (sync)  least-privilege per-call policy:
                       │                      capability tier, credential scope,
                       │                      irreversible -> HITL
                       ▼
                   tool executes ─▶ tool response ── loops back to agent
                       │
                       ▼  (agent done)
                 [4] OUTPUT FILTER   (sync)  det rules + learned guard
                       │                      (Llama Guard / GLM-4.6)
                       ▼
                 user / downstream system

   [async] audit log, conformal coverage check, telemetry

Fail-closed composition: the system-level action is the MOST CONSERVATIVE
across all layers (BLOCK > HITL > SANDBOX > ALLOW). This is the bypass-
handling guarantee: if the output filter (rules + Llama Guard) is bypassed,
(1) the input filter may have caught the injection at ingress, (2) the
tool gateway still gates every call by capability tier, and (3) any
irreversible action still requires HITL regardless of verdict. Blast radius
is bounded by the granted capability set, not by any single detector's
accuracy.

On the trajectory benchmarks the agent step has already occurred, so the
system evaluates the four layers POST-HOC against the recorded trajectory
and reports the combined verdict plus a per-layer breakdown (ablation).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pipeline.enforce import enforce
from pipeline.input_filter import InputFilter
from pipeline.context_guard import ContextGuard
from pipeline.tool_gateway import ToolGateway
from schemas import Action, Decision, Trajectory, Verdict

_ACTION_RANK = {Action.ALLOW: 0, Action.SANDBOX: 1, Action.HITL: 2, Action.BLOCK: 3}
_DECISION_RANK = {Decision.BENIGN: 0, Decision.SUSPICIOUS: 1, Decision.MALICIOUS: 2}


@dataclass
class LayerReport:
    layer: str
    decision: Decision
    action: Action
    note: str
    sync: bool


@dataclass
class SystemVerdict:
    decision: Decision
    action: Action
    layers: list[LayerReport] = field(default_factory=list)
    has_irreversible: bool = False
    notes: list[str] = field(default_factory=list)

    def to_verdict(self) -> Verdict:
        return Verdict(decision=self.decision, confidence=1.0 if self.decision is not Decision.BENIGN else 0.0,
                       action=self.action, notes=self.notes + [f"{l.layer}: {l.decision.value}/{l.action.value}"
                                                                  for l in self.layers])


class LayeredProtectionSystem:
    """Composes input filter + context guard + tool gateway + output filter."""

    name = "layered-system"
    version = "1.0"

    def __init__(self,
                 input_filter: InputFilter | None = None,
                 context_guard: ContextGuard | None = None,
                 tool_gateway: ToolGateway | None = None,
                 output_filter=None,                 # any Verdict-producing baseline (det / GLM)
                 enable: tuple[str, ...] = ("input", "context", "gateway", "output")) -> None:
        self.input_filter = input_filter or InputFilter()
        self.context_guard = context_guard or ContextGuard()
        self.tool_gateway = tool_gateway or ToolGateway()
        self.output_filter = output_filter
        self.enable = set(enable)

    # -- the per-trajectory evaluation --

    def evaluate(self, trajectory: Trajectory) -> SystemVerdict:
        layers: list[LayerReport] = []

        # [1] input filter (sync)
        if "input" in self.enable:
            iv = self.input_filter.verify(trajectory)
            layers.append(LayerReport("input-filter", iv.decision, iv.action,
                                      iv.notes[0] if iv.notes else "", sync=True))

        # [2] context guard (sync): leakage detection in output + redaction count
        if "context" in self.enable:
            cv = self.context_guard.verify(trajectory)
            layers.append(LayerReport("context-guard", cv.decision, cv.action,
                                      cv.notes[0] if cv.notes else "", sync=True))

        # [3] tool gateway (sync, per call): least-privilege policy
        if "gateway" in self.enable:
            gv = self.tool_gateway.verify(trajectory)
            layers.append(LayerReport("tool-gateway", gv.decision, gv.action,
                                      gv.notes[0] if gv.notes else "", sync=True))

        # [4] output filter (sync): det rules + learned guard
        if "output" in self.enable and self.output_filter is not None:
            ov = self.output_filter.verify(trajectory)
            layers.append(LayerReport("output-filter", ov.decision, ov.action,
                                      ov.notes[0] if ov.notes else "", sync=True))

        # Combine: most conservative decision across layers.
        worst_decision = Decision.BENIGN
        worst_action = Action.ALLOW
        for lr in layers:
            if _DECISION_RANK[lr.decision] > _DECISION_RANK[worst_decision]:
                worst_decision = lr.decision
            if _ACTION_RANK[lr.action] > _ACTION_RANK[worst_action]:
                worst_action = lr.action

        # Irreversible-action override: any gateway HITL means an irreversible
        # call is pending -> the system MUST NOT auto-execute, independent of
        # the output verdict. This is the fail-closed bypass guarantee.
        has_irreversible = any(lr.layer == "tool-gateway" and lr.action is Action.HITL
                               for lr in layers)
        if has_irreversible:
            worst_action = Action.HITL if worst_action is Action.ALLOW else worst_action

        return SystemVerdict(
            decision=worst_decision, action=worst_action,
            layers=layers, has_irreversible=has_irreversible,
            notes=[f"system: {len(layers)} layers, worst={worst_decision.value}/{worst_action.value}"],
        )

    # Verifier-protocol adapter so the system drops into the existing harness.
    def verify(self, trajectory: Trajectory) -> Verdict:
        return self.evaluate(trajectory).to_verdict()

    @staticmethod
    def request_path() -> list[tuple[str, str, bool]]:
        """Static description of the deployment request path for the paper.

        (layer, responsibility, synchronous?) — synchronous checks gate
        execution; asynchronous checks are audit/telemetry.
        """
        return [
            ("input-filter",  "injection pre-scan at ingress",                 True),
            ("context-guard", "secret/PII redaction + prompt minimization",    True),
            ("llm-agent",     "the model under protection",                     True),
            ("tool-gateway",  "least-privilege per-call policy gate",           True),
            ("output-filter", "det rules + learned guard before delivery",      True),
            ("audit-log",     "signed transcript + verdict telemetry",          False),
            ("conformal",     "distribution-free coverage monitor",             False),
        ]
