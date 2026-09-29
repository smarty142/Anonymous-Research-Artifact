"""Least-privilege tool gateway — the deterministic policy layer that gates
EVERY tool call before execution.

This is the "deterministic policy checks on every tool call" layer of the
protection system, and the instantiation of least privilege as a first-class
defense. It composes the capability schema (``taxonomy/capabilities.yaml``)
with the existing classification logic in ``pipeline.parse`` so the gateway
stays consistent with the T.A.* capability detectors already in the pipeline.

Enforcement (fail-closed):
  * a tool call outside the session's granted tiers  -> BLOCK (escalate)
  * a tool call in an irreversible/credential/admin/delegate tier
    -> HITL (human approval), independently of the output-side verdict
  * a call whose tier is granted but targets a sensitive path / sink
    -> SANDBOX + audit
  * everything else -> ALLOW (audited)

The key property: even if the output-side filter (rules + Llama Guard) is
bypassed and a malicious irreversible action reaches the gateway, it STILL
cannot execute without a human. Blast radius is bounded by the granted
capability set, not by the detector's accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from pipeline import parse as _parse   # reuse existing classification sets
from schemas import Action, Decision, Trajectory, Verdict

_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "taxonomy" / "capabilities.yaml"


@dataclass
class GatewayDecision:
    tool: str
    tier: str
    action: Action
    reason: str
    needs_confirmation: bool = False
    sink_args_tainted: list = field(default_factory=list)


class CapabilitySchema:
    """Loads taxonomy/capabilities.yaml; classifies tools into tiers."""

    def __init__(self, path: Path = _SCHEMA_PATH) -> None:
        with open(path) as f:
            self._doc = yaml.safe_load(f)
        self.tiers = self._doc["tiers"]
        self.default_grant = set(self._doc.get("default_grant", ["read", "write", "network"]))
        self.confirmation = self._doc.get("confirmation", {})
        self.sensitive_paths = self.confirmation.get("sensitive_path_patterns", [])
        self.irreversible_tiers = set(self.confirmation.get("irreversible_tiers", []))

    def classify(self, tool_name: str, args: Optional[dict] = None) -> str:
        """Map a tool name to a privilege tier.

        Order matters: most-sensitive first (a tool can match multiple
        suffixes). Reuses ``pipeline.parse``'s authoritative sets so the
        gateway never disagrees with the T.A.* capability detectors.
        """
        t = (tool_name or "").lower()
        if not t:
            return "read"
        if tool_name in _parse._CREDENTIAL_TOOLS:
            return "credential"
        if tool_name in _parse._IRREVERSIBLE_TOOLS:
            return "irreversible"
        if any(k in t for k in ("shell", "exec", "run_command", "sudo")):
            return "admin"
        if any(k in t for k in ("delegate", "spawn", "subagent")):
            return "delegate"
        # suffix-based fallback, mirroring parse.py
        if any(t.endswith(s) for s in ("_send", "_transfer", "_delete", "_drop",
                                       "_deploy", "_publish", "_post")):
            return "irreversible"
        if any(t.endswith(s) for s in ("_write", "_set", "_create", "_update",
                                       "_put", "_add", "_remove")):
            return "write"
        if any(t.endswith(s) for s in ("_get", "_read", "_list", "_search",
                                       "_find", "_fetch", "_view", "_show")):
            return "read"
        if any(k in t for k in ("http", "url", "web", "fetch", "request")):
            return "network"
        return "read"   # fail-safe default tier is the lowest privilege

    def tier_policy(self, tier: str) -> dict:
        return self.tiers.get(tier, {}).get("policy", {})

    def grants(self, tier: str, granted: set[str]) -> bool:
        return tier in granted


class ToolGateway:
    """Per-call policy gate. Stateless across calls; stateful only via the
    session's granted tier set (passed in)."""

    name = "tool-gateway"
    version = "1.0"

    def __init__(self, schema: CapabilitySchema | None = None,
                 granted: Optional[set[str]] = None) -> None:
        self.schema = schema or CapabilitySchema()
        self.granted = set(granted) if granted is not None else set(self.schema.default_grant)

    def check_call(self, tool_name: str, args: dict) -> GatewayDecision:
        args = args or {}
        tier = self.schema.classify(tool_name, args)
        policy = self.schema.tier_policy(tier)

        # 1. Capability grant: tier not granted -> block (escalate).
        if not self.schema.grants(tier, self.granted):
            return GatewayDecision(
                tool=tool_name, tier=tier, action=Action.BLOCK,
                reason=f"tier '{tier}' not in session grant {sorted(self.granted)}")

        # 2. Sensitive-path targeting (even read tier) -> sandbox.
        blob = " ".join(str(v) for v in args.values()).lower()
        sensitive_hit = next((p for p in self.schema.sensitive_paths if p.lower() in blob), None)

        # 3. Confirmation policy: irreversible/credential/admin/delegate -> HITL.
        needs_confirm = policy.get("confirmation") == "human" or tier in self.schema.irreversible_tiers

        # 4. Tainted sink args (arg value appears sensitive) -> flag for audit.
        sink_keys = set()
        for group in self.schema._doc.get("sink_args", {}).values():
            sink_keys.update(group)
        tainted = [k for k, v in args.items()
                   if k in sink_keys and isinstance(v, str)
                   and any(s in v.lower() for s in ("evil", "attacker", "exfil",
                                    "<attacker", "169.254", "..", "drop", "delete"))]

        if needs_confirm:
            return GatewayDecision(
                tool=tool_name, tier=tier, action=Action.HITL,
                reason=f"irreversible/privileged tier '{tier}' requires human approval",
                needs_confirmation=True, sink_args_tainted=tainted)
        if sensitive_hit or tainted:
            return GatewayDecision(
                tool=tool_name, tier=tier, action=Action.SANDBOX,
                reason=(f"sensitive target '{sensitive_hit}'" if sensitive_hit
                        else f"tainted sink args {tainted}"),
                sink_args_tainted=tainted)
        if policy.get("sandbox"):
            return GatewayDecision(
                tool=tool_name, tier=tier, action=Action.SANDBOX,
                reason=f"tier '{tier}' runs sandboxed")
        return GatewayDecision(
            tool=tool_name, tier=tier, action=Action.ALLOW,
            reason=f"tier '{tier}' granted, low blast radius")

    def verify(self, trajectory: Trajectory) -> Verdict:
        """Aggregate per-call decisions over a trajectory into one verdict.

        The gateway's verdict is the most conservative action across all
        calls (BLOCK > HITL > SANDBOX > ALLOW), so a single privileged
        call escalates the whole trajectory.
        """
        worst = Action.ALLOW
        reason = "no tool calls gated"
        any_tainted = False
        calls = 0
        for s in trajectory.steps:
            if s.tool_call is None:
                continue
            calls += 1
            d = self.check_call(s.tool_call.name, s.tool_call.args)
            if d.sink_args_tainted:
                any_tainted = True
            order = {Action.ALLOW: 0, Action.SANDBOX: 1, Action.HITL: 2, Action.BLOCK: 3}
            if order[d.action] > order[worst]:
                worst = d.action
                reason = f"{d.tool}({d.tier}): {d.reason}"
        # Map back to a Decision for the unified metrics harness.
        if worst == Action.BLOCK:
            decision = Decision.MALICIOUS
        elif worst == Action.HITL:
            decision = Decision.SUSPICIOUS
        elif worst == Action.SANDBOX:
            decision = Decision.SUSPICIOUS if any_tainted else Decision.BENIGN
        else:
            decision = Decision.BENIGN
        return Verdict(
            decision=decision, confidence=1.0 if worst != Action.ALLOW else 0.0,
            action=worst, matches=[],
            notes=[f"tool-gateway: {calls} calls, worst={worst.value}; {reason}"],
        )
