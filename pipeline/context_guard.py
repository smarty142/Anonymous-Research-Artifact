"""Context guard — secrets/PII redaction, leakage detection, and prompt
minimization. The privacy-by-design layer of the protection system.

Two roles:
  (1) INPUT-side (privacy by design): redact secrets/PII from the context
      the agent carries, so a later exfiltration attempt has nothing
      sensitive to send. What counts as "sensitive" is configured in
      ``taxonomy/sensitivity.yaml`` — domain-dependent (finance vs
      healthcare vs e-commerce swap the regex categories).
  (2) OUTPUT-side (leakage detection): if the agent's EMITTED text
      contains a secret/PII pattern, that is a leakage finding (taxonomy
      T.B.13 PII, T.A.3 exfiltration, T.B.6 system-prompt leak) -> flag.

The guard is deterministic and dependency-free (regex only, no model call).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from baselines.base import BaseVerifier
from schemas import Action, Decision, Trajectory, Verdict

_POLICY_PATH = Path(__file__).resolve().parent.parent / "taxonomy" / "sensitivity.yaml"
_UNTRUSTED = {"email", "web", "file", "unknown"}


@dataclass
class Finding:
    category: str
    snippet: str
    in_output: bool


@dataclass
class RedactionResult:
    redacted_text: str
    findings: list[Finding] = field(default_factory=list)
    n_redactions: int = 0


class SensitivityPolicy:
    def __init__(self, path: Path = _POLICY_PATH) -> None:
        with open(path) as f:
            self._doc = yaml.safe_load(f)
        self.default_redact_to = self._doc.get("default_redact_to", "[REDACTED]")
        self.categories = self._doc.get("categories", [])
        self.minimization = self._doc.get("minimization", {})
        # Compile patterns once.
        self._compiled = []
        for cat in self.categories:
            for pat in cat.get("patterns", []):
                try:
                    self._compiled.append((re.compile(pat), cat))
                except re.error:
                    pass

    def scan(self, text: str) -> list[Finding]:
        if not text:
            return []
        out = []
        for rx, cat in self._compiled:
            for m in rx.finditer(text):
                out.append(Finding(category=cat["name"],
                                   snippet=m.group(0)[:60],
                                   in_output=bool(cat.get("in_output", True))))
        return out

    def redact(self, text: str) -> RedactionResult:
        if not text:
            return RedactionResult(redacted_text=text)
        redacted = text
        findings = []
        n = 0
        for rx, cat in self._compiled:
            token = cat.get("redact_to", self.default_redact_to)
            new, count = rx.subn(token, redacted)
            if count:
                redacted = new
                n += count
                findings.append(Finding(category=cat["name"],
                                        snippet=token, in_output=bool(cat.get("in_output", True))))
        return RedactionResult(redacted_text=redacted, findings=findings, n_redactions=n)


class ContextGuard(BaseVerifier):
    """Redacts sensitive data from context and flags leakage in output."""

    name = "context-guard"
    version = "1.0"

    def __init__(self, policy: SensitivityPolicy | None = None) -> None:
        self.policy = policy or SensitivityPolicy()

    # -- input-side: redaction + minimization --

    def redact_text(self, text: str) -> RedactionResult:
        return self.policy.redact(text)

    def minimize_trajectory(self, t: Trajectory) -> Trajectory:
        """Return a copy with untrusted tool responses truncated/dropped
        to bound how much untrusted text flows into later context. This is
        prompt/context minimization: the agent gets only what it needs."""
        import copy
        budget = int(self.policy.minimization.get("untrusted_step_char_budget", 2000))
        drop = bool(self.policy.minimization.get("drop_unreferenced_untrusted", True))
        t2 = copy.deepcopy(t)
        for s in t2.steps:
            if s.tool_response is None:
                continue
            if s.tool_response.source not in _UNTRUSTED:
                continue
            out = s.tool_response.output
            txt = out if isinstance(out, str) else str(out)
            if drop and len(txt) > budget:
                s.tool_response.output = txt[:budget] + "...[truncated by context-guard]"
        return t2

    # -- output-side: leakage detection --

    def _leakage_findings(self, t: Trajectory) -> list[Finding]:
        out = []
        for s in t.steps:
            if not s.text:
                continue
            for f in self.policy.scan(s.text):
                if f.in_output:
                    out.append(f)
        return out

    def _verify(self, trajectory: Trajectory) -> Verdict:
        leaks = self._leakage_findings(trajectory)
        if leaks:
            cats = sorted({f.category for f in leaks})
            return Verdict(
                decision=Decision.MALICIOUS, confidence=0.9,
                action=Action.BLOCK, matches=[],
                notes=[f"context-guard: leakage of {cats} in emitted text "
                       f"({len(leaks)} match(es))"],
            )
        return Verdict(
            decision=Decision.BENIGN, confidence=0.0, action=Action.ALLOW,
            notes=["context-guard: no leakage patterns in emitted text"],
        )
