"""Input filter — the first layer of the protection system.

Scans the user prompt and any retrieved / untrusted content for
instruction-shaped injection BEFORE the LLM reads it as an instruction.
This is "anticipate attacks before building": the cheapest place to catch
an injection is at ingress, because the agent has not yet acted on it.

Reuses ``baselines.intent_signal.IntentSignalVerifier``'s instruction-shape
scorer (exemplar bank + keyword score) so the input filter and the
output-side detector never disagree on what "instruction-shaped" means.

Decision policy:
  * injection_shape(prompt or any untrusted response) >= tau_high -> BLOCK
    (the request carries an explicit override instruction; do not forward)
  * >= tau_low  -> SANDBOX (forward, but the agent is told this channel
    is untrusted and the tool-gateway + output-filter stay armed)
  * otherwise  -> ALLOW
"""
from __future__ import annotations

from baselines.base import BaseVerifier
from baselines.intent_signal import IntentSignalVerifier, _render_tool_response_text
from schemas import Action, Decision, Trajectory, Verdict

_UNTRUSTED = {"email", "web", "file", "unknown"}


class InputFilter(BaseVerifier):
    """Pre-LLM injection scan on the user prompt + untrusted content."""

    name = "input-filter"
    version = "1.0"

    def __init__(self, intent: IntentSignalVerifier | None = None,
                 tau_high: float = 0.75, tau_low: float = 0.45) -> None:
        self._intent = intent or IntentSignalVerifier()
        self.tau_high = tau_high
        self.tau_low = tau_low
        self._available = getattr(self._intent, "_available", False)

    def _max_instruction_shape(self, t: Trajectory) -> tuple[float, str]:
        """Max instruction-shape over the prompt + untrusted responses.

        Returns (score, channel) where channel names where the signal peaked.
        """
        best = 0.0
        where = "none"
        if self._available:
            s = self._intent._instruction_shape(t.task or "")
            if s > best:
                best, where = s, "user-prompt"
        for step in t.steps:
            if step.tool_response is None:
                continue
            if step.tool_response.source not in _UNTRUSTED:
                continue
            txt = _render_tool_response_text(step.tool_response.output)
            s = self._intent._instruction_shape(txt)
            if s > best:
                best, where = s, f"untrusted-response({step.tool_response.source})"
        return best, where

    def _verify(self, trajectory: Trajectory) -> Verdict:
        if not self._available:
            return Verdict(decision=Decision.BENIGN, confidence=0.0, action=Action.ALLOW,
                           notes=["input-filter: intent model unavailable"])
        score, where = self._max_instruction_shape(trajectory)
        if score >= self.tau_high:
            return Verdict(decision=Decision.MALICIOUS, confidence=score,
                           action=Action.BLOCK,
                           notes=[f"input-filter: instruction-shape {score:.2f} in {where} "
                                  f"(>= {self.tau_high}) — blocking at ingress"])
        if score >= self.tau_low:
            return Verdict(decision=Decision.SUSPICIOUS, confidence=score,
                           action=Action.SANDBOX,
                           notes=[f"input-filter: instruction-shape {score:.2f} in {where} "
                                  f"(>= {self.tau_low}) — forwarding as untrusted"])
        return Verdict(decision=Decision.BENIGN, confidence=1.0 - score, action=Action.ALLOW,
                       notes=[f"input-filter: max instruction-shape {score:.2f} in {where}"])
