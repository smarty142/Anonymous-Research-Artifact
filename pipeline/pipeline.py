"""Pipeline orchestrator: runs the 5 stages in order on a Trajectory."""
from __future__ import annotations

import time

from pipeline.context import PipelineContext
from pipeline.decide import decide
from pipeline.enforce import enforce
from pipeline.parse import parse
from pipeline.semantic_check import semantic_check
from pipeline.threat_match import threat_match
from pipeline.tokenize import tokenize
from schemas import Action, Verdict
from taxonomy.loader import Taxonomy

_IRREVERSIBLE_TOOLS = frozenset({
    "send_email", "send_message", "transfer_money", "delete_file",
    "delete_email", "drop_table", "write_file", "shell_exec",
    "run_command", "execute_sql", "deploy", "rollback",
})


class DeterministicPipeline:
    """The full deterministic verification pipeline."""

    name = "det-pipeline"
    version = "0.1.0"

    def __init__(self, taxonomy: Taxonomy | None = None) -> None:
        self.taxonomy = taxonomy or Taxonomy.load_default()

    def verify(self, trajectory) -> Verdict:
        t0 = time.perf_counter()
        ctx = PipelineContext(trajectory=trajectory)
        # Stash the taxonomy so each stage can resolve detector -> pattern
        setattr(ctx, "_taxonomy", self.taxonomy)
        # Stage 1
        tokenize(ctx)
        # Stage 2
        parse(ctx)
        # Stage 3 (threat match + taint flow + semantic check)
        threat_match(ctx)
        # Stage 3.5: semantic embedding check (optional)
        semantic_check(ctx)
        # Stage 4
        decision, confidence = decide(ctx)
        # Stage 5
        has_irreversible = any(
            s.tool_call is not None and s.tool_call.name in _IRREVERSIBLE_TOOLS
            for s in trajectory.steps
        )
        action, note = enforce(decision, confidence, has_irreversible)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        v = Verdict(
            decision=decision,
            confidence=confidence,
            action=action,
            matches=list(ctx.matches),
            notes=list(ctx.notes) + [f"enforce: {note}",
                                     f"matched_patterns={len(ctx.matches)}"],
            verifier_name=self.name,
            verifier_version=self.version,
            elapsed_ms=elapsed_ms,
        )
        return v