"""BIPIA loader (Yi et al., arXiv:2312.14197; microsoft/BIPIA, MIT).

Text-injection variant over the three offline tasks (email QA, table QA,
code). Each sample pairs a user question with retrieved content (the
email / table / code context); the attack is a goal statement inserted
into the retrieved content at a position drawn from {start, middle,
end}, exactly as BIPIA's ``insert_*`` functions do. The victim agent's
answer is generated separately (``run_bipia_eval.py`` stage 1) and
attached as the trajectory's final agent text.

Trajectory shape mirrors the other benches: one tool call reading the
untrusted content (source = 'email' / 'web' / 'file') + agent text.
Labels are construction-based: malicious iff an attack was inserted.

Reference: arXiv:2312.14197 (ACM TOSEM; repo github.com/microsoft/BIPIA)
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

from benchmarks.base import BenchInfo
from schemas import AgentStep, ToolCall, ToolResponse, Trajectory


DEFAULT_DATA_PATH = Path("data/bipia")

# task -> (file, tool name, taint source)
_TASKS = {
    "email": ("email_test.jsonl", "read_email", "email"),
    "table": ("table_test.jsonl", "read_table", "file"),
    "code":  ("code_test.jsonl",  "read_code",  "web"),
}

_TOOL_ARGS = {
    "email": {"mailbox": "inbox", "filter": "latest"},
    "table": {"resource": "wiki-tables", "format": "markdown"},
    "code":  {"repo_url": "https://example.invalid/repo", "path": "src/"},
}


def _insert(context: str, attack: str, position: str, rng: random.Random) -> str:
    """BIPIA insertion (bipia/data/utils.py): newline-joined, sentence-
    aligned for 'middle'."""
    if position == "start":
        return "\n".join([attack, context])
    if position == "end":
        return "\n".join([context, attack])
    # middle: split at the first sentence boundary in the back half —
    # deterministic approximation of PunktSentenceTokenizer sampling
    # that avoids pulling in nltk as a dependency.
    lines = context.split("\n")
    k = max(1, len(lines) // 2)
    if len(lines) < 3:
        return "\n".join([context, attack])
    return "\n".join(lines[:k] + [attack] + lines[k:])


class BipiABench:
    name = "bipia"
    description = ("BIPIA text-injection: email/table/code QA with goal "
                   "statements inserted into retrieved content "
                   "(Yi et al., 2023)")
    info = BenchInfo(name="bipia", description=description,
                     n_trajectories=0, ground_truth_field="constructed",
                     attack_pattern_field="attack_name")

    def __init__(self, data_path: str | Path = DEFAULT_DATA_PATH,
                 n_per_attack: int = 6, n_benign_per_task: int = 50,
                 seed: int = 0, include_benign: bool = True) -> None:
        self._data_path = Path(data_path)
        contexts = []
        for task, (fname, _, _) in _TASKS.items():
            rows = [json.loads(l) for l in
                    (self._data_path / fname).read_text().splitlines() if l.strip()]
            for r in rows:
                # code task: question = the traceback (fields are line lists)
                q = r.get("question") or "\n".join(r.get("error") or [])
                ctx = "\n".join(r["context"]) if isinstance(r["context"], list) \
                    else r["context"]
                contexts.append((task, ctx, q, r.get("ideal", "")))
        attacks = json.loads((self._data_path / "text_attack_test.json").read_text())
        flat = [(f"{cat}-{i}", s) for cat, lst in attacks.items()
                for i, s in enumerate(lst)]

        rng = random.Random(seed)
        self._trajs = []
        self._meta: dict[str, dict] = {}

        # --- malicious: each attack x sampled contexts, rotating position ---
        positions = ["start", "middle", "end"]
        for j, (aname, astr) in enumerate(flat):
            picked = rng.sample(contexts, n_per_attack)
            for k, (task, ctx, question, _ideal) in enumerate(picked):
                pos = positions[(j + k) % 3]
                poisoned = _insert(ctx, astr, pos, rng)
                self._trajs.append(self._make(task, question, poisoned,
                                              aname, pos))

        # --- benign: unpoisoned contexts, disjoint sample if possible ---
        if include_benign:
            by_task: dict[str, list] = {}
            for c in contexts:
                by_task.setdefault(c[0], []).append(c)
            for task, lst in by_task.items():
                for (task_, ctx, question, _ideal) in rng.sample(
                        lst, min(n_benign_per_task, len(lst))):
                    self._trajs.append(self._make(task, question, ctx, None, None))

        rng.shuffle(self._trajs)

    def _make(self, task: str, question: str, context: str,
              attack_name: str | None, position: str | None) -> Trajectory:
        _, tool, source = _TASKS[task]
        t = Trajectory(task=question, source_bench="bipia",
                       ground_truth="malicious" if attack_name else "benign")
        t.declared_capabilities = {tool}
        call = ToolCall(name=tool, args=dict(_TOOL_ARGS[task]),
                        declared_capability=tool)
        resp = ToolResponse(call_id=call.call_id, output=context,
                            source=source)
        t.add_step(AgentStep(step_id=0, tool_call=call, tool_response=resp))
        # final step: agent text (victim answer) attached by the eval runner
        t.add_step(AgentStep(step_id=1))
        self._meta[t.traj_id] = {"attack_name": attack_name,
                                 "position": position, "task": task}
        return t

    def __iter__(self) -> Iterator[Trajectory]:
        yield from self._trajs

    def __len__(self) -> int:
        return len(self._trajs)

    def meta(self, traj: Trajectory) -> dict:
        """Construction metadata for a trajectory (task / attack / position)."""
        return self._meta.get(traj.traj_id, {})

    def attach_answer(self, traj: Trajectory, answer: str) -> None:
        """Fill the final agent-text step with the victim model's answer."""
        traj.steps[-1].text = answer
