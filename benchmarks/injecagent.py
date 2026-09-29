"""InjecAgent loader (Zhan et al., 2024).

Uses the test_cases_dh_base.json file which contains 510 complete attack
trajectories. Each record has both the user task and the injected
attacker payload in one place, which makes it directly usable as a
trajectory in our output-side verifier.

Reference: arXiv:2403.02691
"""
from __future__ import annotations

import json
import ast
from pathlib import Path
from typing import Iterator

from benchmarks.base import BenchInfo
from schemas import AgentStep, ToolCall, ToolResponse, Trajectory


DEFAULT_DATA_PATH = Path("data/InjecAgent/data/test_cases_dh_base.json")


def _parse_tool_params(s: str) -> dict:
    """InjecAgent stores tool params as a Python repr string. Parse it safely."""
    if not s or not isinstance(s, str):
        return {}
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return {}


def _parse_tool_response(s: str) -> object:
    """InjecAgent stores the response as a Python repr string with
    ``<Attacker Instruction>`` as a placeholder."""
    if not s or not isinstance(s, str):
        return {}
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return {"raw": s}


class InjecAgentBench:
    name = "injecagent"
    description = "InjecAgent: 510 indirect-injection cases (Zhan et al., 2024)"

    def __init__(self, data_path: str | Path = DEFAULT_DATA_PATH,
                 max_trajs: int | None = None,
                 contamination_fix: bool = True) -> None:
        self._data_path = Path(data_path)
        if not self._data_path.exists():
            raise FileNotFoundError(
                f"InjecAgent data not found at {self._data_path}. Clone:\n"
                f"  git clone --depth 1 https://github.com/uiuc-kang-lab/InjecAgent "
                f"{self._data_path.parent.parent}"
            )
        self._contamination_fix = contamination_fix
        with open(self._data_path) as f:
            raw = json.load(f)
        self._trajs = [self._convert(r) for r in raw]
        if max_trajs is not None:
            self._trajs = self._trajs[:max_trajs]
        self.info = BenchInfo(
            name=self.name, description=self.description,
            n_trajectories=len(self._trajs),
            ground_truth_field="ground_truth",
            attack_pattern_field="attack_type",
        )

    def _convert(self, record: dict) -> Trajectory:
        """Convert an InjecAgent test-case record into a Trajectory.

        Each record contains:
          - User Instruction (the agent's task)
          - Attacker Instruction (the injected payload)
          - User Tool / Tool Parameters (the legitimate call)
          - Tool Response (the response with <Attacker Instruction> embedded)
          - Thought (the agent's chain-of-thought, often empty in base set)
          - Attack Type (Physical Harm / Data Stealing / ...)
          - Modifed (whether the response was modified to embed injection)
        """
        user_task = record.get("User Instruction") or ""
        attacker_instr = record.get("Attacker Instruction") or ""
        user_tool = record.get("User Tool") or "unknown_tool"
        tool_params_str = record.get("Tool Parameters") or ""
        tool_response_str = record.get("Tool Response") or ""
        attack_type = record.get("Attack Type") or "unknown"
        modified = record.get("Modifed", 0)
        thought = record.get("Thought") or ""

        # Construct a real trajectory that our verifier can score.
        # Step 1: agent's reasoning
        # Step 2: agent's tool call (legitimate)
        # Step 3: tool response carrying the attacker payload
        # Step 4: agent's emitted text (often empty in InjecAgent base set)

        tool_params = _parse_tool_params(tool_params_str)
        tool_response = _parse_tool_response(tool_response_str)

        # Inject the attacker instruction into the response payload if needed.
        # If the response contains the literal placeholder, substitute.
        if isinstance(tool_response, dict):
            for k, v in list(tool_response.items()):
                if isinstance(v, str) and "<Attacker Instruction>" in v:
                    tool_response[k] = v.replace("<Attacker Instruction>",
                                                attacker_instr)
                elif isinstance(v, list):
                    new_list = []
                    for item in v:
                        if isinstance(item, dict):
                            new_item = {}
                            for k2, v2 in item.items():
                                if isinstance(v2, str) and "<Attacker Instruction>" in v2:
                                    new_item[k2] = v2.replace("<Attacker Instruction>",
                                                              attacker_instr)
                                else:
                                    new_item[k2] = v2
                            new_list.append(new_item)
                        else:
                            new_list.append(item)
                    tool_response[k] = new_list

        # Build the trajectory
        t = Trajectory(
            traj_id=f"injecagent/{record.get('User Tool', '?')[:30]}/{attack_type}",
            task=user_task,
            declared_capabilities={user_tool} if user_tool else set(),
            source_bench="injecagent",
            ground_truth="malicious" if modified else "benign",
            taint_map={},
        )

        # Step 0: agent's thought
        if thought:
            t.add_step(AgentStep(step_id=0, text=thought))

        # Step 1: legitimate user-tool call
        call_id = "c_legit"
        t.add_step(AgentStep(
            step_id=len(t.steps),
            text=f"Calling {user_tool} on user request.",
            tool_call=ToolCall(name=user_tool, args=tool_params, call_id=call_id),
            tool_response=ToolResponse(
                call_id=call_id,
                output=tool_response,
                source="email" if "email" in user_tool.lower() or "Gmail" in user_tool
                       else "web" if any(k in user_tool.lower() for k in ("amazon", "web", "fetch"))
                       else "file" if "file" in user_tool.lower()
                       else "unknown",
            ),
        ))

        # Step 2: the agent's emitted response. ONLY when the attack actually
        # executed (modified=1) do we model the agent as acting on the
        # injected instruction — appending the attacker instruction as emitted
        # text for a benign (modified=0) case would contaminate the benign
        # class with instruction-shaped content it should not contain,
        # inflating every text-aware classifier's FPR. The original
        # (pre-fix, contamination_fix=False) behavior appended it
        # unconditionally, which is what the §5.6 pre/post comparison
        # quantifies.
        if attacker_instr and (modified or not self._contamination_fix):
            # The injection model is: agent reads a tool response that contains
            # the attacker instruction, then echoes or acts on it.
            # We model this by appending a step where the agent emits the
            # attacker instruction as text (i.e., the attack manifests as
            # the agent doing what the injection says).
            t.add_step(AgentStep(
                step_id=len(t.steps),
                text=attacker_instr,
            ))

        return t

    def __iter__(self) -> Iterator[Trajectory]:
        return iter(self._trajs)

    def __len__(self) -> int:
        return len(self._trajs)


def attack_type_from_traj(t: Trajectory) -> str | None:
    if t.ground_truth != "malicious":
        return None
    return t.traj_id.rsplit("/", 1)[-1]