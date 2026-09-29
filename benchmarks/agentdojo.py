"""AgentDojo loader (Debenedetti et al., 2024).

Loads AgentDojo trajectory files from a pre-computed ``runs/`` directory
shipped in the upstream repo. We do NOT depend on the ``agentdojo``
Python package (which requires Python 3.10+); we read the JSON trajectory
files directly.

Each file is a full conversation: system prompt, user task, assistant
tool calls, tool responses. We convert to our ``Trajectory`` schema by
extracting tool calls + responses from the message log. The
``ground_truth`` is "malicious" if the attack type is anything other
than "none" (the no-injection baseline), "benign" otherwise.

Reference: arXiv:2406.13352
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from benchmarks.base import BenchInfo
from schemas import AgentStep, ToolCall, ToolResponse, Trajectory


# Default: gpt-3.5-turbo-0125 is shipped in the upstream repo
DEFAULT_RUNS_ROOT = Path("agentdojo_clone/runs/gpt-3.5-turbo-0125")
DEFAULT_SUITE = "workspace"


def _message_to_steps(messages: list[dict]) -> tuple[list[AgentStep], set[str]]:
    """Convert a list of chat-format messages into our Trajectory.steps.

    Returns the list of steps and the set of declared tool names mentioned.
    """
    steps: list[AgentStep] = []
    declared: set[str] = set()
    pending_calls: dict[str, ToolCall] = {}

    for m in messages:
        role = m.get("role")
        if role == "system":
            # Surface the system prompt as a text step (it often names tools)
            content = m.get("content") or ""
            if content:
                steps.append(AgentStep(step_id=len(steps), text=content))
                # Look for tool names mentioned in the system prompt
                for tool_name in (
                    "search_files_by_filename", "read_file", "send_email",
                    "list_files", "get_file", "delete_file", "write_file",
                    "fetch_url", "post_webhook", "db_query", "shell_exec",
                    "delegate", "transfer_money", "drop_table",
                    "list_emails", "read_email", "delete_email",
                    "AmazonGetProductDetails", "GmailSendEmail",
                    "AugustSmartLockGrantGuestAccess", "FileWrite",
                    "ScheduleMeeting", "SendMoney", "AddContact",
                ):
                    if tool_name in content:
                        declared.add(tool_name)
        elif role == "user":
            content = m.get("content") or ""
            if content:
                steps.append(AgentStep(step_id=len(steps), text=content))
        elif role == "assistant":
            content = m.get("content") or ""
            tool_calls = m.get("tool_calls") or []
            if tool_calls:
                # One step per assistant message; possibly multiple tool calls
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    if isinstance(fn, str):
                        # Some AgentDojo runs store function as a bare string
                        name = fn
                        args = tc.get("args", {}) or {}
                    else:
                        name = (fn.get("function") or fn.get("name")
                                or "unknown_tool")
                        args = fn.get("args") or fn.get("arguments") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            try:
                                import ast
                                args = ast.literal_eval(args)
                            except Exception:
                                args = {}
                    declared.add(name)
                    call_id = tc.get("id", f"c_{len(pending_calls)}")
                    pending_calls[call_id] = ToolCall(
                        name=name, args=args, call_id=call_id,
                    )
                    steps.append(AgentStep(
                        step_id=len(steps),
                        text=content if content else None,
                        tool_call=ToolCall(name=name, args=args, call_id=call_id),
                    ))
            else:
                if content:
                    steps.append(AgentStep(step_id=len(steps), text=content))
        elif role == "tool":
            call_id = m.get("tool_call_id", "")
            content = m.get("content") or ""
            tool_call_info = m.get("tool_call", {})
            # Tool call metadata gives us the source hint
            source = "unknown"
            if isinstance(tool_call_info, dict):
                fn = tool_call_info.get("function", "")
            elif isinstance(tool_call_info, str):
                fn = tool_call_info
            else:
                fn = ""
            if "email" in str(fn).lower() or "gmail" in str(fn).lower():
                source = "email"
            elif any(k in str(fn).lower() for k in ("amazon", "fetch", "url", "web")):
                source = "web"
            elif "file" in str(fn).lower():
                source = "file"
            steps.append(AgentStep(
                step_id=len(steps),
                tool_response=ToolResponse(
                    call_id=call_id,
                    output=content,
                    source=source,
                ),
            ))

    return steps, declared


class AgentDojoBench:
    name = "agentdojo"
    description = "AgentDojo: 753 pre-computed trajectories from gpt-3.5-turbo runs"

    def __init__(self, runs_root: str | Path = DEFAULT_RUNS_ROOT,
                 suite: str = DEFAULT_SUITE,
                 max_trajs: int | None = None) -> None:
        self._runs_root = Path(runs_root)
        self._suite = suite
        if not self._runs_root.exists():
            raise FileNotFoundError(
                f"AgentDojo runs not found at {self._runs_root}. Clone:\n"
                f"  git clone --depth 1 https://github.com/ethz-spylab/agentdojo.git\n"
                f"so that {self._runs_root}/workspace exists."
            )
        suite_root = self._runs_root / suite
        if not suite_root.exists():
            raise FileNotFoundError(f"Suite '{suite}' not found at {suite_root}")
        self._files = sorted(suite_root.glob("user_task_*/**/*.json"))
        if max_trajs is not None:
            self._files = self._files[:max_trajs]
        self.info = BenchInfo(
            name=self.name, description=self.description,
            n_trajectories=len(self._files),
            ground_truth_field="ground_truth",
            attack_pattern_field="attack_type",
        )

    def _convert(self, path: Path) -> Trajectory:
        with open(path) as f:
            record = json.load(f)
        messages = record.get("messages", [])
        steps, declared = _message_to_steps(messages)
        attack_type = record.get("attack_type", "none")
        # Inject the actual injection payload into the trajectory text
        # (AgentDojo's tool responses are un-injected; the injection is
        # implicit in the trajectory). We add it as a final text step so
        # our detectors can see the attack payload.
        injections = record.get("injections", {})
        injection_text = ""
        for inj_value in injections.values():
            if isinstance(inj_value, str):
                injection_text += inj_value + "\n"
        if injection_text.strip() and attack_type != "none":
            steps.append(AgentStep(step_id=len(steps), text=injection_text.strip()))
        is_malicious = attack_type != "none"
        t = Trajectory(
            traj_id=f"agentdojo/{path.relative_to(self._runs_root / self._suite).as_posix()}",
            task=str(record.get("user_task_id", "?")),
            declared_capabilities=declared,
            source_bench="agentdojo",
            ground_truth="malicious" if is_malicious else "benign",
        )
        # Avoid double-counting step_id; reassign
        for i, s in enumerate(steps):
            s.step_id = i
        # add_step populates taint_map from tool_responses
        for s in steps:
            t.add_step(s)
        return t

    def __iter__(self) -> Iterator[Trajectory]:
        for path in self._files:
            yield self._convert(path)

    def __len__(self) -> int:
        return len(self._files)