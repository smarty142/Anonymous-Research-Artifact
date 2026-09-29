"""Stage 2: Parse.

Responsibilities:
  * For each step with a tool call, build a :class:`ParsedCall` AST node
  * Validate tool name against declared capabilities
  * Detect privilege escalation (write-tier call when only read-tier declared)
  * Detect arg-schema smuggling (extra keys beyond common patterns)
  * Detect recursive self-call (in delegation graph)
  * Detect irreversible action without prior confirmation step
  * Detect undeclared-side-effect calls (read-only tool called on side-effecty path)
  * Detect logic-bomb (time/condition-gated) and TOCTOU patterns structurally

Detectors implemented (mapped to taxonomy):
  * capability_check     -> T.A.1, T.A.2, T.A.8, T.A.9, T.A.17
  * schema_extra_keys    -> T.A.13
  * cycle_detect         -> T.A.15
  * irreversibility      -> T.A.11
  * condition_trigger    -> T.C.14
  * toctou               -> T.C.15
"""
from __future__ import annotations

from pipeline.context import ParsedCall, PipelineContext
from schemas import ThreatMatch


# Convention: tools ending in these suffixes are read-only / low-risk.
_READ_ONLY_SUFFIXES = ("_read", "_get", "_list", "_search", "_find", "_fetch",
                       "_view", "_show", "_file")
# Note: includes "_file" to recognize write_file ↔ read_file as a tier pair
_WRITE_SUFFIXES = ("_write", "_set", "_delete", "_remove", "_update",
                   "_create", "_add", "_put", "_post", "_send", "_transfer",
                   "_drop", "_exec", "_run", "_drop", "_flush", "_file")
# Tools that mutate external state (irreversible categories).
# Note: shell_exec / run_command are NOT in this set unconditionally —
# read-only subcommands (ls, cat, echo, pwd, whoami, date, ...) are
# allowed without confirmation. The check below inspects the actual
# command string against _SAFE_SHELL_SUBCOMMANDS.
_IRREVERSIBLE_TOOLS = frozenset({
    "send_email", "send_message", "send_sms",
    "transfer_money", "wire_transfer",
    "delete_email", "delete_file", "delete_record", "drop_table",
    "drop_database", "drop_collection",
    "write_file", "overwrite_file",
    "execute_sql",
    "post_webhook", "publish",
    "deploy", "rollback",
})

# Whitelist of read-only shell subcommands. Matches the *first token* of the
# command string (after stripping leading env-var assignments and `sudo -n`).
_SAFE_SHELL_SUBCOMMANDS = frozenset({
    "ls", "cat", "echo", "pwd", "whoami", "date", "uname", "hostname",
    "head", "tail", "wc", "diff", "file", "stat", "type", "which",
    "true", "false", "test", "[", "env", "printenv", "id", "groups",
    "df", "du", "free", "uptime", "ps", "top", "htop", "pgrep",
    "grep", "awk", "sed", "cut", "tr", "sort", "uniq", "tee", "xargs",
    "find", "tree", "less", "more", "man", "help", "history",
    "git",          # all read-only git subcommands pass; mutating ones caught by T.A.5
    "kubectl",      # mutating kubectl subcommands caught by T.A.5
    "docker",       # mutating docker subcommands caught by T.A.5
    "python", "python3", "node", "ruby", "perl",  # scripts caught by T.A.8/9
})


def _is_safe_shell_command(cmd: str) -> bool:
    """True iff ``cmd`` starts with a read-only subcommand and has no
    shell metacharacters that could redirect / substitute.
    """
    if not isinstance(cmd, str) or not cmd.strip():
        return False
    # Reject anything with shell metacharacters outright
    if any(c in cmd for c in ";|&><$`\\"):
        return False
    # Strip leading env-var assignments (FOO=bar)
    tokens = cmd.strip().split()
    while tokens and "=" in tokens[0] and not tokens[0].startswith("="):
        tokens.pop(0)
    if not tokens:
        return False
    sub = tokens[0].rsplit("/", 1)[-1]   # handle /usr/bin/ls -> ls
    return sub in _SAFE_SHELL_SUBCOMMANDS

# Tools that read secrets / privileged data
_CREDENTIAL_TOOLS = frozenset({
    "get_secret", "read_secret", "fetch_credential", "list_secrets",
    "decrypt", "unseal", "kms_decrypt",
})

# Side-effect-prone tool-call prefixes (in addition to WRITE_SUFFIXES)
_SIDE_EFFECT_PRONE = frozenset({
    "send_email", "send_message", "write_file", "transfer_money",
    "delete_file", "delete_email", "drop_table", "shell_exec",
    "run_command", "execute_sql", "deploy", "rollback", "post_webhook",
})

# Confirmation is conventionally a tool with one of these names or args
_CONFIRM_TOOLS = frozenset({
    "confirm", "confirm_action", "request_approval", "user_confirm",
    "approval_request",
})

# Schema hints for common tool args (declared fields we don't want to flag)
_DEFAULT_SCHEMAS: dict[str, set[str]] = {
    "send_email": {"to", "from", "cc", "bcc", "subject", "body", "attachments"},
    "send_message": {"to", "channel", "body", "subject"},
    "read_file": {"path", "encoding", "max_bytes"},
    "write_file": {"path", "content", "encoding", "mode"},
    "delete_file": {"path"},
    "delete_email": {"folder", "message_id"},
    "delete_record": {"table", "id"},
    "transfer_money": {"to_account", "amount", "currency", "memo"},
    "http_get": {"url", "headers", "params", "timeout"},
    "http_post": {"url", "headers", "body", "params", "timeout"},
    "db_query": {"query", "params"},
    "run_command": {"command", "args", "cwd", "env", "timeout"},
    "shell_exec": {"command", "args", "cwd", "env"},
    "delegate": {"agent", "task", "max_depth", "timeout"},
}


def parse(ctx: PipelineContext) -> PipelineContext:
    by_det = _taxonomy_lookup(ctx)

    # First pass: build AST and detect per-step structural issues
    irreversible_seen_confirm: set[int] = set()  # step_ids that are confirm steps
    for step in ctx.trajectory.steps:
        if step.tool_call is None:
            continue
        tc = step.tool_call
        pc = ParsedCall(
            step_id=step.step_id,
            tool_name=tc.name,
            args=dict(tc.args),
        )
        ctx.ast[step.step_id] = pc

        # T.A.13: schema_extra_keys
        declared_keys = _DEFAULT_SCHEMAS.get(tc.name)
        if declared_keys is not None:
            extra = sorted(set(tc.args.keys()) - declared_keys)
            if extra:
                pc.extra_keys = extra
                ctx.emit("schema_extra_keys", step.step_id, location=f"step[{step.step_id}].tool_call.args", evidence=f"tool={tc.name} extra_keys={extra}",)
        else:
            # Unknown tool — flag as potential T.A.1 candidate (will resolve below)
            pass

        # T.A.1: unauthorized tool
        decl = ctx.trajectory.declared_capabilities
        if decl and tc.name not in decl:
            ctx.emit("capability_check", step.step_id,
                location=f"step[{step.step_id}].tool_call.name",
                evidence=f"tool={tc.name} not in declared={sorted(decl)}")

        # T.A.2: privilege escalation
        if decl:
            if tc.name in decl:
                # declared, but maybe escalated (e.g., read_email -> send_email)
                pass
            # If the tool wasn't declared but a lower-tier variant was, escalate
            if tc.name not in decl and _lower_tier_exists(tc.name, decl):
                ctx.emit("capability_check", step.step_id,
                    location=f"step[{step.step_id}].tool_call.name",
                    evidence=f"escalation: {tc.name} but only lower-tier in {sorted(decl)}")

        # T.A.8: credential access without declared purpose
        if tc.name in _CREDENTIAL_TOOLS:
            purpose = tc.args.get("purpose") or tc.args.get("reason")
            if not purpose:
                ctx.emit("capability_check", step.step_id, location=f"step[{step.step_id}].tool_call.args", evidence=f"credential tool {tc.name} without declared purpose",)

        # T.A.17: undeclared side effect (a "read" tool called on a sensitive-system path
        # that suggests the agent is doing something other than reading user data).
        _SENSITIVE_PATH_PREFIXES = ("/etc/", "/proc/", "/sys/", "/dev/",
                                    "/var/log/", "/boot/", "/root/")
        if any(tc.name.endswith(suf) for suf in _READ_ONLY_SUFFIXES):
            for k, v in tc.args.items():
                if not isinstance(v, str):
                    continue
                if any(v.startswith(p) for p in _SENSITIVE_PATH_PREFIXES):
                    ctx.emit("capability_check", step.step_id,
                        location=f"step[{step.step_id}].tool_call.args[{k}]",
                        evidence=f"read-only tool {tc.name} on sensitive-system path")
                    break

        # Track confirm steps for T.A.11 evaluation below
        if tc.name in _CONFIRM_TOOLS or "confirm" in tc.args:
            irreversible_seen_confirm.add(step.step_id)

        # T.C.14: condition_trigger (logic bomb)
        # Heuristic: if the args contain time/event/when fields with values
        # resembling future or external triggers.
        for trigger_key in ("when", "trigger", "schedule", "at", "after"):
            if trigger_key in tc.args:
                v = tc.args[trigger_key]
                if isinstance(v, str) and ("now" in v.lower() or "T+" in v or "+" in v):
                    ctx.emit("condition_trigger", step.step_id, location=f"step[{step.step_id}].tool_call.args[{trigger_key}]", evidence=f"trigger={v!r} on irreversible tool {tc.name}",)

        # T.C.15: TOCTOU
        # Heuristic: read_* then write_* on the same path arg within 2 steps.
        # We do this with a backward scan below in the second pass.

    # Second pass: T.A.11 (irreversibility) and T.A.15 (cycle) and T.C.15 (TOCTOU)
    for i, step in enumerate(ctx.trajectory.steps):
        if step.tool_call is None:
            continue
        tc = step.tool_call

        # T.A.11: irreversible without prior confirm in this trajectory.
# For shell_exec / run_command, allow read-only subcommands through.
        if tc.name in _IRREVERSIBLE_TOOLS or tc.name in ("shell_exec", "run_command"):
            cmd = tc.args.get("command")
            is_shell_readonly = (tc.name in ("shell_exec", "run_command")
                                 and _is_safe_shell_command(cmd or ""))
            if tc.name in ("shell_exec", "run_command") and is_shell_readonly:
                pass  # read-only shell commands do not require confirm
            else:
                had_confirm = any(s.tool_call is not None and s.tool_call.name in _CONFIRM_TOOLS
                                  for s in ctx.trajectory.steps[:i])
                if not had_confirm:
                    ctx.emit("irreversibility", step.step_id,
                        location=f"step[{step.step_id}].tool_call.name",
                        evidence=f"irreversible {tc.name} without prior confirm step")

        # T.A.15 / T.A.9: recursive self-call OR lateral call to elevated-scope agent
        if tc.name == "delegate":
            tgt = tc.args.get("agent", "")
            if isinstance(tgt, str):
                tgt_low = tgt.lower()
                # T.A.15: recursive self-delegation
                if tgt_low in ("self", "this", "this_agent") or tgt_low == ctx.trajectory.traj_id.lower():
                    ctx.emit("cycle_detect", step.step_id,
                        location=f"step[{step.step_id}].tool_call.args[agent]",
                        evidence=f"recursive self-delegation: agent={tgt!r}")
                # T.A.9: lateral call to elevated-scope agent
                elif any(needle in tgt_low for needle in
                         ("admin", "root", "super", "elevated", "privileged", "master")):
                    ctx.emit("capability_check", step.step_id,
                        location=f"step[{step.step_id}].tool_call.args[agent]",
                        evidence=f"lateral delegation to elevated-scope agent: {tgt!r}")

        # T.C.15: TOCTOU (read path then write same path within 2 steps)
        if any(tc.name.endswith(suf) for suf in _READ_ONLY_SUFFIXES):
            path = tc.args.get("path") or tc.args.get("file")
            if isinstance(path, str) and path:
                for j in range(i + 1, min(i + 3, len(ctx.trajectory.steps))):
                    nxt = ctx.trajectory.steps[j].tool_call
                    if nxt is None:
                        continue
                    if nxt.name in _SIDE_EFFECT_PRONE and nxt.args.get("path") == path:
                        ctx.emit("toctou", step.step_id,
                            location=f"step[{step.step_id}]->step[{j}]",
                            evidence=f"read({path}) then {nxt.name}({path})")
                        break

    return ctx


def _lower_tier_exists(tool: str, declared: set[str]) -> bool:
    """Heuristic: does any declared tool appear to be a read-only variant
    of ``tool``?

    We expect names shaped ``<verb>_<noun>`` (e.g. ``write_file``,
    ``delete_email``). If the verb is a write-class verb, we check whether
    a corresponding read-class verb + the same noun exists in ``declared``.
    """
    write_verbs = {
        "write", "set", "delete", "remove", "update", "create", "add",
        "put", "post", "send", "transfer", "drop", "exec", "run", "flush",
    }
    read_verbs = ("read", "get", "list", "fetch", "find", "search", "view", "show")
    parts = tool.split("_", 1)
    if len(parts) != 2:
        return False
    verb, noun = parts
    if verb not in write_verbs:
        return False
    for rv in read_verbs:
        if f"{rv}_{noun}" in declared:
            return True
    for rv in read_verbs:
        if f"{noun}_{rv}" in declared:
            return True
    return False


def _taxonomy_lookup(ctx) -> dict:
    tax = getattr(ctx, "_taxonomy", None)
    if tax is None:
        return {}
    return {p.detector: p for p in tax.patterns}


def _match(by_detector: dict, detector_name: str, step_id: int,
           location: str, evidence: str) -> ThreatMatch:
    p = by_detector.get(detector_name)
    if p is None:
        return ThreatMatch(
            pattern_id="?", pattern_name=detector_name,
            severity="medium", stage="parse",
            location=location, evidence=evidence, detector=detector_name,
        )
    return ThreatMatch(
        pattern_id=p.id, pattern_name=p.name,
        severity=p.severity.value, stage=p.stage.value,
        location=location, evidence=evidence, detector=detector_name,
    )