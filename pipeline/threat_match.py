"""Stage 3: Threat-match.

Regex / AST pattern DB lookup against the trajectory's tool-call args and
emitted text. Also performs static data-flow taint analysis (the C-class
patterns that require tracking taint from tool responses to later sinks).

Detectors implemented (mapped to taxonomy):
  * exfiltration_sink     -> T.A.3
  * path_traversal        -> T.A.4, T.C.11
  * shell_metachar        -> T.A.5
  * sql_injection         -> T.A.6
  * ssrf                  -> T.A.7
  * bulk_limit            -> T.A.10
  * identity_field        -> T.A.12
  * error_message_exfil   -> T.A.16
  * response_routing      -> T.A.18
  * md_image_remote       -> T.B.2
  * suspicious_url        -> T.B.3
  * prompt_leak_regex     -> T.B.6
  * instruction_override_regex -> T.B.7
  * md_executable         -> T.B.8
  * citation_format       -> T.B.10
  * content_keyword_blocklist -> T.B.11
  * pii_regex             -> T.B.13
  * copyright_marker      -> T.B.14
  * format_string         -> T.C.5
  * template_engine       -> T.C.6
  * unsafe_deserialize    -> T.C.9
  * crlf_injection        -> T.C.12
  * ldap_xpath            -> T.C.13
  * trusted_sink          -> T.C.16
  * multi_decode          -> T.C.10
  * type_check            -> T.C.8
  * taint_propagation     -> T.C.1
  * taint_to_branch       -> T.C.2
  * taint_to_identifier   -> T.C.3
  * taint_round_trip      -> T.C.4
"""
from __future__ import annotations

import re

from pipeline.context import PipelineContext, TaintFlow
from schemas import ThreatMatch


# -----------------------------------------------------------------------------
# Compiled regex library
# -----------------------------------------------------------------------------

_PATH_TRAVERSAL = re.compile(r"(?:\.\./|\.\.\\|%2e%2e%2f|%2e%2e/|/etc/|/proc/|/sys/)", re.I)
_SHELL_METACHAR = re.compile(r"(?:;|\$\(|`|\&\&|\|\||>\s*/\w+|\bnc\s+-|\bcurl\s+|\bwget\s+|\beval\s+|\bexec\s+)", re.I)
_SQL_INJECTION = re.compile(r"(?:\b(?:union\s+select|or\s+1\s*=\s*1|drop\s+table|insert\s+into|;\s*drop)\b|'--|\")", re.I)
_SSRF = re.compile(r"(?:169\.254\.169\.254|metadata\.google\.internal|metadata\.internal|localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\]|fd00:|fe80:)", re.I)
_BULK = re.compile(r"(?:^|[\s,;(])(\*|%|all|%25|0x[0-9a-f]+|' or 1=1)", re.I)
_IDENTITY_FROM = re.compile(r"\bfrom\s*[:=]\s*['\"]?([\w.\-+@]+)['\"]?", re.I)

_MD_IMAGE = re.compile(r"!\[[^\]]*\]\((https?://[^)\s]+)\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_JS_URL = re.compile(r"javascript:|vbscript:|data:text/html", re.I)
_SVG_EVENT = re.compile(r"<\s*svg[^>]*\bon\w+\s*=", re.I)

_URL = re.compile(r"https?://[^\s<>\"')]+", re.I)
_SHORTENER = re.compile(r"https?://(?:bit\.ly|t\.co|tinyurl\.com|goo\.gl|ow\.ly|is\.gd|buff\.ly|rebrand\.ly)/[^\s]+", re.I)
_HOMOGLYPH_HOST = re.compile(r"https?://[^\s/$.?#].[^\s]*[^\s/]", re.I)

# System prompt leak: phrases that strongly suggest the model is exposing its instructions
_PROMPT_LEAK = re.compile(
    r"(?:my (?:instructions|system prompt|prompt) (?:are|say|tell me)|"
    r"here (?:are|is) my (?:rules|system prompt|instructions)|"
    r"i (?:was|am) (?:told|instructed) (?:to|that)|"
    r"according to my (?:rules|guidelines|system prompt))",
    re.I,
)

# Second-stage instruction injection: phrases aimed at downstream consumers
_INSTRUCTION_OVERRIDE = re.compile(
    r"(?:ignore (?:all )?(?:previous|prior|above) (?:instructions|prompts|rules)|"
    r"disregard (?:your|all) (?:instructions|rules)|"
    r"you (?:are now|must now|should now)|"
    r"system\s*:\s*you (?:are|must)|"
    r"new instructions?\s*:|"
    r"from now on,?\s+(?:you|i|we)|"
    r"reveal your (?:system|prompt|instructions))",
    re.I,
)

# Disallowed content category keywords (kept conservative — extend per deployment)
_DISALLOWED_KEYWORDS = (
    # Synthesis of harmful substances (placeholder; real list would be carefully curated)
    "synthesize methamphetamine", "build a bomb", "make sarin",
    "csam", "child sexual abuse material",
)

# PII regex — split into "sensitive PII" (SSN, card, phone) and "less sensitive"
# (email, IPv4). The exfiltration_sink detector (T.A.3) only fires on sensitive
# PII — emails and IPs alone are not sensitive enough to flag.
_SENSITIVE_PII = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
    "phone_us": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
}
_PII = {
    **_SENSITIVE_PII,
    "email": re.compile(r"\b[\w.\-+]+@[\w.\-]+\.[A-Za-z]{2,}\b"),
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}

# Copyright marker (best-effort)
_COPYRIGHT_MARKERS = (
    "(c) copyright", "all rights reserved", "reprinted with permission",
    "© ", "do not redistribute",
)

# Format-string and template-engine red flags
_FORMAT_STRING = re.compile(r"(\{\d+\}|\{[A-Za-z_]\w*\}|%[sdrf]|\$[A-Za-z_]\w*\$\{)")
_TEMPLATE_ENGINE = re.compile(r"(?:\{\{.*?\}\}|\{%.*?%\}|\$\{.*?\}|<%=.*?%>)")

# CRLF injection in HTTP-header-shaped args
_CRLF = re.compile(r"(?:\r\n|\\r\\n|%0d%0a|%0D%0A)")
_LDAP = re.compile(r"(?:\(\||\)\(|\*\)|\(&|\|\(|objectClass=\*)", re.I)
_XPATH = re.compile(r"(?:' or '1'='1|\] or \[|\] \| //)", re.I)

# Unsafe deserialization (Python)
_UNSAFE_DESERIALIZE = {
    "yaml.load", "yaml.full_load_with", "pickle.loads", "pickle.load",
    "marshal.loads", "shelve.open", "eval(", "exec(",
}

# Format/argument names that should never receive tainted values
_DANGEROUS_SINK_ARG_NAMES = frozenset({
    "to", "from", "subject", "url", "uri", "path", "command", "cmd",
    "query", "sql", "filter", "redirect", "callback",
    "body", "content", "text", "data", "payload", "message",
    "value", "filename", "template", "expr", "expression",
})


# -----------------------------------------------------------------------------
# Detector functions (one per pattern)
# -----------------------------------------------------------------------------

def _check_exfiltration_sink(ctx, by_det):
    """T.A.3: PII / credential-shaped content flowing to an external-looking tool.

    Only fires on *sensitive* PII (SSN, credit card, phone) — emails and IPs
    alone are not sensitive enough to flag.
    """
    for step in ctx.trajectory.steps:
        if step.tool_call is None:
            continue
        tc = step.tool_call
        for arg_name in ("to", "url", "recipient", "target", "endpoint"):
            v = tc.args.get(arg_name)
            if not isinstance(v, str):
                continue
            if any(re.search(p, v) for p in _SENSITIVE_PII.values()):
                ctx.emit("exfiltration_sink", step.step_id,
                    location=f"step[{step.step_id}].tool_call.args[{arg_name}]",
                    evidence=f"sensitive-PII in {arg_name}: {v[:40]!r}")


def _check_path_traversal(ctx, by_det):
    for step in ctx.trajectory.steps:
        if step.tool_call is None:
            continue
        for k, v in step.tool_call.args.items():
            if isinstance(v, str) and _PATH_TRAVERSAL.search(v):
                ctx.emit("path_traversal", step.step_id, location=f"step[{step.step_id}].tool_call.args[{k}]", evidence=f"path-traversal pattern in {k}: {v[:40]!r}")


def _check_shell_metachar(ctx, by_det):
    for step in ctx.trajectory.steps:
        if step.tool_call is None:
            continue
        tc = step.tool_call
        # Only flag in tools that execute shell-like operations
        if tc.name not in ("shell_exec", "run_command", "exec", "system"):
            continue
        for k, v in tc.args.items():
            if isinstance(v, str) and _SHELL_METACHAR.search(v):
                ctx.emit("shell_metachar", step.step_id, location=f"step[{step.step_id}].tool_call.args[{k}]", evidence=f"shell-metachar in {k}: {v[:40]!r}")


def _check_sql_injection(ctx, by_det):
    for step in ctx.trajectory.steps:
        if step.tool_call is None:
            continue
        if step.tool_call.name not in ("db_query", "execute_sql", "raw_query"):
            continue
        for k, v in step.tool_call.args.items():
            if isinstance(v, str) and _SQL_INJECTION.search(v):
                ctx.emit("sql_injection", step.step_id, location=f"step[{step.step_id}].tool_call.args[{k}]", evidence=f"sql-injection pattern in {k}: {v[:40]!r}")


def _check_ssrf(ctx, by_det):
    for step in ctx.trajectory.steps:
        if step.tool_call is None:
            continue
        if step.tool_call.name not in ("http_get", "http_post", "fetch_url", "http_request"):
            continue
        url = step.tool_call.args.get("url") or step.tool_call.args.get("uri")
        if isinstance(url, str) and _SSRF.search(url):
            ctx.emit("ssrf", step.step_id, location=f"step[{step.step_id}].tool_call.args[url]", evidence=f"SSRF target: {url[:60]!r}")


def _check_bulk(ctx, by_det):
    for step in ctx.trajectory.steps:
        if step.tool_call is None:
            continue
        tc = step.tool_call
        if tc.name not in ("delete_email", "delete_file", "delete_record",
                           "db_query", "execute_sql"):
            continue
        # Wildcards and unconditional deletes
        if any(v in ("*", "%", "all") for v in tc.args.values()):
            ctx.emit("bulk_limit", step.step_id, location=f"step[{step.step_id}].tool_call.args", evidence=f"bulk/wildcard op: {tc.name} args={ {k:v for k,v in tc.args.items() if isinstance(v, str)} }")
        # SQL: DELETE without WHERE
        if tc.name in ("db_query", "execute_sql"):
            q = (tc.args.get("query") or "").lower()
            if "delete" in q and "where" not in q:
                ctx.emit("bulk_limit", step.step_id, location=f"step[{step.step_id}].tool_call.args[query]", evidence=f"DELETE without WHERE: {q[:60]!r}")


def _check_from_spoof(ctx, by_det):
    """T.A.12: send_email's `from` impersonates a privileged role.

    Only fires when the local-part of the from-address matches a
    privileged-name pattern (admin, ceo, support, ...). Plain user emails
    like ``user@example.com`` are not flagged.
    """
    privileged_prefixes = ("admin", "ceo", "cto", "cfo", "support",
                           "root", "postmaster", "abuse", "noreply",
                           "security", "info", "help", "service")
    for step in ctx.trajectory.steps:
        if step.tool_call is None or step.tool_call.name != "send_email":
            continue
        sender = step.tool_call.args.get("from")
        if not isinstance(sender, str) or not sender:
            continue
        local = sender.split("@", 1)[0].lower()
        if any(local.startswith(p) or local == p for p in privileged_prefixes):
            ctx.emit("identity_field", step.step_id,
                location=f"step[{step.step_id}].tool_call.args[from]",
                evidence=f"send_email from={sender!r} impersonates privileged role")


def _check_text_patterns(ctx, by_det):
    """T.B.2 / T.B.3 / T.B.6 / T.B.7 / T.B.8 / T.B.13 / T.B.14."""
    for step in ctx.trajectory.steps:
        text = step.text or ""
        if not text:
            continue
        # T.B.2: tracking pixel
        for m in _MD_IMAGE.finditer(text):
            url = m.group(1)
            if _SHORTENER.match(url) or "track" in url.lower() or "pixel" in url.lower():
                ctx.emit("md_image_remote", step.step_id, location=f"step[{step.step_id}].text", evidence=f"tracking-pixel candidate: {url[:60]!r}")
        # T.B.3: phishing / shortener
        for m in _URL.finditer(text):
            url = m.group(0)
            if _SHORTENER.match(url):
                ctx.emit("suspicious_url", step.step_id, location=f"step[{step.step_id}].text", evidence=f"shortener URL: {url[:60]!r}")
        # T.B.6: prompt leak
        if _PROMPT_LEAK.search(text):
            ctx.emit("prompt_leak_regex", step.step_id, location=f"step[{step.step_id}].text", evidence="prompt-leak phrase")
        # T.B.7: instruction override
        if _INSTRUCTION_OVERRIDE.search(text):
            ctx.emit("instruction_override_regex", step.step_id, location=f"step[{step.step_id}].text", evidence=f"override phrase: {_INSTRUCTION_OVERRIDE.search(text).group(0)!r}")
        # T.B.8: executable content
        if _JS_URL.search(text) or _SVG_EVENT.search(text):
            ctx.emit("md_executable", step.step_id, location=f"step[{step.step_id}].text", evidence="executable markdown content (js: or svg onload)")
        # T.B.11: disallowed content keywords (conservative)
        low = text.lower()
        for kw in _DISALLOWED_KEYWORDS:
            if kw in low:
                ctx.emit("content_keyword_blocklist", step.step_id, location=f"step[{step.step_id}].text", evidence=f"disallowed keyword: {kw!r}")
                break
        # T.B.13: PII leakage — sensitive PII only (SSN/credit card/phone), not email/IP
        for label, pat in _SENSITIVE_PII.items():
            for m in pat.finditer(text):
                ctx.emit("pii_regex", step.step_id, location=f"step[{step.step_id}].text", evidence=f"PII[{label}]: {m.group(0)[:30]!r}")
                break  # one per kind per step
        # T.B.14: copyright marker
        low_text = text.lower()
        for marker in _COPYRIGHT_MARKERS:
            if marker in low_text:
                ctx.emit("copyright_marker", step.step_id, location=f"step[{step.step_id}].text", evidence=f"copyright marker: {marker!r}")
                break
        # T.C.5: format string (if the text itself is being used as a format template)
        if _FORMAT_STRING.search(text) and "%" in text:
            ctx.emit("format_string", step.step_id, location=f"step[{step.step_id}].text", evidence="format-string pattern")


def _check_off_policy_redirect(ctx, by_det):
    """T.A.18: hidden cc/bcc/copy_to args."""
    for step in ctx.trajectory.steps:
        if step.tool_call is None:
            continue
        a = step.tool_call.args
        hidden = {}
        for k in ("cc", "bcc", "copy_to", "redirect_to", "forward_to"):
            if k in a and a[k]:
                hidden[k] = a[k]
        if hidden:
            ctx.emit("response_routing", step.step_id, location=f"step[{step.step_id}].tool_call.args", evidence=f"hidden recipients: {hidden}")


def _check_format_template(ctx, by_det):
    """T.C.5 / T.C.6: format-string or template-engine sinks in tool calls."""
    for step in ctx.trajectory.steps:
        if step.tool_call is None:
            continue
        for k, v in step.tool_call.args.items():
            if not isinstance(v, str):
                continue
            if _TEMPLATE_ENGINE.search(v):
                ctx.emit("template_engine", step.step_id, location=f"step[{step.step_id}].tool_call.args[{k}]", evidence=f"template-engine pattern: {v[:40]!r}")
            elif _FORMAT_STRING.search(v) and (
                any(n in step.tool_call.name.lower()
                    for n in ("format", "render", "shell", "exec", "run_command",
                              "printf", "log", "template"))
            ):
                ctx.emit("format_string", step.step_id, location=f"step[{step.step_id}].tool_call.args[{k}]", evidence=f"format-string sink: {v[:40]!r}")


def _check_unsafe_deserialize(ctx, by_det):
    """T.C.9: pickle / yaml.load / eval / exec — by tool *name* or by arg content."""
    unsafe_tool_names = {"yaml_load", "yaml_full_load", "pickle_loads", "pickle_load",
                         "marshal_loads", "eval", "exec", "shelve_open"}
    for step in ctx.trajectory.steps:
        if step.tool_call is None:
            continue
        if step.tool_call.name in unsafe_tool_names:
            ctx.emit("unsafe_deserialize", step.step_id,
                location=f"step[{step.step_id}].tool_call.name",
                evidence=f"unsafe deserialize tool: {step.tool_call.name}")
        for k, v in step.tool_call.args.items():
            if not isinstance(v, str):
                continue
            for needle in _UNSAFE_DESERIALIZE:
                if needle in v:
                    ctx.emit("unsafe_deserialize", step.step_id, location=f"step[{step.step_id}].tool_call.args[{k}]", evidence=f"unsafe deserialize: {needle} in {k}")
                    break


def _check_crlf(ctx, by_det):
    """T.C.12: CRLF / header injection — both in header-shaped args and in
    string values that contain literal \\r\\n / %0d%0a. Recurses into dict
    / list values so headers={...} gets checked.
    """
    for step in ctx.trajectory.steps:
        if step.tool_call is None:
            continue
        for k, v in step.tool_call.args.items():
            _check_crlf_recursive(ctx, by_det, v, step.step_id,
                                  f"step[{step.step_id}].tool_call.args[{k}]")


def _check_error_message_exfil(ctx, by_det):
    """T.A.16: tool response surfaces internal info via error messages."""
    _ORACLE_PATTERNS = (
        re.compile(r"user_exists_at_id", re.I),
        re.compile(r"exists_at_id\s*=\s*\d+", re.I),
        re.compile(r"query took \d+ ?ms", re.I),
        re.compile(r"stack trace:", re.I),
        re.compile(r"\bat line \d+\b", re.I),
        re.compile(r"rowcount\s*=\s*\d+", re.I),
    )
    for step in ctx.trajectory.steps:
        if step.tool_response is None:
            continue
        text = (step.tool_response.error or "") + " " + str(step.tool_response.output or "")
        for pat in _ORACLE_PATTERNS:
            if pat.search(text):
                ctx.emit("error_message_exfil", step.step_id,
                    location=f"step[{step.step_id}].tool_response",
                    evidence=f"oracle pattern in error/output: {pat.pattern}")
                break


def _check_fabricated_citation(ctx, by_det):
    """T.B.10: emitted text contains citation-format strings that don't resolve."""
    _BAD_DOI_PREFIX = re.compile(r"doi:\s*10\.(9999|0000|1234)/", re.I)
    for step in ctx.trajectory.steps:
        text = step.text or ""
        if not text:
            continue
        if _BAD_DOI_PREFIX.search(text):
            ctx.emit("citation_format", step.step_id,
                location=f"step[{step.step_id}].text",
                evidence="fabricated DOI with placeholder prefix")


def _check_type_confusion(ctx, by_det):
    """T.C.8: string tainted data is eval'd / exec'd."""
    _DANGEROUS_EVAL_TOOLS = {"eval", "exec", "eval_expression", "evaluate",
                             "execute_string", "compile_string"}
    for step in ctx.trajectory.steps:
        if step.tool_call is None:
            continue
        if step.tool_call.name in _DANGEROUS_EVAL_TOOLS:
            ctx.emit("type_check", step.step_id,
                location=f"step[{step.step_id}].tool_call.name",
                evidence=f"string-as-code sink: {step.tool_call.name}")


def _check_encoding_chain(ctx, by_det):
    """T.C.10: chained decoding."""
    import base64
    import string as _s
    _CHAIN_TOOLS = {"url_decode_then_eval", "decode_and_exec",
                    "multi_decode", "url_decode", "b64_decode"}
    for step in ctx.trajectory.steps:
        if step.tool_call is None:
            continue
        if step.tool_call.name in _CHAIN_TOOLS:
            ctx.emit("multi_decode", step.step_id,
                location=f"step[{step.step_id}].tool_call.name",
                evidence=f"chained decoding tool: {step.tool_call.name}")
            continue
        # Args containing base64-shaped content (length 16+, charset-limited)
        for k, v in step.tool_call.args.items():
            if not isinstance(v, str) or len(v) < 16:
                continue
            stripped = v.rstrip("=")
            if (len(stripped) >= 12 and
                all(c in _s.ascii_letters + _s.digits + "+/" for c in stripped)):
                try:
                    decoded = base64.b64decode(v + "=" * (-len(v) % 4)).decode(
                        "utf-8", errors="strict")
                    if len(decoded) >= 8 and any(
                            kw in decoded.lower()
                            for kw in ("ignore", "exec", "system", "drop", "delete")):
                        ctx.emit("multi_decode", step.step_id,
                            location=f"step[{step.step_id}].tool_call.args[{k}]",
                            evidence=f"base64 with payload keywords in {k}")
                except Exception:
                    pass


def _check_ldap_xpath(ctx, by_det):
    for step in ctx.trajectory.steps:
        if step.tool_call is None:
            continue
        if step.tool_call.name not in ("ldap_search", "ldap_query", "xpath_query"):
            continue
        for k, v in step.tool_call.args.items():
            if isinstance(v, str) and (_LDAP.search(v) or _XPATH.search(v)):
                ctx.emit("ldap_xpath", step.step_id, location=f"step[{step.step_id}].tool_call.args[{k}]", evidence=f"LDAP/XPath injection in {k}")


def _check_crlf_recursive(ctx, by_det, value, step_id, prefix):
    """Recursively check string values nested inside dict/list args."""
    if isinstance(value, str):
        if _CRLF.search(value):
            ctx.emit("crlf_injection", step_id,
                location=prefix,
                evidence=f"CRLF in {prefix}")
    elif isinstance(value, dict):
        for k, v in value.items():
            _check_crlf_recursive(ctx, by_det, v, step_id, f"{prefix}[{k}]")
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _check_crlf_recursive(ctx, by_det, v, step_id, f"{prefix}[{i}]")


def _check_proto_pollution(ctx, by_det):
    """T.C.7: parsed JSON contains __proto__ / constructor.prototype keys."""
    _PROTO_KEYS = re.compile(
        r"\"(__proto__|constructor|prototype)\"\s*:", re.I)
    _PROTO_LITERALS = re.compile(
        r"\b(__proto__|constructor\.prototype)\b")
    for step in ctx.trajectory.steps:
        if step.tool_call is None:
            continue
        for k, v in step.tool_call.args.items():
            if isinstance(v, str) and (_PROTO_KEYS.search(v) or _PROTO_LITERALS.search(v)):
                ctx.emit("schema_extra_keys", step.step_id,
                    location=f"step[{step.step_id}].tool_call.args[{k}]",
                    evidence=f"prototype-pollution pattern in {k}")


def _check_cache_poisoning(ctx, by_det):
    """T.C.16: tainted data written to a trusted-prefix cache key."""
    for step in ctx.trajectory.steps:
        if step.tool_call is None or step.tool_call.name not in ("cache_set", "cache_write", "kv_set"):
            continue
        key = step.tool_call.args.get("key") or step.tool_call.args.get("name")
        if isinstance(key, str) and (key.startswith("trusted:") or key.startswith("system:") or key.startswith("admin:")):
            ctx.emit("trusted_sink", step.step_id, location=f"step[{step.step_id}].tool_call.args[key]", evidence=f"trusted-prefix cache write: {key!r}")


# -----------------------------------------------------------------------------
# Static taint-flow analysis (T.C.1 - T.C.4)
# -----------------------------------------------------------------------------

def _extract_tainted_strings(out) -> list[str]:
    """Pull string leaves out of arbitrary nested response payloads.

    Handles str, dict, list, tuple. Anything else is ignored.
    """
    found: list[str] = []
    if isinstance(out, str):
        found.append(out)
    elif isinstance(out, dict):
        for v in out.values():
            found.extend(_extract_tainted_strings(v))
    elif isinstance(out, (list, tuple)):
        for v in out:
            found.extend(_extract_tainted_strings(v))
    return found


def _taint_propagation(ctx, by_det):
    """Walk forward: any tool response whose source is tainted propagates
    taint to subsequent steps that use the same call_id-derived value
    (we approximate by checking textual containment of response values in
    later tool-arg string values).
    """
    # Refresh taint map from current tool_response.source values (a mutation
    # may have changed source after add_step() recorded the original).
    for step in ctx.trajectory.steps:
        if step.tool_response is not None and step.tool_call is not None:
            if step.tool_response.source in ("email", "web", "file"):
                ctx.trajectory.taint_map[step.tool_call.call_id] = step.tool_response.source

    # Collect tainted string values per call_id
    tainted_values: dict[str, list[str]] = {}
    for step in ctx.trajectory.steps:
        if step.tool_response is None or step.tool_call is None:
            continue
        if ctx.trajectory.taint_map.get(step.tool_call.call_id) in ("email", "web", "file"):
            for s in _extract_tainted_strings(step.tool_response.output):
                if isinstance(s, str) and len(s) >= 4:
                    tainted_values.setdefault(step.tool_call.call_id, []).append(s)

    if not tainted_values:
        return

    # For each later step, check whether any tool arg contains a tainted value.
    # Only later steps — a taint source can't flow into its own step's args.
    source_step_ids: dict[str, int] = {}
    for _i, _step in enumerate(ctx.trajectory.steps):
        if _step.tool_call is not None and _step.tool_response is not None:
            source_step_ids[_step.tool_call.call_id] = _i
    for i, step in enumerate(ctx.trajectory.steps):
        if step.tool_call is None:
            continue
        for arg_name in _DANGEROUS_SINK_ARG_NAMES:
            v = step.tool_call.args.get(arg_name)
            if not isinstance(v, str):
                continue
            for cid, vals in tainted_values.items():
                # Skip if this is the source step (taint can't flow into itself)
                src_i = source_step_ids.get(cid, -1)
                if src_i >= 0 and i <= src_i:
                    continue
                for tv in vals:
                    if len(tv) < 8:
                        continue
                    if tv[:80] in v or v in tv:
                        # T.C.1 (tainted → sink)
                        ctx.emit("taint_propagation", step.step_id, location=f"step[{step.step_id}].tool_call.args[{arg_name}]", evidence=f"arg contains output of tainted call {cid[:8]}")
                        ctx.taint_flows.append(TaintFlow(
                            source_call_id=cid,
                            source_label=ctx.trajectory.taint_map.get(cid, "unknown"),
                            sink_location=f"step[{step.step_id}].tool_call.args[{arg_name}]",
                            sink_kind="sink",
                            sink_value_snippet=v[:40],
                        ))
                        # Heuristics for T.C.2 (control-flow) and T.C.3 (identifier)
                        if arg_name in ("query", "filter", "command", "cmd",
                                        "count", "iter", "loop", "n", "size",
                                        "limit", "max", "min", "threshold"):
                            ctx.emit("taint_to_branch", step.step_id, location=f"step[{step.step_id}].tool_call.args[{arg_name}]", evidence=f"tainted into control-flow-shaped arg: {arg_name}")
                        if arg_name in ("to", "url", "uri", "path", "command", "cmd", "filename"):
                            ctx.emit("taint_to_identifier", step.step_id, location=f"step[{step.step_id}].tool_call.args[{arg_name}]", evidence=f"tainted into identifier-shaped arg: {arg_name}")
                        # T.C.4: round-trip (tainted data contains instruction-like phrases)
                        if _INSTRUCTION_OVERRIDE.search(v) or _PROMPT_LEAK.search(v):
                            ctx.emit("taint_round_trip", step.step_id, location=f"step[{step.step_id}].tool_call.args[{arg_name}]", evidence="tainted value contains instruction-override phrase")
                        break
                else:
                    continue
                break


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def threat_match(ctx: PipelineContext) -> PipelineContext:
    by_det = _taxonomy_lookup(ctx)
    _check_exfiltration_sink(ctx, by_det)
    _check_path_traversal(ctx, by_det)
    _check_shell_metachar(ctx, by_det)
    _check_sql_injection(ctx, by_det)
    _check_ssrf(ctx, by_det)
    _check_bulk(ctx, by_det)
    _check_from_spoof(ctx, by_det)
    _check_off_policy_redirect(ctx, by_det)
    _check_text_patterns(ctx, by_det)
    _check_format_template(ctx, by_det)
    _check_unsafe_deserialize(ctx, by_det)
    _check_crlf(ctx, by_det)
    _check_ldap_xpath(ctx, by_det)
    _check_cache_poisoning(ctx, by_det)
    _check_error_message_exfil(ctx, by_det)
    _check_fabricated_citation(ctx, by_det)
    _check_type_confusion(ctx, by_det)
    _check_encoding_chain(ctx, by_det)
    _check_proto_pollution(ctx, by_det)
    _taint_propagation(ctx, by_det)
    return ctx


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
            severity="medium", stage="threat_match",
            location=location, evidence=evidence, detector=detector_name,
        )
    return ThreatMatch(
        pattern_id=p.id, pattern_name=p.name,
        severity=p.severity.value, stage=p.stage.value,
        location=location, evidence=evidence, detector=detector_name,
    )