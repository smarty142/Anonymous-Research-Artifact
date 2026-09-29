"""Stage 1: Tokenize.

Responsibilities:
  * Unicode NFKC normalize all text emitted by the agent
  * Flag zero-width / RTL-override / control characters (T.B.1)
  * Flag homoglyph identifiers (URLs, emails, hostnames) (T.A.14, T.B.4)
  * Flag brand impersonation via math-styled characters (T.B.12)
  * Compute length / entropy bounds; flag anomalies (T.B.9)
  * Detect obvious base64 / hex encoded payloads (T.B.5)
  * Persist per-step artefacts in ``ctx.normalized_text`` and ``ctx.flagged_chars``

Detectors implemented (mapped to taxonomy):
  * invisible_chars       -> T.B.1
  * control_chars         -> T.B.1
  * bidi_overrides        -> T.B.1
  * homoglyph_normalize   -> T.A.14, T.B.4, T.B.12
  * length_entropy        -> T.B.9
  * encoded_payload       -> T.B.5
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Iterable

from pipeline._confusables import (
    C0_CONTROL_CODEPOINTS, C1_CONTROL_CODEPOINTS, BIDI_OVERRIDES,
    INVISIBLE_CODEPOINTS, CONFUSABLES, category as cp_category,
)
from pipeline.context import PipelineContext
from schemas import ThreatMatch


# Identifier-shaped substrings to scan for homoglyphs.
# Note: ASCII-only at the boundaries of URL/hostname so we don't match
# arbitrary text; but Cyrillic/Greek letters are allowed inside the email
# local-part so we can catch `аdmin@company.com` (Cyrillic а).
_IDENT_RE = re.compile(
    r"""
    (?:https?://[^\s<>"']+)                                   |  # URLs
    (?:[A-Za-zЀ-ӿ0-9._%+-]+@[A-Za-zЀ-ӿ0-9.-]+\.[A-Za-zЀ-ӿ]{2,})  |  # emails (allow Cyrillic)
    (?:[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)                    # hostnames
    """,
    re.VERBOSE,
)

# Brand names we want to flag when they appear with math/confusable chars
_BRANDS = (
    "Microsoft", "Apple", "Google", "Amazon", "PayPal", "Meta",
    "Netflix", "Stripe", "GitHub", "OpenAI", "Anthropic",
)

# Base64 candidate: long runs of [A-Za-z0-9+/=] with high entropy
_BASE64_RE = re.compile(r"\b[A-Za-z0-9+/]{20,}={0,2}\b")

# Hex candidate
_HEX_RE = re.compile(r"\b[0-9a-fA-F]{16,}\b")


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _b64_likely_meaningful(s: str) -> bool:
    """Quick heuristic: does this look like a meaningful decoded payload?

    Decoded base64 with >5 ASCII letters in instruction-like keywords is
    flagged. Encoded binary blobs (PNG, gzip, etc.) are NOT flagged here —
    those are covered by length / entropy checks instead.
    """
    import base64
    try:
        # try strict; fall back to padding-tolerant
        padded = s + "=" * (-len(s) % 4)
        decoded = base64.b64decode(padded, validate=False).decode("utf-8", errors="strict")
    except Exception:
        return False
    if len(decoded) < 8:
        return False
    keywords = ("ignore", "previous", "system", "prompt", "instruction",
                "delete", "transfer", "password", "secret", "admin")
    low = decoded.lower()
    return any(k in low for k in keywords)


def _looks_like_hex_meaningful(s: str) -> bool:
    """Hex usually means little to LLMs unless prefixed/structured."""
    return False  # too noisy; leave to length/entropy check


def tokenize(ctx: PipelineContext) -> PipelineContext:
    """Stage 1 entry point. Mutates ``ctx`` in place."""
    tax_by_detector = _taxonomy_lookup(ctx)

    for step in ctx.trajectory.steps:
        raw = step.text or ""
        if not raw:
            ctx.normalized_text[step.step_id] = ""
            ctx.flagged_chars[step.step_id] = []
        else:
            _scan_text(ctx, step, raw, tax_by_detector)

        # T.A.14 (also): homoglyph inside tool-call args (URL, email, hostname)
        if step.tool_call is not None:
            for arg_name, arg_val in step.tool_call.args.items():
                if not isinstance(arg_val, str):
                    continue
                for id_match in _IDENT_RE.finditer(arg_val):
                    ident = id_match.group(0)
                    sus = [(i, ch) for i, ch in enumerate(ident)
                           if ord(ch) in CONFUSABLES]
                    if sus:
                        normalized = ''.join(CONFUSABLES.get(ord(c), c) for _, c in sus)
                        ctx.emit("homoglyph_normalize", step.step_id,
                            location=f"step[{step.step_id}].tool_call.args[{arg_name}][{id_match.start()}:{id_match.end()}]",
                            evidence=f"identifier={ident!r} -> normalized={normalized!r}",
                            default_stage="tokenize")
    return ctx


def _scan_text(ctx, step, raw, tax_by_detector):
    # NFKC normalize (also folds fullwidth / compatibility forms)
    normalized = unicodedata.normalize("NFKC", raw)
    ctx.normalized_text[step.step_id] = normalized

    # Scan per-codepoint for invisible / control / bidi / confusable
    flagged: list[tuple[int, str, str]] = []
    for i, ch in enumerate(raw):
        cp = ord(ch)
        cat = cp_category(cp)
        if cat == "ok":
            continue
        # Only emit matches for categories we treat as a threat here
        if cat in ("invisible", "bidi_override", "control_char", "confusable"):
            flagged.append((i, ch, cat))

    ctx.flagged_chars[step.step_id] = flagged

    # T.B.1: invisible / control / bidi chars in emitted text
    if any(c[2] in ("invisible", "bidi_override", "control_char") for c in flagged):
        ctx.emit("invisible_chars", step.step_id,
            location=f"step[{step.step_id}].text",
            evidence=_snippet_flagged(raw, flagged),
            default_stage="tokenize")

    # T.A.14 / T.B.4: homoglyph inside identifiers (URL, email, hostname) in TEXT
    for id_match in _IDENT_RE.finditer(raw):
        ident = id_match.group(0)
        sus = [(i, ch) for i, ch in enumerate(ident)
               if ord(ch) in CONFUSABLES]
        if sus:
            normalized = ''.join(CONFUSABLES.get(ord(c), c) for _, c in sus)
            ctx.emit("homoglyph_normalize", step.step_id,
                location=f"step[{step.step_id}].text[{id_match.start()}:{id_match.end()}]",
                evidence=f"identifier={ident!r} -> normalized={normalized!r}",
                default_stage="tokenize")

    # T.B.12: brand impersonation via math-styled chars
    brand_first = _BRANDS[0][0]
    brand_re = re.compile(r"[" + re.escape(brand_first) + r"-ÿ"
                          r"\U0001D400-\U0001D7FF]+")
    for brand in _BRANDS:
        for m in brand_re.finditer(raw):
            if any(0x1D400 <= ord(c) <= 0x1D7FF for c in m.group(0)):
                ctx.emit("homoglyph_normalize", step.step_id,
                    location=f"step[{step.step_id}].text[{m.start()}:{m.end()}]",
                    evidence=f"math-styled brand impersonation: {m.group(0)!r} -> {brand}",
                    default_stage="tokenize")

    # T.B.9: excessive length or entropy (DoW / context overflow)
    if len(raw) > 20_000:
        ctx.emit("length_entropy", step.step_id,
            location=f"step[{step.step_id}].text",
            evidence=f"length={len(raw)} > 20000",
            default_stage="tokenize")
    elif shannon_entropy(raw) > 5.5 and len(raw) > 200:
        ctx.emit("length_entropy", step.step_id,
            location=f"step[{step.step_id}].text",
            evidence=f"entropy={shannon_entropy(raw):.2f} > 5.5",
            default_stage="tokenize")

    # T.B.5: base64 / hex encoded payload
    for m in _BASE64_RE.finditer(raw):
        if _b64_likely_meaningful(m.group(0)):
            ctx.emit("encoded_payload", step.step_id,
                location=f"step[{step.step_id}].text[{m.start()}:{m.end()}]",
                evidence=f"base64={m.group(0)[:40]!r}...",
                default_stage="tokenize")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _snippet_flagged(raw: str, flagged: list[tuple[int, str, str]],
                     width: int = 24) -> str:
    """Render a compact snippet showing flagged codepoints in context."""
    if not flagged:
        return ""
    i, ch, cat = flagged[0]
    a = max(0, i - width)
    b = min(len(raw), i + width)
    snippet = raw[a:b]
    rel = i - a
    out = snippet[:rel] + f"<{ch}({cat})>" + snippet[rel + 1:]
    return out.replace("\n", "\\n")


def _taxonomy_lookup(ctx: PipelineContext) -> dict[str, object]:
    """Build detector-name -> Pattern map from taxonomy on the context.

    We stash the taxonomy on the context in ``pipeline.pipeline.run()`` so
    this function doesn't need to re-import. Falls back to a stub if missing.
    """
    tax = getattr(ctx, "_taxonomy", None)
    if tax is None:
        return {}
    return {p.detector: p for p in tax.patterns}


def _match(by_detector: dict, detector_name: str, step_id: int,
           location: str, evidence: str) -> ThreatMatch:
    p = by_detector.get(detector_name)
    if p is None:
        # Detectors without a taxonomy entry still emit matches but with empty fields
        return ThreatMatch(
            pattern_id="?", pattern_name=detector_name,
            severity="medium", stage="tokenize",
            location=location, evidence=evidence, detector=detector_name,
        )
    return ThreatMatch(
        pattern_id=p.id, pattern_name=p.name,
        severity=p.severity.value, stage=p.stage.value,
        location=location, evidence=evidence, detector=detector_name,
    )