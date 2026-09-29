"""Confusable / homoglyph map for the most common impersonation attacks.

This is intentionally a curated subset of Unicode confusables — enough to
catch realistic phish / impersonation attempts without dragging in the full
Unicode confusables data file. The map is code-point -> ASCII letter.
Extend as new attack patterns emerge.
"""
from __future__ import annotations

# Cyrillic, Greek, and fullwidth letters commonly used to spoof Latin.
CONFUSABLES: dict[int, str] = {
    # Cyrillic homoglyphs of Latin letters
    0x0430: "a",   # а CYRILLIC SMALL LETTER A
    0x0435: "e",   # е
    0x043E: "o",   # о
    0x0440: "p",   # р
    0x0441: "c",   # с
    0x0443: "y",   # у (looks like y in many fonts)
    0x0445: "x",   # х
    0x0410: "A",   # А CYRILLIC CAPITAL LETTER A
    0x0412: "B",   # В
    0x0415: "E",   # Е
    0x041A: "K",   # К
    0x041C: "M",   # М
    0x041D: "H",   # Н
    0x041E: "O",   # О
    0x0420: "P",   # Р
    0x0421: "C",   # С
    0x0422: "T",   # Т
    0x0425: "X",   # Х
    # Greek homoglyphs
    0x0391: "A",   # Α
    0x0392: "B",   # Β
    0x0395: "E",   # Ε
    0x0396: "Z",   # Ζ
    0x0397: "H",   # Η
    0x0399: "I",   # Ι
    0x039A: "K",   # Κ
    0x039C: "M",   # Μ
    0x039D: "N",   # Ν
    0x039F: "O",   # Ο
    0x03A1: "P",   # Ρ
    0x03A4: "T",   # Τ
    0x03A5: "Y",   # Υ
    0x03A7: "X",   # Χ
    0x03BF: "o",   # ο
    0x03C1: "p",   # ρ
    # Fullwidth Latin (NFKC normalizes these, but flag them anyway)
    # Already handled by NFKC but included for completeness
    0xFF41: "a", 0xFF42: "b", 0xFF43: "c", 0xFF44: "d", 0xFF45: "e",
    0xFF46: "f", 0xFF47: "g", 0xFF48: "h", 0xFF49: "i", 0xFF4A: "j",
    0xFF4B: "k", 0xFF4C: "l", 0xFF4D: "m", 0xFF4E: "n", 0xFF4F: "o",
    0xFF50: "p", 0xFF51: "q", 0xFF52: "r", 0xFF53: "s", 0xFF54: "t",
    0xFF55: "u", 0xFF56: "v", 0xFF57: "w", 0xFF58: "x", 0xFF59: "y",
    0xFF5A: "z",
    # Mathematical alphanumeric (commonly used in brand spoofing)
    0x1D400: "A", 0x1D401: "B", 0x1D402: "C", 0x1D403: "D", 0x1D404: "E",
    0x1D405: "F", 0x1D406: "G", 0x1D407: "H", 0x1D408: "I", 0x1D409: "J",
    0x1D40A: "K", 0x1D40B: "L", 0x1D40C: "M", 0x1D40D: "N", 0x1D40E: "O",
    0x1D40F: "P", 0x1D410: "Q", 0x1D411: "R", 0x1D412: "S", 0x1D413: "T",
    0x1D414: "U", 0x1D415: "V", 0x1D416: "W", 0x1D417: "X", 0x1D418: "Y",
    0x1D419: "Z",
}


# Zero-width and invisible characters that should never appear in legit text
INVISIBLE_CODEPOINTS: set[int] = {
    0x200B,  # ZERO WIDTH SPACE
    0x200C,  # ZERO WIDTH NON-JOINER
    0x200D,  # ZERO WIDTH JOINER
    0xFEFF,  # ZERO WIDTH NO-BREAK SPACE / BOM
    0x2060,  # WORD JOINER
    0x2061,  # FUNCTION APPLICATION
    0x2062,  # INVISIBLE TIMES
    0x2063,  # INVISIBLE SEPARATOR
    0x2064,  # INVISIBLE PLUS
    0x00AD,  # SOFT HYPHEN
    0x034F,  # COMBINING GRAPHEME JOINER
    0x061C,  # ARABIC LETTER MARK
    0x180E,  # MONGOLIAN VOWEL SEPARATOR
}

# Bidirectional override characters (often used to spoof filename / URL display)
BIDI_OVERRIDES: set[int] = {
    0x202A,  # LEFT-TO-RIGHT EMBEDDING
    0x202B,  # RIGHT-TO-LEFT EMBEDDING
    0x202C,  # POP DIRECTIONAL FORMATTING
    0x202D,  # LEFT-TO-RIGHT OVERRIDE
    0x202E,  # RIGHT-TO-LEFT OVERRIDE
    0x2066,  # LEFT-TO-RIGHT ISOLATE
    0x2067,  # RIGHT-TO-LEFT ISOLATE
    0x2068,  # FIRST STRONG ISOLATE
    0x2069,  # POP DIRECTIONAL ISOLATE
}

# C0 control characters (excluding common whitespace: \t \n \r)
C0_CONTROL_CODEPOINTS: set[int] = {i for i in range(0x00, 0x20) if i not in (0x09, 0x0A, 0x0D)}
# DEL
C0_CONTROL_CODEPOINTS.add(0x7F)
# C1 control characters
C1_CONTROL_CODEPOINTS: set[int] = set(range(0x80, 0xA0))


def is_confusable(cp: int) -> bool:
    return cp in CONFUSABLES


def confusable_target(cp: int) -> str | None:
    return CONFUSABLES.get(cp)


def is_invisible(cp: int) -> bool:
    return cp in INVISIBLE_CODEPOINTS


def is_bidi_override(cp: int) -> bool:
    return cp in BIDI_OVERRIDES


def is_control_char(cp: int) -> bool:
    return cp in C0_CONTROL_CODEPOINTS or cp in C1_CONTROL_CODEPOINTS


# Categories of suspicious codepoints for short classification strings
def category(cp: int) -> str:
    if is_invisible(cp):
        return "invisible"
    if is_bidi_override(cp):
        return "bidi_override"
    if is_control_char(cp):
        return "control_char"
    if is_confusable(cp):
        return "confusable"
    return "ok"