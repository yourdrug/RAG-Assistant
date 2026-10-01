"""Prompt injection detection patterns — pure business logic.

Framework-agnostic regex patterns for detecting direct prompt injection,
role hijacking, prompt extraction, and obfuscation techniques.
No infrastructure imports — safe for domain layer.
"""

from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Unicode normalization (defense against obfuscation)
# ---------------------------------------------------------------------------

# Zero-width and invisible characters used to evade pattern matching.
_ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\u200e\u200f\u2060\ufeff\u00ad\u180e]")


def normalize_unicode(text: str) -> str:
    """Normalize text to defeat obfuscation-based injection.

    1. Strips zero-width / invisible characters (U+200B..U+200F, U+2060,
       U+FEFF, U+00AD, U+180E).
    2. Applies NFKC normalization to fold fullwidth/homoglyph characters
       to their canonical ASCII forms where applicable.
    """
    text = _ZERO_WIDTH_RE.sub("", text)
    text = unicodedata.normalize("NFKC", text)
    return text


# ---------------------------------------------------------------------------
# Injection pattern library
# ---------------------------------------------------------------------------
#
# Each pattern is (pattern_id, category, compiled_regex, severity).
# Severity: "block" = reject immediately, "flag" = log + allow.

INJECTION_PATTERNS: list[tuple[str, str, re.Pattern, str]] = [
    # -- Direct instruction override --
    (
        "d001",
        "instruction_override",
        re.compile(
            r"(?:игнорируй|игнорировать|забудь|забыть)\s+(?:все|всё|все\s+предыдущие|предыдущие)"
            r"(?:\s+(?:инструкции|команды|правила|указания))?",
            re.IGNORECASE,
        ),
        "block",
    ),
    (
        "d002",
        "instruction_override",
        re.compile(
            r"ignore\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+(?:instructions?|prompts?|rules?|commands?|directives?)",
            re.IGNORECASE,
        ),
        "block",
    ),
    (
        "d003",
        "instruction_override",
        re.compile(
            r"(?:disregard|override|forget)\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+(?:instructions?|prompts?|rules?)",
            re.IGNORECASE,
        ),
        "block",
    ),
    (
        "d004",
        "instruction_override",
        re.compile(
            r"(?:new|updated?)\s+(?:system\s+)?(?:prompt|instructions?|rules?)\s*[:=]",
            re.IGNORECASE,
        ),
        "block",
    ),
    # -- Role hijacking --
    (
        "d005",
        "role_hijack",
        re.compile(
            r"(?:you\s+are\s+now|теперь\s+ты|отныне\s+ты|ты\s+теперь)\s+(?:a\s+)?(?:DAN|jailbroken|unrestricted|без\s+ограничений)",
            re.IGNORECASE,
        ),
        "block",
    ),
    (
        "d006",
        "role_hijack",
        re.compile(
            r"(?:developer\s+mode|dev\s+mode|jailbreak|DAN\s+mode|do\s+anything\s+now)",
            re.IGNORECASE,
        ),
        "block",
    ),
    (
        "d007",
        "role_hijack",
        re.compile(
            r"(?:act\s+as|pretend\s+(?:to\s+be|you\s+are)|притворяйся|представь\s+себя)"
            r".*(?:unrestricted|без\s+ограничений|no\s+restrictions|without\s+restrictions|jailbroken)",
            re.IGNORECASE,
        ),
        "block",
    ),
    # -- System prompt extraction --
    (
        "d008",
        "prompt_extraction",
        re.compile(
            r"(?:output|print|show|reveal|display|выведи|покажи|раскрой|скажи)\s+"
            r"(?:your|the|свой|твой|ваш|my)\s+"
            r"(?:system\s+prompt|system\s+instructions|internal\s+instructions|"
            r"системный\s+промпт|внутренние\s+инструкции)",
            re.IGNORECASE,
        ),
        "block",
    ),
    (
        "d009",
        "prompt_extraction",
        re.compile(
            r"(?:что|what)\s+(?:такое|is)\s+(?:твой|your)\s+" r"(?:системный\s+промпт|system\s+prompt)",
            re.IGNORECASE,
        ),
        "block",
    ),
    (
        "d010",
        "prompt_extraction",
        re.compile(
            r"(?:repeat|повтори|прочитай|read)\s+(?:your|твои|свои|все)\s+"
            r"(?:initial\s+instructions|инструкции|правила|prompts?)",
            re.IGNORECASE,
        ),
        "block",
    ),
    # -- Fake system messages --
    (
        "d011",
        "fake_system",
        re.compile(
            r"\[(?:system|admin|developer)\]",
            re.IGNORECASE,
        ),
        "block",
    ),
    (
        "d012",
        "fake_system",
        re.compile(
            r"###\s*(?:new\s+)?(?:system\s+)?(?:prompt|instructions?|rules?)\s*###",
            re.IGNORECASE,
        ),
        "block",
    ),
    # -- Tool / action injection --
    (
        "d013",
        "tool_injection",
        re.compile(
            r"(?:выполни|execute|run|do)\s+(?:следующее|the\s+following|эти\s+команды)",
            re.IGNORECASE,
        ),
        "flag",
    ),
    (
        "d014",
        "tool_injection",
        re.compile(
            r"(?:sudo|rm\s+-rf|curl\s+|wget\s+|chmod\s+|/etc/passwd)",
            re.IGNORECASE,
        ),
        "block",
    ),
    # -- Encoding / obfuscation (flag only — may be legitimate) --
    (
        "d015",
        "obfuscation",
        re.compile(
            r"(?:base64|decode|раскодируй|deobfuscate|rot13|hex\s+decode)",
            re.IGNORECASE,
        ),
        "flag",
    ),
]

# Precompiled for performance
_BLOCK_PATTERNS = [(pid, cat, rx) for pid, cat, rx, sev in INJECTION_PATTERNS if sev == "block"]
_FLAG_PATTERNS = [(pid, cat, rx) for pid, cat, rx, sev in INJECTION_PATTERNS if sev == "flag"]


def detect_injection(text: str) -> list[tuple[str, str]]:
    """Detect prompt injection patterns in text.

    Normalizes Unicode first to defeat obfuscation, then checks all patterns.
    Returns list of (pattern_id, category) for each match.
    """
    normalized = normalize_unicode(text)
    matches: list[tuple[str, str]] = []
    for pid, cat, rx in _BLOCK_PATTERNS + _FLAG_PATTERNS:
        if rx.search(normalized):
            matches.append((pid, cat))
    return matches


def has_blocking_injection(text: str) -> bool:
    """Return True if text contains an injection pattern with 'block' severity."""
    normalized = normalize_unicode(text)
    return any(rx.search(normalized) for _, _, rx in _BLOCK_PATTERNS)


def has_flagged_injection(text: str) -> bool:
    """Return True if text contains an injection pattern with 'flag' severity."""
    normalized = normalize_unicode(text)
    return any(rx.search(normalized) for _, _, rx in _FLAG_PATTERNS)
