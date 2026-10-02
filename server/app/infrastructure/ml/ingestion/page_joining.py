"""Join PDF page edges without splitting words or numeric ranges."""

import re

_TRAILING_WORD_RE = re.compile(r"([^\W_]+)([-\u00ad])\s*$")
_LEADING_WORD_RE = re.compile(r"\s*([^\W_]+)")
_COMPOUND_WORD_RE = re.compile(r"[^\W_]+(?:-[^\W_]+)+")
_WORD_RE = re.compile(r"[^\W_]+")

# These components commonly retain a lexical hyphen. Document-local spelling
# takes precedence; a hard hyphen alone cannot disambiguate every compound.
_COMPOUND_PREFIXES = frozenset(
    {
        "интернет",
        "веб",
        "онлайн",
        "офлайн",
        "бизнес",
        "пресс",
        "вице",
        "экс",
        "мини",
        "макси",
        "арт",
        "поп",
        "рок",
        "штаб",
        "северо",
        "юго",
        "научно",
        "социально",
        "общественно",
        "e",
        "ex",
        "self",
        "well",
    }
)


def page_word_evidence(texts: list[str]) -> tuple[set[str], set[str]]:
    """Use spellings elsewhere in the same text run to preserve real hyphens."""
    text = "\n".join(texts).lower()
    return set(_WORD_RE.findall(text)), set(_COMPOUND_WORD_RE.findall(text))


def join_page_word(before: str, after: str, evidence: tuple[set[str], set[str]]) -> tuple[str, str] | None:
    """Return adjusted page edges if a word or range continues across pages."""
    left = _TRAILING_WORD_RE.search(before)
    right = _LEADING_WORD_RE.match(after)
    if left is None or right is None:
        return None
    stem, hyphen = left.groups()
    suffix = right.group(1)
    words, compounds = evidence
    if stem.isdigit() and suffix.isdigit():
        return before.rstrip(), after.lstrip()
    if not stem.isalpha() or not suffix.isalpha():
        return None
    joined = (stem + suffix).lower()
    compound = (stem + "-" + suffix).lower()
    if hyphen == "\u00ad" or joined in words:
        remove = True
    elif compound in compounds or stem.lower() in _COMPOUND_PREFIXES:
        remove = False
    else:
        # Uppercase identifiers and proper-name compounds keep their hyphen.
        remove = stem[-1].islower() and suffix[0].islower()
    return before.rstrip()[:-1] if remove else before.rstrip(), after.lstrip()
