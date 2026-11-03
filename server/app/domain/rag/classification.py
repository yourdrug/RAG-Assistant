"""Question classification and conditional-context heuristics.

Rules are ordered explicitly; all matching is independent of retrieval and LLM clients.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from domain.knowledge.enums import DocDomain
from domain.rag.breadth import Breadth
from domain.shared.dates import RU_MONTHS, parse_date_guess
from domain.shared.exceptions import ValidationError


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import date


MIN_CONDITIONAL_CHUNKS = 2


_QUANTITATIVE_CONDITIONAL_PATTERNS = [
    r"сколько\s+(нужно|необходимо|следует|требуется|должно)",
    r"сколько\s+.+\bесли\b",
    r"какое\s+количество\s+.+\bесли\b",
    r"в\s+каком\s+размере",
    r"какой\s+процент\s+.+\bесли\b",
]


_NARROW_OVERRIDE_PATTERNS = [
    r"как\w*\s+(убедиться|проверить|узнать|найти|получить|скачать|открыть)",
    r"где\s+(найти|скачать|посмотреть|открыть)",
    r"что\s+(такое|означает|является)",
    r"какой\s+(пароль|срок|размер|номер|формат|статус)",
    r"каки[ех]\s+исключени",
    r"каки[ех]\s+особы",
    r"каки[ех]\s+альтернатив",
    r"почему\s+",
    r"можно\s+ли\s+",
]


_BROAD_PATTERNS = [
    r"подробно",
    r"объясни\s+вс[ёе]",
    r"расскажи\s+про",
    r"как\w*\s+(работает|устроено|происходит|проводится)",
    r"порядок\s+(получения|выдачи|оформления|получить)",
    r"система\s+\w+",
    r"вс[ёе]\s+про\b",
    r"полностью",
    r"детальн",
    r"максимальн",
    r"каки[ех]\s+(условия|требования|правила|нормы|критерии)",
    r"перечисл\w*",
    r"что\s+входит",
    r"что\s+включ\w+",
    r"список\s+\w+",
    r"какие\s+\w+\s+нужн",
    r"сколько\s+",  # остальные "сколько ..." тоже лучше раскрывать подробно
]


COMPOUND_PATTERNS = [
    r"сравни\s+.+\s+и\s+",
    r"расскажи\s+про\s+.*\s+и\s+",
    r"как\w*\s+.*\s+и\s+",
    r"что\s+.*\s+и\s+что\s+",
    r"какие\s+.*\s+и\s+какие\s+",
    r"опиши\s+.*\s+а\s+также\s+",
    r"объясни\s+.*\s+и\s+",
]


_CONDITIONAL_MARKERS = [
    r"за\s+исключением",
    r"исключени",
    r"но\s+не\s+менее",
    r"но\s+не\s+более",
    r"в\s+случа[еяи]",
    r"при\s+условии",
    r"если\s+.+\s+-\s+.+",
    r"\d+%",
    r"не\s+менее\s+\d+",
    r"не\s+более\s+\d+",
]


_EXACT_REF_RE = re.compile(r"(статья|пункт|раздел|глава|параграф|п\.|ст\.)\s*\d+", re.IGNORECASE)


_OUT_OF_DOMAIN_PATTERNS = [
    # Competitive programming / algorithms
    r"(codeforces|leetcode|hackerrank|задач[ауи]\s+по\s+олимпиад)",
    r"алгоритм\s+(быстрой\s+сортировки|сортировки|двоичного\s+поиска)",
    r"(решени[ея]\s+задач[иу]\s+codeforces|codeforces\s+\d+[a-zA-Z])",
    r"(time\s+complexity|пространственн\w+\s+сложност|big\s*o|O\(n\))",
    # Non-corporate topics in mathematics and natural sciences
    r"(доказатель\w*\s+теорем|вычисли\w*\s+(интеграл|производн))",
    r"(закон[ауи]\s+Ньютона|квантов\w+\s+механик|периодическ\w+\s+систем)",
    # Programming tutorials (non-corporate)
    r"(как\s+написать\s+(скрипт|программ|функци)|tutorial|туториал)",
    r"(python\s+для\s+начинающих|изучени[ея]\s+programming)",
    # Personal / non-work topics
    r"(рецепт[ауи]\s+готовк|что\s+приготовить|кулинарн)",
    r"(прогноз\s+погод|гороскоп|как\s+похудеть)",
]


_OUT_OF_DOMAIN_RE = re.compile("|".join(_OUT_OF_DOMAIN_PATTERNS), re.IGNORECASE)


def classify_question_breadth(question: str) -> Breadth:
    """Apply quantity rules, then narrow overrides, then broad rules; default to narrow."""
    q = question.lower()

    # Conditional quantity rules take priority over narrow overrides.
    if any(re.search(r"\b(?:" + p + ")", q) for p in _QUANTITATIVE_CONDITIONAL_PATTERNS):
        return Breadth.BROAD

    if any(re.search(r"\b(?:" + p + ")", q) for p in _NARROW_OVERRIDE_PATTERNS):
        return Breadth.NARROW

    return Breadth.BROAD if any(re.search(r"\b(?:" + p + ")", q) for p in _BROAD_PATTERNS) else Breadth.NARROW


def needs_decomposition(question: str) -> bool:
    """Heuristic: check if the question contains multiple independent sub-topics.

    Uses regex patterns to detect compound structures ("X и Y", "сравни X и Y").
    Returns True if the question likely benefits from decomposition into
    separate retrieval queries.
    """
    q = question.lower()
    return any(re.search(r"\b(?:" + p + ")", q) for p in COMPOUND_PATTERNS)


def should_enumerate_cases(
    context_texts: Sequence[str | None],
) -> bool:
    """Detect when context contains multiple conditional/exclusive rules.

    Returns True when the LLM should enumerate all cases with conditions
    rather than giving a single-number answer. The question-side specificity
    is left to the LLM -- the BROAD prompt's formatting rules handle
    sub-point expansion.

    """
    if len(context_texts) < MIN_CONDITIONAL_CHUNKS:
        return False

    chunks_with_conditions = sum(
        1 for text in context_texts if text and any(re.search(p, text.lower()) for p in _CONDITIONAL_MARKERS)
    )

    return chunks_with_conditions >= MIN_CONDITIONAL_CHUNKS


def classify_query_domain(question: str, patterns: Mapping[str, tuple[str, ...]]) -> str:
    """Select the first matching registered profile; fall back to general."""
    return next(
        (
            key
            for key, rules in patterns.items()
            if any(re.search(r"\b(?:" + rule + ")", question, re.IGNORECASE) for rule in rules)
        ),
        DocDomain.GENERAL,
    )


def has_exact_reference(question: str) -> bool:
    """Check if question contains an exact structural reference (article, paragraph, etc.)."""
    return bool(_EXACT_REF_RE.search(question))


def is_out_of_domain(question: str) -> bool:
    """Check if a question is clearly outside the corporate document scope.

    Returns True if the question matches patterns that indicate it's about
    competitive programming, personal topics, or other non-corporate content.

    This is a pre-retrieval filter -- fast regex check before any LLM call.
    """
    return bool(_OUT_OF_DOMAIN_RE.search(question))


QUESTION_DATE_PATTERN = (
    r"\d{4}-\d{2}-\d{2}|\d{1,2}[./]\d{1,2}[./]\d{4}|"
    r"\d{1,2}\s+(?:" + "|".join(RU_MONTHS) + r")\s+\d{4}"
)
_QUESTION_AS_OF_RE = re.compile(
    r"\b(?:по состоянию на|на дату|на)\s+(?P<date>" + QUESTION_DATE_PATTERN + r")(?!\w)",
    re.IGNORECASE,
)


def explicit_question_date(text: str) -> date | None:
    """Extract a date explicitly requested as the temporal state of the answer."""
    match = _QUESTION_AS_OF_RE.search(text)

    if match is None:
        return None

    parsed = parse_date_guess(match.group("date"))

    if parsed is None:
        msg = "Invalid date in question"
        raise ValidationError(msg)

    return parsed
