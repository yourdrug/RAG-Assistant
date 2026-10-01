"""Question classification and conditional-context heuristics.

Rules are ordered explicitly; all matching is independent of retrieval and LLM clients.
"""

from __future__ import annotations

import re

from domain.value_objects.doc_domain import DocDomain
from domain.value_objects.llm_provider import Breadth


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


_LEGAL_PATTERNS = [
    r"вправе\s+ли",
    r"обязан\s+ли",
    r"подлежит\s+ли",
    r"несёт\s+ли\s+ответственность",
    r"стать[юяе]\s+\d+",
    r"пункт[ае]?\s+\d+",
    r"в\s+соответствии\s+с",
    r"согласно\s+(закону|договору|статье)",
    r"нарушени[ея]\s+(условий|закона)",
]


_EXACT_REF_RE = re.compile(r"(статья|пункт|раздел|глава|параграф|п\.|ст\.)\s*\d+", re.IGNORECASE)


_OUT_OF_DOMAIN_PATTERNS = [
    # Competitive programming / algorithms
    r"(codeforces|leetcode|hackerrank|algorithm|алгоритм|задач[ауи]\s+по\s+олимпиад)",
    r"(решени[ея]\s+задач[иу]\s+codeforces|codeforces\s+\d+[a-zA-Z])",
    r"(time\s+complexity|пространственн\w+\s+сложност|big\s*o|O\(n\))",
    # Math / physics / chemistry (non-corporate)
    r"(доказатель\w*\s+теорем|формул[аы]\s+вычислени|интеграл|производн)",
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
    if any(re.search(p, q) for p in _QUANTITATIVE_CONDITIONAL_PATTERNS):
        return Breadth.BROAD

    if any(re.search(p, q) for p in _NARROW_OVERRIDE_PATTERNS):
        return Breadth.NARROW

    return Breadth.BROAD if any(re.search(p, q) for p in _BROAD_PATTERNS) else Breadth.NARROW


def needs_decomposition(question: str) -> bool:
    """Heuristic: check if the question contains multiple independent sub-topics.

    Uses regex patterns to detect compound structures ("X и Y", "сравни X и Y").
    Returns True if the question likely benefits from decomposition into
    separate retrieval queries.
    """
    q = question.lower()
    return any(re.search(p, q) for p in COMPOUND_PATTERNS)


def should_enumerate_cases(question: str, context_texts: list[str | None]) -> bool:
    """Detect when context contains multiple conditional/exclusive rules.

    Returns True when the LLM should enumerate all cases with conditions
    rather than giving a single-number answer. The question-side specificity
    is left to the LLM -- the BROAD prompt's formatting rules handle
    sub-point expansion.

    ``question`` is kept for API compatibility; only context affects this decision.
    """
    if len(context_texts) < 2:
        return False

    chunks_with_conditions = sum(
        1 for text in context_texts if text and any(re.search(p, text.lower()) for p in _CONDITIONAL_MARKERS)
    )

    return chunks_with_conditions >= 2


def classify_query_domain(question: str) -> DocDomain:
    """Classify query as 'legal' or 'general' based on question patterns."""
    q = question.lower()
    return DocDomain.LEGAL if any(re.search(p, q) for p in _LEGAL_PATTERNS) else DocDomain.GENERAL


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
