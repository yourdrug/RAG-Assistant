"""Bounded search probes derived from the question, without answer knowledge."""

import re

from infrastructure.domain_packs.belarus_law.evidence.applicability import (
    approved_classifier,
    timing_requested,
)


def evidence_search_queries(question: str) -> list[str]:
    queries = [question]

    if classifier := approved_classifier(question):
        # Search the referenced object without the referring EDI/waybill
        # structure dominating both lexical and dense candidate generation.
        queries.append(f"Утверждение: классификатор {classifier}")

    table = re.search(r"таблиц\w*\s+(\d+(?:\.\d+)*)", question, re.IGNORECASE)

    if table:
        # A long comparison query can retrieve only the more salient field.
        # Search each named field independently, retaining the original query.
        queries.extend(
            f"Таблица {table[1]}, поле {match[2]} «{match[1]}»"
            for match in re.finditer(r"пол[еяю]\s+«([^»]+)»\s*\(позици[яи]\s+(\d+)", question, re.IGNORECASE)
        )

    if re.search(r"режим|периодич|как часто", question, re.IGNORECASE):
        # The category's long description can dominate both dense and sparse
        # search. Search the requested action separately; the original query
        # and the answer-reading rules still carry the category and negation.
        probe = re.sub(
            r"о\s+товарах,?\s+(?:не\s+)?включ[её]н\w*\s+в\s+переч\w*[^,?]*,?",
            "",
            question,
            flags=re.IGNORECASE,
        )

        if probe != question:
            queries.append(" ".join(probe.split()))

    if timing_requested(question):
        # Acts state timing in fixed wording, not in the question's words
        # ("частота отправки"); this probe carries only that wording.
        queries.append(TIMING_WORDING_PROBE)

    return list(dict.fromkeys(queries))[:4]


TIMING_WORDING_PROBE = (
    "Периодичность передачи сведений: в режиме реального времени, не реже одного раза в сутки"
)
