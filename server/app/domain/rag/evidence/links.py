"""Recognize explicit replacement and procedure links in adjacent source chunks."""

import re

from domain.knowledge.enums import PageContentType
from domain.rag.retrieval import RetrievalDocument


STAGE_RE = re.compile(r"^При\s+(?:передаче|отгрузке|реализации)\b", re.IGNORECASE)


def chunk_body(doc: RetrievalDocument) -> str:
    """Ignore repeated structural headings when finding a sentence's condition."""
    lines = doc.page_content.splitlines()
    parents = {parent.get("content", "").strip() for parent in doc.metadata.get("parent_units", [])}
    return "\n".join(
        line for line in lines if not line.startswith("[Раздел:") and line.strip() not in parents
    )


def replacement_scopes(docs: list[RetrievalDocument]) -> dict[int, str]:
    scopes = {}

    for target, old, bridge, new in zip(docs, docs[1:], docs[2:], docs[3:], strict=False):
        if not re.search(r"в\s+таблице\s+\d+(?:\.\d+)*", target.page_content, re.IGNORECASE):
            continue

        if not re.search(r"\bпозицию\s*:?\s*$", target.page_content):
            continue

        if not re.fullmatch(r"\s*заменить\s+позицией\s*:?\s*", bridge.page_content, re.IGNORECASE):
            continue

        if any(doc.metadata.get("content_type") != PageContentType.TABLE.value for doc in (old, new)):
            continue

        # Links are source adjacency, never relevance/prompt ordering.
        indices = [doc.metadata["chunk_index"] for doc in (target, old, bridge, new)]

        if indices != list(range(indices[0], indices[0] + 4)):
            continue

        start = list(re.finditer(r"в\s+таблице\s+\d+(?:\.\d+)*", target.page_content, re.IGNORECASE))[
            -1
        ].start()
        scope = "\n".join(
            [target.page_content[start:], old.page_content, bridge.page_content, new.page_content]
        )
        scopes[old.metadata["chunk_index"]] = scope
        scopes[new.metadata["chunk_index"]] = scope

    return scopes


def procedure_scopes(docs: list[RetrievalDocument]) -> dict[int, str]:
    scopes = {}

    for previous, current in zip(docs, docs[1:], strict=False):
        if current.metadata["chunk_index"] != previous.metadata["chunk_index"] + 1:
            continue

        if current.metadata.get("section") != previous.metadata.get("section"):
            continue

        if any(STAGE_RE.match(line) for line in chunk_body(current).splitlines()):
            continue

        condition = next((line for line in chunk_body(previous).splitlines() if STAGE_RE.match(line)), None)
        continuation = next(
            (line for line in chunk_body(current).splitlines() if re.match(r"В накладной\s", line)), None
        )

        if condition and continuation:
            scope = condition + "\n" + continuation
            scopes[previous.metadata["chunk_index"]] = scope
            scopes[current.metadata["chunk_index"]] = scope

    return scopes


def link_modes(question: str) -> tuple[bool, bool]:
    return bool(re.search(r"измен|замен", question, re.IGNORECASE)), bool(
        re.search(r"поряд|этап|последующ", question, re.IGNORECASE)
    )


def link_seed_priority(doc: RetrievalDocument) -> bool:
    if doc.metadata.get("content_type") == PageContentType.TABLE.value:
        return bool(doc.metadata.get("table_header"))

    return not any(
        STAGE_RE.match(line) or line.startswith("В накладной ") for line in chunk_body(doc).splitlines()
    )
