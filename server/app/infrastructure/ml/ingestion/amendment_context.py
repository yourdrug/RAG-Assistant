"""Keep explicit table replacements interpretable after independent retrieval."""

import re

from domain.value_objects.page_content_type import PageContentType

REPLACEMENT_BRIDGE_RE = re.compile(r"^\s*заменить\s+позицией\s*:?\s*$", re.IGNORECASE)
TABLE_TARGET_RE = re.compile(r"в\s+таблице\s+\d+(?:\.\d+)*", re.IGNORECASE)
POSITION_TARGET_END_RE = re.compile(r"\bпозицию\s*:?\s*$", re.IGNORECASE)


def attach_table_replacement_scope(docs) -> None:
    """Attach the exact target/old/operation/new sequence to both table parts.

    Only adjacent sections under the same heading and source qualify. Existing
    parent_units persistence and prompt budgeting carry this indivisible scope;
    no relationship is inferred from row equality or the order of search hits.
    """
    for index in range(1, len(docs) - 2):
        target, old, bridge, new = docs[index - 1 : index + 3]
        matches = list(TABLE_TARGET_RE.finditer(target.page_content))
        if not matches or not POSITION_TARGET_END_RE.search(target.page_content):
            continue
        if not REPLACEMENT_BRIDGE_RE.fullmatch(bridge.page_content):
            continue
        if any(doc.metadata.get("content_type") != PageContentType.TABLE.value for doc in (old, new)):
            continue
        if (
            len(
                {
                    (doc.metadata.get("source"), doc.metadata.get("section"))
                    for doc in (target, old, bridge, new)
                }
            )
            != 1
        ):
            continue
        # Earlier amendments in the same prose section are not part of this one.
        content = "\n".join(
            [
                target.page_content[matches[-1].start() :],
                old.page_content,
                bridge.page_content,
                new.page_content,
            ]
        )
        parent = {"content": content}
        for doc in (old, new):
            doc.metadata["parent_units"] = [*doc.metadata.get("parent_units", []), dict(parent)]
