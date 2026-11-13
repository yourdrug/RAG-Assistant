"""Keep explicit table replacements interpretable after independent retrieval."""

from collections.abc import Sequence
import re

from domain.knowledge.enums import PageContentType
from domain.rag.ranking import RetrievalDocument


REPLACEMENT_BRIDGE_RE = re.compile(r"^\s*заменить\s+позицией\s*:?\s*$", re.IGNORECASE)
TABLE_TARGET_RE = re.compile(r"в\s+таблице\s+\d+(?:\.\d+)*", re.IGNORECASE)
POSITION_TARGET_END_RE = re.compile(r"\bпозицию\s*:?\s*$", re.IGNORECASE)


def attach_table_replacement_scope(docs: Sequence[RetrievalDocument]) -> None:
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

        # The operation immediately follows the old row in the source. Keep it
        # with the last row batch as source text, independently of dense context.
        if all(
            old.metadata.get(key) == bridge.metadata.get(key)
            for key in ("page", "pages", "page_start", "page_end")
        ):
            old.metadata["table_replacement_suffix"] = bridge.page_content.strip()

        parent = {"content": content}
        normalized = " ".join(content.split())

        for doc in (old, new):
            # The atomic scope supersedes generic adjacent-text table context
            # (table_context_sections), which would otherwise repeat the target
            # or bridge, or pull in earlier amendments from the same paragraph.
            kept = [
                unit
                for unit in doc.metadata.get("parent_units", [])
                if unit.get("unit_kind") != "table_context"
                and " ".join(unit.get("content", "").split()) not in normalized
            ]
            doc.metadata["parent_units"] = [*kept, dict(parent)]
