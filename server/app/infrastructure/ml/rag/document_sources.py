"""Stable document identities shared by prompt formatting and source extraction."""

from collections import OrderedDict


def document_identity(doc) -> tuple[str, object]:
    document_id = doc.metadata.get("document_id")
    if document_id is not None:
        return ("document", document_id)
    # Preserve the full path: equal basenames do not imply equal documents.
    return ("source", doc.metadata.get("source", "unknown"))


def group_by_document(docs) -> list[list]:
    groups: OrderedDict[tuple, list] = OrderedDict()
    for item in docs:
        doc = item[0] if isinstance(item, tuple) else item
        groups.setdefault(document_identity(doc), []).append(item)
    return list(groups.values())
