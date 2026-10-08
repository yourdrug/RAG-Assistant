"""Preserve distinct structural provisions when a candidate budget is applied."""

from domain.domain_profile.protocol import SENTENCE_UNIT_KIND


def prioritize_distinct_provisions(items: list) -> list:
    """Stable first pass over provisions, followed by their remaining chunks.

    Accept documents or (document, score) pairs. Sentence splits share their
    parent's provision. Unstructured chunks and tables retain their own slots.
    Scope includes the document/version so unrelated provisions never collapse.
    No text is removed: remaining chunks can fill any unused budget.
    """
    seen = set()
    first = []
    remaining = []
    for item in items:
        doc = item[0] if isinstance(item, tuple) else item
        metadata = doc.metadata
        section = metadata.get("section") or metadata.get("heading")
        source = metadata.get("document_id") or metadata.get("source")
        if not section or not source or metadata.get("table_id"):
            first.append(item)
            continue
        parts = section.replace(" › ", " > ").split(" > ")
        if parts[-1] == SENTENCE_UNIT_KIND:
            parts = parts[:-1]
        if not parts:
            first.append(item)
            continue
        key = (source, metadata.get("act_version_id"), tuple(parts))
        if key in seen:
            remaining.append(item)
        else:
            seen.add(key)
            first.append(item)
    return first + remaining
