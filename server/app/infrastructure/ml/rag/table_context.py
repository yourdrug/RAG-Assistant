"""Select table rows without losing repeated headers or split cell conditions."""

import re

from langchain.schema import Document


TABLE_OVERVIEW_RE = re.compile(
    r"перечисл|список\s+(?:строк|полей|реквизит|видов|категори)|"
    r"(?:сравн\w*|сопостав\w*)\s+(?:все\s+)?(?:строк|поля\b|полей\b|реквизит|категори)|"
    r"все\s+(?:строк|значени|поля\b|полей\b|реквизит|вид[ыа]|категори)|"
    r"какие\s+(?:поля\b|строк|сведени|реквизит|категори)|"
    r"list\b|compare\s+(?:rows|fields|categories)|all\s+(?:rows|fields|categories)",
    re.IGNORECASE,
)


def requests_table_overview(query: str) -> bool:
    """Conditional-rule expansion alone does not authorize reading an entire table."""
    return bool(TABLE_OVERVIEW_RE.search(query))


def focus_table_rows(doc: Document, query: str) -> Document:
    """Focus an unambiguous row-key match; preserve all columns and scope text.

    When no row can be identified safely, keep the retrieved batch. Its row
    interval still bounds any continuation fetch; never guess a different row.
    """
    low = doc.metadata.get("table_row_start")
    high = doc.metadata.get("table_row_end")
    header = doc.metadata.get("table_header")
    if not isinstance(low, int) or not isinstance(high, int) or low == high or not header:
        return doc
    prefix, found, body = doc.page_content.partition(header + "\n")
    if not found:
        return doc
    rows = body.splitlines()
    if len(rows) != high - low + 1:
        return doc
    matches = []
    for offset, row in enumerate(rows):
        if row_matches_query(row, query):
            matches.append(offset)
    if len(matches) != 1:
        return doc
    offset = matches[0]
    end = offset + 1
    # Editorial notes and continuation cells belong to the preceding data row.
    # Do not drop their conditions while focusing a field.
    while end < len(rows) and is_row_annotation(rows[end]):
        end += 1
    return Document(
        page_content=prefix + header + "\n" + "\n".join(rows[offset:end]),
        metadata={**doc.metadata, "table_row_start": low + offset, "table_row_end": low + end - 1},
    )


def is_row_annotation(row: str) -> bool:
    cells = re.split(r"(?<!\\)\|", row.strip())[1:-1]
    if not cells:
        return False
    key = cells[0].strip()
    return not key or bool(re.match(r"\(|примечан|услови|исключени", key, re.IGNORECASE))


def row_matches_query(row: str, query: str) -> bool:
    cells = re.split(r"(?<!\\)\|", row.strip())[1:-1]
    if not cells or not cells[0].strip():
        return False
    candidates = [cells[0].strip()]
    if len(cells) > 1:
        name = re.sub(r"‹br›|<br\s*/?>|\b(?:an|n|a)\.\.\d+", " ", cells[1])
        name = re.sub(r"\s+", " ", name).strip()
        if name and not name.isdigit():
            candidates.append(name)
    return any(
        re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", query, re.IGNORECASE) for value in candidates
    )
