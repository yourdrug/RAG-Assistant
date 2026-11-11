"""Select table rows without losing repeated headers or split cell conditions."""

import re
from typing import Any

from domain.rag.ranking import RetrievalDocument


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


def focused_table_content(doc: RetrievalDocument, query: str) -> tuple[str, dict[str, Any]] | None:
    """Focus an unambiguous row-key match; preserve all columns and scope text.

    When no row can be identified safely, keep the retrieved batch. Its row
    interval still bounds any continuation fetch; never guess a different row.
    """
    low = doc.metadata.get("table_row_start")
    high = doc.metadata.get("table_row_end")
    header = doc.metadata.get("table_header")

    if not isinstance(low, int) or not isinstance(high, int) or low == high or not header:
        return None

    prefix, found, body = doc.page_content.partition(header + "\n")

    if not found:
        return None

    rows = body.splitlines()

    matches = (
        [offset for offset, row in enumerate(rows) if row_matches_query(row, query)]
        if len(rows) == high - low + 1
        else []
    )

    if len(matches) != 1:
        return None

    offset = matches[0]
    end = offset + 1

    # Editorial notes and continuation cells belong to the preceding data row.
    # Do not drop their conditions while focusing a field.
    while end < len(rows) and is_row_annotation(rows[end]):
        end += 1

    return (
        prefix + header + "\n" + "\n".join(rows[offset:end]),
        {**doc.metadata, "table_row_start": low + offset, "table_row_end": low + end - 1},
    )


def is_row_annotation(row: str) -> bool:
    cells = re.split(r"(?<!\\)\|", row.strip())[1:-1]

    if not cells:
        return False

    key = cells[0].strip()
    return not key or bool(re.match(r"\(|примечан|услови|исключени", key, re.IGNORECASE))


def row_matches_query(row: str, query: str) -> bool:  # noqa: PLR0911
    cells = re.split(r"(?<!\\)\|", row.strip())[1:-1]

    if not cells or not cells[0].strip():
        return False

    key = cells[0].strip()

    if key.isdigit():
        if re.search(
            r"(?:\b(?:пол[еяю]|строк[аиеу]|граф[аыеу]|позици[яию]|пункт\w*|п\.)|№)\s*"
            + re.escape(key)
            + r"(?![\w.])",
            query,
            re.IGNORECASE,
        ):
            return True
    elif re.search(r"(?<!\w)" + re.escape(key) + r"(?!\w)", query, re.IGNORECASE):
        return True

    if len(cells) <= 1:
        return False

    name = re.sub(r"\u2039br\u203a|<br\s*/?>|\b(?:an|n|a)\.\.\d+", " ", cells[1])

    tokens = word_stems(name)
    query_stems = word_stems(query)
    return bool(tokens) and all(
        any(stems_match(token, candidate) for candidate in query_stems) for token in tokens
    )


WORD_ENDING_RE = re.compile(
    r"(?:иями|иях|ием|ией|ями|ами|ого|его|ому|ему|ыми|ими|ия|ии|ию|ой|ей|ий|ый|ая|яя|ое|ее|ие|ые"
    r"|ов|ев|ам|ям|ом|ем|ах|ях|ы|и|у|ю|я|е|а|о)$"
)
MIN_PREFIX_STEM_LENGTH = 5
MAX_STEM_LENGTH_DIFFERENCE = 2


def word_stems(text: str) -> set[str]:
    """Approximate stems for inflection-tolerant name matching in table rows."""
    return {
        WORD_ENDING_RE.sub("", word) for word in re.findall(r"[а-яёa-z]{3,}", text.lower().replace("ё", "е"))
    }


def stems_match(left: str, right: str) -> bool:
    """Treat stems as equal when one extends the other by a short ending remnant.

    Suffix stripping cannot normalize every case pair ("наименован" from
    "наименование", "наименовани" from "наименования" in some paths), so a
    long shared prefix with a small length difference also counts.
    """
    if left == right:
        return True

    shorter, longer = sorted((left, right), key=len)
    return (
        len(shorter) >= MIN_PREFIX_STEM_LENGTH
        and len(longer) - len(shorter) <= MAX_STEM_LENGTH_DIFFERENCE
        and longer.startswith(shorter)
    )
