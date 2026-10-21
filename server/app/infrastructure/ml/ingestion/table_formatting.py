"""Serialize table cells without confusing inner line breaks with row boundaries."""

import re


def markdown_cell(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().replace("|", "\\|")
    return re.sub(r"[\r\n\u0085\u2028\u2029\v\f]+", " <br> ", text)


def render_table(rows: list[list[str]]) -> str:
    """Render normalized cells, retaining the header-only table convention."""
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]
    header = "| " + " | ".join(padded[0]) + " |"
    if len(padded) == 1:
        return header
    separator = "|" + "|".join(["---"] * width) + "|"
    body = ["| " + " | ".join(row) + " |" for row in padded[1:]]
    return "\n".join([header, separator, *body])
