"""Serialize table cells without confusing inner line breaks with row boundaries."""

import re


def markdown_cell(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().replace("|", "\\|")
    return re.sub(r"[\r\n\u0085\u2028\u2029\v\f]+", " <br> ", text)
