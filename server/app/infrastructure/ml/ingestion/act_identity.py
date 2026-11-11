"""Read an act's own dated heading without mistaking cited acts for its identity."""

from datetime import date
import re


MONTHS = (
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
)
ACT_HEADING = re.compile(
    r"^\s*(?:от\s+)?(?P<date>(?P<day>\d{1,2})\s+(?P<month>"
    + "|".join(MONTHS)
    + r")\s+(?P<year>\d{4})\s*г\.?|(?P<numeric>\d{2}\.\d{2}\.\d{4}))"
    r"\s*(?:№|N)\s*(?P<number>\d+(?:\s*[/\-]\s*\d+)*)[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)


def act_heading_title(text: str) -> tuple[str, str] | None:
    """Only accept a standalone date/number before the operative provisions.

    References inside amendment paragraphs and revision-history tables are
    deliberately excluded. No separators are reconstructed from filenames.
    """
    header = re.split(r"^\s*1[.)]\s|ПОСТАНОВЛЯ\w*\s*:", text[:4000], maxsplit=1, flags=re.MULTILINE)[0]
    match = ACT_HEADING.search(header)

    if match is None:
        return None

    kind = re.search(r"\b(постановление|указ|закон|приказ|решение)\b", header[: match.start()], re.IGNORECASE)

    if kind is None:
        return None

    try:
        if match["numeric"]:
            day, month, year = map(int, match["numeric"].split("."))
        else:
            day = int(match["day"])
            month = MONTHS.index(match["month"].casefold()) + 1
            year = int(match["year"])

        issued = date(year, month, day)
    except ValueError:
        return None

    number = re.sub(r"\s+", "", match["number"])
    return f"{kind[1].capitalize()} от {match['date']} № {number}", issued.isoformat()
