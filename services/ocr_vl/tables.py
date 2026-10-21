"""Convert model HTML tables without shifting merged cells between columns."""

from html.parser import HTMLParser
import re


UNREADABLE = "[неразборчиво]"


class TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.row = None
        self.cell = None
        self.span = (1, 1)

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        elif tag in ("td", "th"):
            self.cell = []
            attributes = dict(attrs)
            self.span = tuple(int(attributes.get(key, "1")) for key in ("rowspan", "colspan"))
            if any(value < 1 or value > 100 for value in self.span):
                raise ValueError("Invalid table span")
        elif tag == "br" and self.cell is not None:
            self.cell.append("\n")

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None and self.row is not None:
            self.row.append(("".join(self.cell).strip(), *self.span))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None


def table_markdown(content):
    if "<table" not in content.lower():
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if (
            len(lines) >= 2
            and re.fullmatch(r"\|(?:\s*:?-{3,}:?\s*\|)+", lines[1])
            and all(line.startswith("|") and line.endswith("|") for line in lines)
        ):
            widths = {len(re.split(r"(?<!\\)\|", line)) for line in lines}
            if len(widths) == 1:
                return "\n".join(lines).replace("�", UNREADABLE)
        return ""
    parser = TableParser()
    parser.feed(content)
    grid = {}
    for row_index, row in enumerate(parser.rows):
        column = 0
        for text, rowspan, colspan in row:
            while (row_index, column) in grid:
                column += 1
            # Repeating a merged label preserves its relation to each row/column.
            for y in range(row_index, row_index + rowspan):
                for x in range(column, column + colspan):
                    if (y, x) in grid:
                        raise ValueError("Overlapping table cells")
                    grid[y, x] = text
            column += colspan
    if not grid:
        return ""
    height = max(y for y, _ in grid) + 1
    width = max(x for _, x in grid) + 1
    lines = []
    for y in range(height):
        cells = []
        for x in range(width):
            cell = grid.get((y, x), UNREADABLE).replace("�", UNREADABLE).replace("|", "\\|")
            cells.append(re.sub(r"[\r\n]+", " <br> ", cell))
        lines.append("| " + " | ".join(cells) + " |")
    lines.insert(1, "| " + " | ".join(["---"] * width) + " |")
    return "\n".join(lines)
