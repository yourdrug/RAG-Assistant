"""Enforce blank lines around ``if`` / ``for`` / ``while`` statements.

Rules:
    * a blank line before the statement unless it is the first one in its block;
    * a blank line after it unless it is the last one in its block;
    * ``elif`` / ``else`` belong to the preceding statement, so nothing is
      inserted before them (same for ``else`` of ``for`` / ``while``).

A comment attached to a statement stays attached: the blank line goes above it.

Usage:
    python tools/blank_lines.py app tests tools          # fix in place
    python tools/blank_lines.py --check app        # only report, change nothing
    python tools/blank_lines.py --also try --also with app

Exit code is 1 if any file was (or, with --check, would be) changed, which is
what pre-commit expects from a fixing hook.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys
from typing import TypeVar, cast

import libcst as cst


S = TypeVar("S", bound=cst.BaseStatement)
Targets = tuple[type[cst.BaseCompoundStatement], ...]

DEFAULT_TARGETS: Targets = (cst.If, cst.For, cst.While)
EXTRA_TARGETS: dict[str, Targets] = {
    "try": (cst.Try, cst.TryStar),
    "with": (cst.With,),
    "match": (cst.Match,),
}


class BlankLinesTransformer(cst.CSTTransformer):
    """Insert blank lines around the configured compound statements."""

    def __init__(self, targets: Targets) -> None:
        """Remember which statement types need surrounding blank lines."""
        super().__init__()
        self.targets = targets

    def leave_Module(  # noqa: N802
        self,
        original_node: cst.Module,  # noqa: ARG002
        updated_node: cst.Module,
    ) -> cst.Module:
        """Fix the top-level statements of a module."""
        return updated_node.with_changes(body=self.fix_body(updated_node.body))

    def leave_IndentedBlock(  # noqa: N802
        self,
        original_node: cst.IndentedBlock,  # noqa: ARG002
        updated_node: cst.IndentedBlock,
    ) -> cst.IndentedBlock:
        """Fix the statements of an indented block."""
        return updated_node.with_changes(body=self.fix_body(updated_node.body))

    def fix_body(self, body: Sequence[S]) -> list[S]:
        fixed: list[S] = []

        for index, statement in enumerate(body):
            needs_gap = index > 0 and (self.is_target(body[index - 1]) or self.is_target(statement))

            if index == 1 and self.is_docstring(body[0]):
                needs_gap = False

            fixed.append(self.add_blank_line_before(statement) if needs_gap else statement)

        return fixed

    @staticmethod
    def is_docstring(statement: cst.BaseStatement) -> bool:
        return (
            isinstance(statement, cst.SimpleStatementLine)
            and len(statement.body) == 1
            and isinstance(statement.body[0], cst.Expr)
            and isinstance(statement.body[0].value, cst.SimpleString | cst.ConcatenatedString)
        )

    def is_target(self, statement: cst.BaseStatement) -> bool:
        return isinstance(statement, self.targets)

    @staticmethod
    def add_blank_line_before(statement: S) -> S:
        if not isinstance(statement, cst.SimpleStatementLine | cst.BaseCompoundStatement):
            return statement

        leading = statement.leading_lines

        if leading and leading[0].comment is None:
            return statement  # already starts with a blank line

        blank = cst.EmptyLine(indent=False)

        return cast("S", statement.with_changes(leading_lines=[blank, *leading]))


def iter_python_files(paths: Sequence[Path]) -> list[Path]:
    """Expand directories into the ``*.py`` files they contain."""
    files: list[Path] = []

    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.py")))
        else:
            files.append(path)

    return files


def process_file(path: Path, transformer: cst.CSTTransformer, *, check: bool) -> bool:
    """Apply the transformer to one file. Return True if the file needed changes."""
    source = path.read_bytes()
    result = cst.parse_module(source).visit(transformer)

    if result.bytes == source:
        return False

    if not check:
        path.write_bytes(result.bytes)

    return True


def read_excluded_paths(path: Path | None) -> set[Path]:
    """Read exact exclusions relative to the baseline file, independent of cwd."""
    excluded: set[Path] = set()

    if path is None:
        return excluded

    lines = path.read_text(encoding="utf-8").splitlines()

    return {
        (path.parent / line.strip()).resolve()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the codemod over the given files and directories."""
    parser = argparse.ArgumentParser(description="Blank lines around if/for/while.")
    parser.add_argument("paths", nargs="+", type=Path, help="files or directories")
    parser.add_argument("--check", action="store_true", help="report only, do not write")
    parser.add_argument("--exclude-from", type=Path, help="baseline of exact paths relative to this file")
    parser.add_argument(
        "--also",
        action="append",
        choices=sorted(EXTRA_TARGETS),
        default=[],
        help="additional statement kinds to surround with blank lines",
    )
    args = parser.parse_args(argv)
    excluded = read_excluded_paths(args.exclude_from)

    targets = DEFAULT_TARGETS

    for name in args.also:
        targets += EXTRA_TARGETS[name]

    transformer = BlankLinesTransformer(targets)
    failed = False
    changed = False

    for path in iter_python_files(args.paths):
        if path.resolve() in excluded:
            continue

        try:
            if process_file(path, transformer, check=args.check):
                changed = True
                sys.stdout.write(f"{'would fix' if args.check else 'fixed'}: {path}\n")
        except cst.ParserSyntaxError as error:
            failed = True
            sys.stderr.write(f"syntax error in {path}: {error}\n")

    return 1 if changed or failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
