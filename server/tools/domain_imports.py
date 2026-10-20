"""Rewrite symbol imports using the domain move table; default is a read-only diff."""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING

import libcst as cst
from libcst.codemod import CodemodContext, VisitorBasedCodemodCommand
from libcst.codemod.visitors import AddImportsVisitor, RemoveImportsVisitor
from libcst.helpers import get_full_name_for_node
from libcst.metadata import ImportAssignment, QualifiedNameProvider, QualifiedNameSource, ScopeProvider


if TYPE_CHECKING:
    from collections.abc import Collection, Sequence
    from typing import ClassVar

    from libcst.metadata import BaseMetadataProvider


DEFAULT_MAPPING = Path(__file__).with_name("domain_moves.json")


class DomainImports(VisitorBasedCodemodCommand):
    """Move imported symbols without rewriting unrelated names or string literals."""

    METADATA_DEPENDENCIES: ClassVar[Collection[type[BaseMetadataProvider[object]]]] = (
        QualifiedNameProvider,
        ScopeProvider,
    )

    def __init__(self, context: CodemodContext, moves: dict[str, str]) -> None:
        super().__init__(context)
        self.moves = moves

    def destination(self, node: cst.CSTNode) -> str | None:
        names = self.get_metadata(QualifiedNameProvider, node, set())
        destinations = {
            self.moves[name.name]
            for name in names
            if name.source == QualifiedNameSource.IMPORT and name.name in self.moves
        }

        if len(destinations) > 1:
            message = "Ambiguous imported symbol; resolve its bindings before moving it"
            raise ValueError(message)

        return next(iter(destinations), None)

    def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.Name:  # noqa: N802
        target = self.destination(original_node)

        if target is None:
            return updated_node

        names = self.get_metadata(QualifiedNameProvider, original_node, set())

        if self.is_explicit_alias(original_node):
            return updated_node

        if any(name.name.rsplit(".", 1)[-1] == original_node.value for name in names):
            return updated_node.with_changes(value=target.rsplit(".", 1)[-1])

        return updated_node

    def is_explicit_alias(self, node: cst.Name) -> bool:
        scope = self.get_metadata(ScopeProvider, node, None)

        if scope is None:
            return False

        for assignment in scope[node.value]:
            if not isinstance(assignment, ImportAssignment):
                continue

            imported = assignment.node

            if (
                isinstance(imported, cst.ImportFrom)
                and not isinstance(imported.names, cst.ImportStar)
                and any(
                    alias.asname and get_full_name_for_node(alias.asname.name) == node.value
                    for alias in imported.names
                )
            ):
                return True

        return False

    def leave_Attribute(  # noqa: N802
        self, original_node: cst.Attribute, updated_node: cst.Attribute
    ) -> cst.BaseExpression:
        target = self.destination(original_node)

        if target is None:
            return updated_node

        module, _symbol = target.rsplit(".", 1)
        AddImportsVisitor.add_needed_import(self.context, module)
        RemoveImportsVisitor.remove_unused_import_by_node(self.context, original_node.value)
        return cst.parse_expression(target)

    def leave_SimpleStatementLine(  # noqa: N802
        self, original_node: cst.SimpleStatementLine, updated_node: cst.SimpleStatementLine
    ) -> cst.SimpleStatementLine:
        statements: list[cst.BaseSmallStatement] = []

        for original, updated in zip(original_node.body, updated_node.body, strict=True):
            if not isinstance(original, cst.ImportFrom) or not isinstance(updated, cst.ImportFrom):
                statements.append(updated)
                continue

            statements.extend(self.move_import(original, updated))

        return updated_node.with_changes(body=statements)

    def move_import(self, original: cst.ImportFrom, updated: cst.ImportFrom) -> list[cst.ImportFrom]:
        module = get_full_name_for_node(original.module) if original.module is not None else None

        if original.relative or module is None:
            return [updated]

        if isinstance(original.names, cst.ImportStar):
            if any(key.startswith(module + ".") for key in self.moves):
                message = f"Expand star import from {module} before moving it"
                raise ValueError(message)

            return [updated]

        groups: dict[str, list[cst.ImportAlias]] = {}

        for alias in original.names:
            symbol = get_full_name_for_node(alias.name)
            target = self.moves.get(f"{module}.{symbol}", f"{module}.{symbol}")
            new_module, new_symbol = target.rsplit(".", 1)
            groups.setdefault(new_module, []).append(alias.with_changes(name=cst.Name(new_symbol)))

        statements: list[cst.ImportFrom] = []

        for new_module, aliases in groups.items():
            if original.rpar is None:
                aliases[-1] = aliases[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)

            statements.append(updated.with_changes(module=cst.parse_expression(new_module), names=aliases))

        return statements


def rewrite(source: str, moves: dict[str, str]) -> str:
    return DomainImports(CodemodContext(), moves).transform_module(cst.parse_module(source)).code


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--context", help="Only destinations in this domain context")
    parser.add_argument("--write", action="store_true", help="Apply the displayed changes")
    args = parser.parse_args(argv)
    moves = json.loads(args.mapping.read_text(encoding="utf-8"))

    if args.context:
        moves = {old: new for old, new in moves.items() if new.startswith(f"domain.{args.context}.")}

        if not moves:
            parser.error(f"No destinations for context {args.context}")

    files = sorted(
        {file for path in args.paths for file in (path.rglob("*.py") if path.is_dir() else [path])}
    )
    changes: list[tuple[Path, str, str]] = []

    for path in files:
        source = path.read_bytes().decode("utf-8")
        rewritten = rewrite(source, moves)

        if rewritten != source:
            changes.append((path, source, rewritten))

    # Parse the entire selection before writing, so an unsupported import cannot cause a partial run.
    for path, source, rewritten in changes:
        if args.write:
            path.write_bytes(rewritten.encode("utf-8"))
        else:
            sys.stdout.writelines(
                difflib.unified_diff(
                    source.splitlines(keepends=True),
                    rewritten.splitlines(keepends=True),
                    fromfile=str(path),
                    tofile=str(path),
                )
            )

    return int(bool(changes) and not args.write)


if __name__ == "__main__":
    raise SystemExit(main())
