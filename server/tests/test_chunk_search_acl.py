"""ACL regression tests for the SQL translation in ChunkRepository.search_substring.

C-1 (AUDIT.md): the per-condition groups must be AND-clauses (visibility +
owner/group), the full visibility scope is the OR of all conditions, and
``document_id`` must narrow the result at the top level — never grant access.

The repository is exercised through its real query-building code path; the
captured SQLAlchemy statement is compiled and evaluated semantically against
in-memory "rows" (no database needed).
"""

from __future__ import annotations

import asyncio
import operator
import re
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import operators as sa_ops
from sqlalchemy.sql.elements import (
    BinaryExpression,
    BindParameter,
    BooleanClauseList,
    Grouping,
    Null,
    TextClause,
    True_,
)

from domain.identity.roles import UserKind, UserRole
from domain.identity.user_context import UserContext
from infrastructure.repositories.chunk.sqlalchemy_chunk_repository import (
    SQLAlchemyChunkRepository,
)


if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import date

    from sqlalchemy.sql import Select
    from sqlalchemy.sql.elements import (
        ClauseElement,
    )


# ---------------------------------------------------------------------------
# Statement capture
# ---------------------------------------------------------------------------


class EmptyResult:
    def all(self) -> list[tuple[Any, ...]]:
        return []


class CaptureSession:
    def __init__(self) -> None:
        self.captured: Select[Any] | None = None

    async def execute(self, stmt: Select[Any], *args: Any, **kwargs: Any) -> EmptyResult:
        del args, kwargs
        self.captured = stmt
        return EmptyResult()


async def capture(
    query: str,
    user: dict[str, Any],
    group_ids: list[int],
    mode: str,
    document_id: int | None,
    *,
    as_of_date: date | None = None,
) -> Select[Any]:

    session = CaptureSession()
    repo = SQLAlchemyChunkRepository(MagicMock(spec=AsyncSession, execute=AsyncMock(wraps=session.execute)))
    ctx = UserContext(
        user_id=user["id"],
        user_kind=UserKind(user["kind"]),
        user_role=UserRole(user.get("role", "user")),
        group_ids=tuple(group_ids),
    )
    await repo.search_substring(
        query, ctx, limit=20, mode=mode, document_id=document_id, as_of_date=as_of_date
    )
    assert session.captured is not None, "statement was not executed"
    return session.captured


# ---------------------------------------------------------------------------
# Clause tree -> semantic structure
# ---------------------------------------------------------------------------


def to_tree(clause: ClauseElement) -> tuple[Any, ...]:
    if isinstance(clause, Grouping):
        return to_tree(clause.element)

    if isinstance(clause, BooleanClauseList):
        if clause.operator is sa_ops.and_:
            op = "AND"
        elif clause.operator is sa_ops.or_:
            op = "OR"
        else:  # pragma: no cover - defensive
            msg = f"Unexpected boolean operator: {clause.operator}"
            raise AssertionError(msg)

        return (op, [to_tree(c) for c in clause.clauses])

    if isinstance(clause, BinaryExpression):
        return comparison_tree(clause)

    if isinstance(clause, TextClause):
        return ("regex",)

    msg = f"Unexpected clause type: {type(clause)}"
    raise AssertionError(msg)


def comparison_tree(clause: BinaryExpression[Any]) -> tuple[Any, ...]:
    col = clause.left.name
    right = clause.right
    value = right.value if isinstance(right, BindParameter) else right

    if isinstance(right, Null):
        value = None
    elif isinstance(right, True_):
        value = True

    comparisons = {
        sa_ops.is_: "is",
        sa_ops.is_not: "is_not",
        operator.le: "le",
        operator.gt: "gt",
        operator.eq: "eq",
    }

    if clause.operator in comparisons:
        return ("cmp", col, comparisons[clause.operator], value)

    if clause.operator is sa_ops.in_op:
        assert value is not None
        return ("cmp", col, "in", list(value))

    if clause.operator is sa_ops.ilike_op:
        return ("cmp", col, "ilike", str(value).strip("%"))

    if clause.operator is sa_ops.regexp_match_op:
        return ("cmp", col, "regexp", str(value).replace("[^[:alnum:]_]", r"[^\w]"))

    msg = f"Unexpected comparison operator: {clause.operator}"
    raise AssertionError(msg)


def eval_tree(node: tuple[Any, ...], row: dict[str, Any], pattern: str | None) -> bool:
    kind = node[0]

    if kind == "AND":
        return all(eval_tree(n, row, pattern) for n in node[1])

    if kind == "OR":
        return any(eval_tree(n, row, pattern) for n in node[1])

    if kind == "regex":
        # Postgres \y word boundary ≈ Python \b for these tests
        assert pattern is not None
        py_pattern = pattern.replace(r"\y", r"\b")
        return re.search(py_pattern, row["content"], re.IGNORECASE) is not None

    return evaluate_comparison(node, row)


def evaluate_comparison(node: tuple[Any, ...], row: dict[str, Any]) -> bool:
    _, col, op, value = node
    actual = row.get(col, True if col == "is_current" else None)
    operations: dict[str, Callable[[Any, Any], bool]] = {
        "is": lambda a, b: a is b,
        "is_not": lambda a, b: a is not b,
        "le": lambda a, b: a is not None and a <= b,
        "gt": lambda a, b: a is not None and a > b,
        "regexp": lambda a, b: re.search(b, str(a or ""), flags=re.IGNORECASE) is not None,
        "eq": lambda _a, b: bool(row.get(col) == b),
        "in": lambda _a, b: row.get(col) in b,
        "ilike": lambda _a, b: b.lower() in str(row.get(col, "")).lower(),
    }

    if op not in operations:
        msg = f"Unknown op {op}"
        raise AssertionError(msg)

    return operations[op](actual, value)


def where_tree(stmt: Select[Any]) -> tuple[Any, ...]:
    clause = stmt.whereclause
    assert clause is not None
    return to_tree(clause)


def search_pattern(stmt: Select[Any]) -> str:
    """Extract the search pattern from the ILIKE clause in the statement."""
    clause = stmt.whereclause

    if clause is None:
        return ""

    return pattern_from_clause(clause)


def pattern_from_clause(clause: ClauseElement) -> str:
    if isinstance(clause, Grouping):
        return pattern_from_clause(clause.element)

    if isinstance(clause, BooleanClauseList):
        return next(filter(None, map(pattern_from_clause, clause.clauses)), "")

    if isinstance(clause, BinaryExpression) and clause.operator is sa_ops.ilike_op:
        return str(getattr(clause.right, "value", "")).strip("%")

    return ""


def find_op(node: tuple[Any, ...], op: str) -> list[tuple[Any, ...]] | None:
    if node[0] == op:
        children: list[tuple[Any, ...]] = node[1]
        return children

    if node[0] in ("AND", "OR"):
        for child in node[1]:
            found = find_op(child, op)

            if found is not None:
                return found

    return None


def walk_contexts(
    node: tuple[Any, ...], under_or: bool = False, found: list[tuple[tuple[Any, ...], bool]] | None = None
) -> list[tuple[tuple[Any, ...], bool]]:
    """Collect (cmp_node, under_or) pairs.

    True in under_or means the cmp sits somewhere below an OR node
    (i.e. it could be matched on its own).
    """
    if found is None:
        found = []

    if node[0] == "cmp":
        found.append((node, under_or))
    elif node[0] in ("AND", "OR"):
        child_under_or = under_or or node[0] == "OR"

        for child in node[1]:
            walk_contexts(child, child_under_or, found)

    return found


def scope_of(top: tuple[Any, ...]) -> tuple[Any, ...]:
    """Return the visibility-scope subtree of the top-level WHERE."""
    children: list[tuple[Any, ...]] = top[1] if top[0] == "AND" else [top]

    for child in children:
        if child[0] == "OR":
            return child

        if child[0] == "AND" and any(c[0] == "cmp" and c[1] == "visibility" for c in child[1]):
            return child

        if child[0] == "cmp" and child[1] == "visibility":
            return child

    msg = "visibility scope not found in WHERE clause"
    raise AssertionError(msg)


def condition_parts(condition: tuple[Any, ...]) -> list[tuple[Any, ...]]:
    """Normalize a scope condition: bare cmp (single-part) or AND-group."""
    if condition[0] == "cmp":
        return [condition]

    if condition[0] == "AND":
        return [c for c in condition[1] if c[0] == "cmp"]

    msg = f"Condition must be AND-group or bare cmp, got {condition[0]}"
    raise AssertionError(msg)


# ---------------------------------------------------------------------------
# Compiled-SQL structure: AND inside conditions, OR between conditions,
# document_id at the top level.
# ---------------------------------------------------------------------------


class TestCompiledScopeStructure:
    def test_client_condition_is_and_of_visibility_and_owner(self) -> None:
        stmt = asyncio.run(capture("secret", {"id": 1, "kind": "client", "role": "user"}, [], "exact", None))
        top = where_tree(stmt)
        assert top[0] == "AND", "top-level WHERE must be an AND of content match and ACL scope"
        scope = scope_of(top)
        # Single condition, two parts (visibility + owner) — must be conjunctive.
        assert scope[0] == "AND", "visibility + owner parts must be AND (C-1 regression: OR)"
        parts = condition_parts(scope)
        assert {(p[1], p[3]) for p in parts} == {("visibility", "client_private"), ("owner_id", 1)}

    def test_client_condition_compiled_sql_uses_and(self) -> None:
        stmt = asyncio.run(capture("secret", {"id": 1, "kind": "client", "role": "user"}, [], "exact", None))
        sql = str(stmt.compile())
        assert "chunks.visibility = :visibility_1 AND chunks.owner_id = :owner_id_1" in sql
        assert "chunks.visibility = :visibility_1 OR" not in sql

    def test_internal_conditions_are_or_of_and_groups(self) -> None:
        stmt = asyncio.run(
            capture("secret", {"id": 1, "kind": "internal", "role": "user"}, [10], "icontains", None)
        )
        top = where_tree(stmt)
        scope = scope_of(top)
        assert scope[0] == "OR"
        assert len(scope[1]) == 3, "internal: public + private + group conditions"

        public = condition_parts(scope[1][0])
        assert (public[0][1], public[0][3]) == ("visibility", "internal_public")

        private = condition_parts(scope[1][1])
        assert {(p[1], p[3]) for p in private} == {
            ("visibility", "internal_private"),
            ("owner_id", 1),
        }

        group = condition_parts(scope[1][2])
        assert {(p[1], p[2]) for p in group} == {("visibility", "eq"), ("group_id", "in")}

    def test_document_id_is_top_level_and_not_inside_or(self) -> None:
        stmt = asyncio.run(capture("secret", {"id": 1, "kind": "client", "role": "user"}, [], "icontains", 5))
        top = where_tree(stmt)
        assert top[0] == "AND"
        contexts = walk_contexts(top)
        docid = [(n, under_or) for n, under_or in contexts if n[1] == "document_id"]
        assert docid, "document_id filter must be present in the WHERE clause"
        assert all(not under_or for _, under_or in docid), (
            "document_id must never appear under an OR — it would grant access instead of narrowing"
        )
        assert any(n[3] == 5 for n, _ in docid)


# ---------------------------------------------------------------------------
# Behavioral: foreign private chunks must never match.
# ---------------------------------------------------------------------------


class TestClientScope:
    def test_client_cannot_match_foreign_private_chunk(self) -> None:
        stmt = asyncio.run(
            capture("contract", {"id": 1, "kind": "client", "role": "user"}, [], "exact", None)
        )
        pattern = search_pattern(stmt)
        tree = where_tree(stmt)

        own = {
            "visibility": "client_private",
            "owner_id": 1,
            "group_id": None,
            "document_id": 9,
            "content": "my contract clause",
        }
        foreign = {
            "visibility": "client_private",
            "owner_id": 2,
            "group_id": None,
            "document_id": 9,
            "content": "their contract clause",
        }
        internal = {
            "visibility": "internal_public",
            "owner_id": None,
            "group_id": None,
            "document_id": 9,
            "content": "public contract",
        }

        assert eval_tree(tree, own, pattern) is True
        assert eval_tree(tree, foreign, pattern) is False
        assert eval_tree(tree, internal, pattern) is False

    def test_document_id_does_not_bypass_acl(self) -> None:
        stmt = asyncio.run(capture("secret", {"id": 1, "kind": "client", "role": "user"}, [], "icontains", 5))
        pattern = None
        tree = where_tree(stmt)

        foreign_doc5 = {
            "visibility": "client_private",
            "owner_id": 2,
            "group_id": None,
            "document_id": 5,
            "content": "top secret material",
        }
        own_other_doc = {
            "visibility": "client_private",
            "owner_id": 1,
            "group_id": None,
            "document_id": 7,
            "content": "my secret material",
        }

        assert eval_tree(tree, foreign_doc5, pattern) is False, (
            "document_id=5 must not grant access to foreign rows"
        )
        assert eval_tree(tree, own_other_doc, pattern) is False, "must not see rows of another document"
        own_doc5 = {**foreign_doc5, "owner_id": 1}
        assert eval_tree(tree, own_doc5, pattern) is True


class TestInternalScope:
    def test_internal_non_member_cannot_match_group_chunk(self) -> None:
        stmt = asyncio.run(
            capture("report", {"id": 1, "kind": "internal", "role": "user"}, [10], "icontains", None)
        )
        tree = where_tree(stmt)

        public = {
            "visibility": "internal_public",
            "owner_id": None,
            "group_id": None,
            "document_id": 3,
            "content": "annual report",
        }
        own_private = {
            "visibility": "internal_private",
            "owner_id": 1,
            "group_id": None,
            "document_id": 3,
            "content": "my report",
        }
        foreign_private = {
            "visibility": "internal_private",
            "owner_id": 2,
            "group_id": None,
            "document_id": 3,
            "content": "secret report",
        }
        member_group = {
            "visibility": "internal_group",
            "owner_id": None,
            "group_id": 10,
            "document_id": 3,
            "content": "team report",
        }
        foreign_group = {
            "visibility": "internal_group",
            "owner_id": None,
            "group_id": 11,
            "document_id": 3,
            "content": "other team report",
        }

        assert eval_tree(tree, public, None) is True
        assert eval_tree(tree, own_private, None) is True
        assert eval_tree(tree, foreign_private, None) is False
        assert eval_tree(tree, member_group, None) is True
        assert eval_tree(tree, foreign_group, None) is False

    def test_internal_with_document_id_narrows_but_never_grants(self) -> None:
        stmt = asyncio.run(
            capture("report", {"id": 1, "kind": "internal", "role": "user"}, [10], "icontains", 3)
        )
        tree = where_tree(stmt)

        foreign_private_doc3 = {
            "visibility": "internal_private",
            "owner_id": 2,
            "group_id": None,
            "document_id": 3,
            "content": "secret report",
        }
        public_other_doc = {
            "visibility": "internal_public",
            "owner_id": None,
            "group_id": None,
            "document_id": 4,
            "content": "annual report",
        }

        assert eval_tree(tree, foreign_private_doc3, None) is False
        assert eval_tree(tree, public_other_doc, None) is False, "document_id must not widen the scope"
