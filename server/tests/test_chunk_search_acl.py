"""ACL regression tests for the SQL translation in ChunkRepository.search_substring.

C-1 (AUDIT.md): the per-condition groups must be AND-clauses (visibility +
owner/group), the full visibility scope is the OR of all conditions, and
``document_id`` must narrow the result at the top level — never grant access.

The repository is exercised through its real query-building code path; the
captured SQLAlchemy statement is compiled and evaluated semantically against
in-memory "rows" (no database needed).
"""

import asyncio
import operator
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from sqlalchemy.sql import operators as sa_ops  # noqa: E402
from sqlalchemy.sql.elements import (  # noqa: E402
    BindParameter,
    BooleanClauseList,
    BinaryExpression,
    Grouping,
    TextClause,
    Null,
    True_,
)

from infrastructure.repositories.chunk.sqlalchemy_chunk_repository import (  # noqa: E402
    SQLAlchemyChunkRepository,
)

# ---------------------------------------------------------------------------
# Statement capture
# ---------------------------------------------------------------------------


class _EmptyResult:
    def all(self):
        return []


class _CaptureSession:
    def __init__(self):
        self.captured = None

    async def execute(self, stmt, *args, **kwargs):
        self.captured = stmt
        return _EmptyResult()


async def _capture(
    query: str, user: dict, group_ids: list[int], mode: str, document_id: int | None, as_of_date=None
):
    from domain.value_objects.user_context import UserContext

    session = _CaptureSession()
    repo = SQLAlchemyChunkRepository(session)
    ctx = UserContext(
        user_id=user["id"],
        user_kind=user["kind"],
        user_role=user.get("role", "user"),
        group_ids=group_ids,
    )
    await repo.search_substring(
        query, ctx, limit=20, mode=mode, document_id=document_id, as_of_date=as_of_date
    )
    assert session.captured is not None, "statement was not executed"
    return session.captured


# ---------------------------------------------------------------------------
# Clause tree -> semantic structure
# ---------------------------------------------------------------------------


def _to_tree(clause):
    if isinstance(clause, Grouping):
        return _to_tree(clause.element)
    if isinstance(clause, BooleanClauseList):
        if clause.operator is sa_ops.and_:
            op = "AND"
        elif clause.operator is sa_ops.or_:
            op = "OR"
        else:  # pragma: no cover - defensive
            raise AssertionError(f"Unexpected boolean operator: {clause.operator}")
        return (op, [_to_tree(c) for c in clause.clauses])
    if isinstance(clause, BinaryExpression):
        return _comparison_tree(clause)
    if isinstance(clause, TextClause):
        return ("regex",)
    raise AssertionError(f"Unexpected clause type: {type(clause)}")


def _comparison_tree(clause):
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
        return ("cmp", col, "in", list(value))
    if clause.operator is sa_ops.ilike_op:
        return ("cmp", col, "ilike", str(value).strip("%"))
    raise AssertionError(f"Unexpected comparison operator: {clause.operator}")


def _eval(node, row: dict, pattern: str | None) -> bool:
    kind = node[0]
    if kind == "AND":
        return all(_eval(n, row, pattern) for n in node[1])
    if kind == "OR":
        return any(_eval(n, row, pattern) for n in node[1])
    if kind == "regex":
        # Postgres \y word boundary ≈ Python \b for these tests
        py_pattern = pattern.replace(r"\y", r"\b")
        return re.search(py_pattern, row["content"], re.IGNORECASE) is not None
    _, col, op, value = node
    actual = row.get(col, True if col == "is_current" else None)
    temporal_operations = {
        "is": lambda a, b: a is b,
        "is_not": lambda a, b: a is not b,
        "le": lambda a, b: a is not None and a <= b,
        "gt": lambda a, b: a is not None and a > b,
    }
    if op in temporal_operations:
        return temporal_operations[op](actual, value)
    if op == "eq":
        return row.get(col) == value
    if op == "in":
        return row.get(col) in value
    if op == "ilike":
        return value.lower() in str(row.get(col, "")).lower()
    raise AssertionError(f"Unknown op {op}")


def _where(stmt):
    return _to_tree(stmt.whereclause)


def _pattern(stmt) -> str:
    """Extract the search pattern from the ILIKE clause in the statement."""
    clause = stmt.whereclause
    if clause is None:
        return ""
    return _pattern_from_clause(clause)


def _pattern_from_clause(clause) -> str:
    if isinstance(clause, Grouping):
        return _pattern_from_clause(clause.element)
    if isinstance(clause, BooleanClauseList):
        for c in clause.clauses:
            p = _pattern_from_clause(c)
            if p:
                return p
        return ""
    if isinstance(clause, BinaryExpression) and clause.operator is sa_ops.ilike_op:
        right = clause.right
        if isinstance(right, BindParameter):
            return str(right.value).strip("%")
        if hasattr(right, "value"):
            return str(right.value).strip("%")
    return ""


def _find_op(node, op: str) -> list | None:
    if node[0] == op:
        return node[1]
    if node[0] in ("AND", "OR"):
        for child in node[1]:
            found = _find_op(child, op)
            if found is not None:
                return found
    return None


def _walk_contexts(node, under_or: bool = False, found: list | None = None) -> list[tuple]:
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
            _walk_contexts(child, child_under_or, found)
    return found


def _scope_of(top):
    """Return the visibility-scope subtree of the top-level WHERE."""
    children = top[1] if top[0] == "AND" else [top]
    for child in children:
        if child[0] == "OR":
            return child
        if child[0] == "AND" and any(c[0] == "cmp" and c[1] == "visibility" for c in child[1]):
            return child
        if child[0] == "cmp" and child[1] == "visibility":
            return child
    raise AssertionError("visibility scope not found in WHERE clause")


def _condition_parts(condition) -> list[tuple]:
    """Normalize a scope condition: bare cmp (single-part) or AND-group."""
    if condition[0] == "cmp":
        return [condition]
    if condition[0] == "AND":
        return [c for c in condition[1] if c[0] == "cmp"]
    raise AssertionError(f"Condition must be AND-group or bare cmp, got {condition[0]}")


# ---------------------------------------------------------------------------
# Compiled-SQL structure: AND inside conditions, OR between conditions,
# document_id at the top level.
# ---------------------------------------------------------------------------


class TestCompiledScopeStructure:
    def test_client_condition_is_and_of_visibility_and_owner(self):
        stmt = asyncio.run(_capture("secret", {"id": 1, "kind": "client", "role": "user"}, [], "exact", None))
        top = _where(stmt)
        assert top[0] == "AND", "top-level WHERE must be an AND of content match and ACL scope"
        scope = _scope_of(top)
        # Single condition, two parts (visibility + owner) — must be conjunctive.
        assert scope[0] == "AND", "visibility + owner parts must be AND (C-1 regression: OR)"
        parts = _condition_parts(scope)
        assert {(p[1], p[3]) for p in parts} == {("visibility", "client_private"), ("owner_id", 1)}

    def test_client_condition_compiled_sql_uses_and(self):
        stmt = asyncio.run(_capture("secret", {"id": 1, "kind": "client", "role": "user"}, [], "exact", None))
        sql = str(stmt.compile())
        assert "chunks.visibility = :visibility_1 AND chunks.owner_id = :owner_id_1" in sql
        assert "chunks.visibility = :visibility_1 OR" not in sql

    def test_internal_conditions_are_or_of_and_groups(self):
        stmt = asyncio.run(
            _capture("secret", {"id": 1, "kind": "internal", "role": "user"}, [10], "icontains", None)
        )
        top = _where(stmt)
        scope = _scope_of(top)
        assert scope[0] == "OR"
        assert len(scope[1]) == 3, "internal: public + private + group conditions"

        public = _condition_parts(scope[1][0])
        assert (public[0][1], public[0][3]) == ("visibility", "internal_public")

        private = _condition_parts(scope[1][1])
        assert {(p[1], p[3]) for p in private} == {
            ("visibility", "internal_private"),
            ("owner_id", 1),
        }

        group = _condition_parts(scope[1][2])
        assert {(p[1], p[2]) for p in group} == {("visibility", "eq"), ("group_id", "in")}

    def test_document_id_is_top_level_and_not_inside_or(self):
        stmt = asyncio.run(
            _capture("secret", {"id": 1, "kind": "client", "role": "user"}, [], "icontains", 5)
        )
        top = _where(stmt)
        assert top[0] == "AND"
        contexts = _walk_contexts(top)
        docid = [(n, under_or) for n, under_or in contexts if n[1] == "document_id"]
        assert docid, "document_id filter must be present in the WHERE clause"
        assert all(
            not under_or for _, under_or in docid
        ), "document_id must never appear under an OR — it would grant access instead of narrowing"
        assert any(n[3] == 5 for n, _ in docid)


# ---------------------------------------------------------------------------
# Behavioral: foreign private chunks must never match.
# ---------------------------------------------------------------------------


class TestClientScope:
    def test_client_cannot_match_foreign_private_chunk(self):
        stmt = asyncio.run(
            _capture("contract", {"id": 1, "kind": "client", "role": "user"}, [], "exact", None)
        )
        pattern = _pattern(stmt)
        tree = _where(stmt)

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

        assert _eval(tree, own, pattern) is True
        assert _eval(tree, foreign, pattern) is False
        assert _eval(tree, internal, pattern) is False

    def test_document_id_does_not_bypass_acl(self):
        stmt = asyncio.run(
            _capture("secret", {"id": 1, "kind": "client", "role": "user"}, [], "icontains", 5)
        )
        pattern = None
        tree = _where(stmt)

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

        assert (
            _eval(tree, foreign_doc5, pattern) is False
        ), "document_id=5 must not grant access to foreign rows"
        assert _eval(tree, own_other_doc, pattern) is False, "must not see rows of another document"
        own_doc5 = {**foreign_doc5, "owner_id": 1}
        assert _eval(tree, own_doc5, pattern) is True


class TestInternalScope:
    def test_internal_non_member_cannot_match_group_chunk(self):
        stmt = asyncio.run(
            _capture("report", {"id": 1, "kind": "internal", "role": "user"}, [10], "icontains", None)
        )
        tree = _where(stmt)

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

        assert _eval(tree, public, None) is True
        assert _eval(tree, own_private, None) is True
        assert _eval(tree, foreign_private, None) is False
        assert _eval(tree, member_group, None) is True
        assert _eval(tree, foreign_group, None) is False

    def test_internal_with_document_id_narrows_but_never_grants(self):
        stmt = asyncio.run(
            _capture("report", {"id": 1, "kind": "internal", "role": "user"}, [10], "icontains", 3)
        )
        tree = _where(stmt)

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

        assert _eval(tree, foreign_private_doc3, None) is False
        assert _eval(tree, public_other_doc, None) is False, "document_id must not widen the scope"
