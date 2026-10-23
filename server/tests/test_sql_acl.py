"""Exercise both SQL ACL consumers against allowed and forbidden rows."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine, or_, select

from domain.identity.roles import UserKind, UserRole
from domain.identity.user_context import UserContext
from infrastructure.database.models import ChunkModel, DocumentModel
from infrastructure.repositories.chunk.chunk_search import build_acl_clauses
from infrastructure.repositories.document.sqlalchemy_document_repository import SQLAlchemyDocumentRepository


@pytest.mark.asyncio
@pytest.mark.parametrize("for_list", [True, False])
@pytest.mark.parametrize(
    "kind,role,scope,expected",
    [
        ("client", "user", {}, {7}),
        ("internal", "user", {}, {1, 2, 4}),
        ("internal", "admin", {}, {1, 2, 3, 4, 5, 6}),
        (
            "internal",
            "curator",
            {"managed_client_ids": [20], "managed_internal_ids": [20], "managed_group_ids": [11]},
            {1, 2, 3, 4, 5, 8},
        ),
    ],
)
async def test_sql_consumers_preserve_scope(
    kind: str, role: str, scope: dict[str, list[int]], expected: set[int], for_list: bool
) -> None:
    user = UserContext(
        user_id=10,
        user_kind=UserKind(kind),
        user_role=UserRole(role),
        group_ids=(10,),
        managed_client_ids=tuple(scope.get("managed_client_ids", [])),
        managed_internal_ids=tuple(scope.get("managed_internal_ids", [])),
        managed_group_ids=tuple(scope.get("managed_group_ids", [])),
    )
    model: type[DocumentModel | ChunkModel]

    if for_list:
        session = MagicMock()
        session.execute = AsyncMock(return_value=MagicMock())
        session.execute.return_value.scalars.return_value.all.return_value = []
        await SQLAlchemyDocumentRepository(session).list_visible(
            kind,
            10,
            [10],
            user_role=role,
            managed_client_ids=scope.get("managed_client_ids"),
            managed_internal_ids=scope.get("managed_internal_ids"),
            managed_group_ids=scope.get("managed_group_ids"),
        )
        predicate = session.execute.call_args.args[0].whereclause
        model = DocumentModel
    else:
        predicate = or_(
            *build_acl_clauses(
                user,
                [10],
                list(user.managed_client_ids),
                list(user.managed_internal_ids),
                list(user.managed_group_ids),
            )
        )
        model = ChunkModel

    engine = create_engine("sqlite://")
    table = Table(
        model.__tablename__,
        MetaData(),
        Column("id", Integer, primary_key=True),
        Column("visibility", String),
        Column("owner_id", Integer),
        Column("group_id", Integer),
    )
    rows = [
        (1, "internal_public", None, None),
        (2, "internal_private", 10, None),
        (3, "internal_private", 20, None),
        (4, "internal_group", None, 10),
        (5, "internal_group", None, 11),
        (6, "internal_group", None, 12),
        (7, "client_private", 10, None),
        (8, "client_private", 20, None),
        (9, "client_private", 30, None),
    ]
    try:
        with engine.begin() as connection:
            table.create(connection)
            connection.execute(
                table.insert(),
                [dict(zip(("id", "visibility", "owner_id", "group_id"), row, strict=True)) for row in rows],
            )
            actual = set(connection.scalars(select(model.id).where(predicate)))

        if role == "admin" and for_list:
            expected = expected | {7, 8, 9}

        assert actual == expected
    finally:
        engine.dispose()
