"""Differential ACL test: the domain visibility rules and their storage translations agree.

Random documents, users, groups and curator scopes are generated; for every user the
domain predicate is compared with the storage filter evaluated by a real engine:

* search:  ``is_in_search_scope()`` vs ``build_qdrant_filter()`` on a local Qdrant;
* listing: ``can_view_document()`` vs ``render_visibility_conditions(for_list=True)`` on SQLite.

The two modes are also tied together: search never exceeds listing, and the only
extra documents a listing shows are other clients' ``CLIENT_PRIVATE`` docs for ADMIN.
Documents deliberately include ACL combinations that ingestion would not produce
(e.g. a group on a private doc): a filter must agree with the domain on any payload.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from hypothesis import HealthCheck, given, settings, strategies as st
from qdrant_client import QdrantClient, models
from sqlalchemy import Integer, String, create_engine, or_, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from domain.identity.roles import UserKind, UserRole
from domain.identity.user_context import UserContext
from domain.knowledge.access_control import can_view_document, get_visibility_conditions, is_in_search_scope
from domain.knowledge.document import Document
from domain.knowledge.enums import DocumentVisibility
from infrastructure.repositories.sql_acl import render_visibility_conditions
from infrastructure.repositories.vector.acl import build_qdrant_filter


if TYPE_CHECKING:
    from collections.abc import Iterator


COLLECTION = "acl_differential"
# Small id universes force owners, groups and scopes to collide often.
USER_IDS = st.integers(min_value=1, max_value=6)
GROUP_IDS = st.integers(min_value=1, max_value=4)
ID_LISTS = st.lists(USER_IDS, max_size=3, unique=True)
GROUP_LISTS = st.lists(GROUP_IDS, max_size=3, unique=True)


class Base(DeclarativeBase):
    pass


class AclRow(Base):
    """Only the ACL columns of the documents table, as the listing query sees them."""

    __tablename__ = "acl_rows"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    visibility: Mapped[str] = mapped_column(String)
    owner_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    group_id: Mapped[int | None] = mapped_column(Integer, nullable=True)


@dataclass(frozen=True)
class Case:
    documents: list[Document]
    users: list[UserContext]


@st.composite
def documents(draw: st.DrawFn, doc_id: int) -> Document:
    return Document(
        id=doc_id,
        visibility=draw(st.sampled_from(DocumentVisibility)),
        owner_id=draw(st.none() | USER_IDS),
        group_id=draw(st.none() | GROUP_IDS),
    )


@st.composite
def users(draw: st.DrawFn) -> UserContext:
    # Every role is drawn with every kind, and every role gets a curator scope: the
    # rules must ignore a scope that does not apply instead of each side guessing.
    return UserContext(
        user_id=draw(USER_IDS),
        user_kind=draw(st.sampled_from(UserKind)),
        user_role=draw(st.sampled_from(UserRole)),
        group_ids=tuple(draw(GROUP_LISTS)),
        managed_client_ids=tuple(draw(ID_LISTS)),
        managed_internal_ids=tuple(draw(ID_LISTS)),
        managed_group_ids=tuple(draw(GROUP_LISTS)),
    )


@st.composite
def cases(draw: st.DrawFn) -> Case:
    count = draw(st.integers(min_value=1, max_value=25))
    docs = [draw(documents(doc_id)) for doc_id in range(1, count + 1)]
    return Case(documents=docs, users=draw(st.lists(users(), min_size=1, max_size=8)))


@contextmanager
def qdrant_with(docs: list[Document]) -> Iterator[QdrantClient]:
    client = QdrantClient(":memory:")
    try:
        client.create_collection(
            COLLECTION, vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE)
        )
        client.upsert(
            COLLECTION,
            [
                models.PointStruct(
                    id=doc.id,
                    vector=[1.0, 0.0],
                    payload={
                        "metadata": {
                            "document_id": doc.id,
                            "visibility": doc.visibility.value,
                            "owner_id": doc.owner_id,
                            "group_id": doc.group_id,
                        }
                    },
                )
                for doc in docs
                if doc.id is not None
            ],
        )
        yield client
    finally:
        client.close()


@contextmanager
def sqlite_with(docs: list[Document]) -> Iterator[Session]:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    try:
        with Session(engine) as session:
            session.add_all(
                AclRow(
                    id=doc.id, visibility=doc.visibility.value, owner_id=doc.owner_id, group_id=doc.group_id
                )
                for doc in docs
            )
            session.commit()
            yield session
    finally:
        engine.dispose()


def qdrant_search_ids(client: QdrantClient, ctx: UserContext, total: int) -> set[int]:
    """Mirror the production call in ``RagSteps.retrieval_work``."""
    scope_filter = build_qdrant_filter(
        ctx,
        list(ctx.group_ids),
        managed_client_ids=list(ctx.managed_client_ids),
        managed_internal_ids=list(ctx.managed_internal_ids),
        managed_group_ids=list(ctx.managed_group_ids),
    )
    points, _ = client.scroll(COLLECTION, scroll_filter=scope_filter, limit=total + 1, with_payload=False)
    return {int(point.id) for point in points}


def sql_listing_ids(session: Session, ctx: UserContext) -> set[int]:
    """Mirror the production query in ``SqlAlchemyDocumentRepository.list_visible``."""
    conditions = get_visibility_conditions(
        ctx.user_kind,
        list(ctx.group_ids),
        for_list=True,
        user_role=ctx.user_role,
        managed_client_ids=list(ctx.managed_client_ids),
        managed_internal_ids=list(ctx.managed_internal_ids),
        managed_group_ids=list(ctx.managed_group_ids),
    )
    clauses = render_visibility_conditions(
        conditions,
        visibility_column=AclRow.visibility,
        owner_column=AclRow.owner_id,
        group_column=AclRow.group_id,
        user_id=ctx.user_id,
        group_ids=list(ctx.group_ids),
    )

    if not clauses:
        return set()

    return set(session.scalars(select(AclRow.id).where(or_(*clauses))))


def describe(doc: Document, ctx: UserContext) -> str:
    return f"doc={doc!r}\nuser={ctx!r}"


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(cases())
def test_domain_scope_matches_qdrant_and_sql_filters(case: Case) -> None:
    by_id = {doc.id: doc for doc in case.documents}
    total = len(case.documents)

    with qdrant_with(case.documents) as client, sqlite_with(case.documents) as session:
        for ctx in case.users:
            searched = qdrant_search_ids(client, ctx, total)
            listed = sql_listing_ids(session, ctx)

            for doc_id, doc in by_id.items():
                in_search = is_in_search_scope(doc, ctx)
                in_list = can_view_document(doc, ctx)

                assert (doc_id in searched) == in_search, "search: domain vs Qdrant\n" + describe(doc, ctx)
                assert (doc_id in listed) == in_list, "listing: domain vs SQL\n" + describe(doc, ctx)
                assert not in_search or in_list, "search exceeds listing\n" + describe(doc, ctx)

                if in_list and not in_search:
                    # The single intended gap: ADMIN lists CLIENT_PRIVATE but never retrieves it.
                    assert ctx.user_kind == UserKind.INTERNAL, describe(doc, ctx)
                    assert ctx.user_role == UserRole.ADMIN, describe(doc, ctx)
                    assert doc.visibility == DocumentVisibility.CLIENT_PRIVATE, describe(doc, ctx)


def test_admin_listing_sees_client_private_but_search_does_not() -> None:
    """Anchor for the differential test: the listing/search gap is real, not vacuous."""
    docs = [
        Document(id=1, visibility=DocumentVisibility.CLIENT_PRIVATE, owner_id=5),
        Document(id=2, visibility=DocumentVisibility.INTERNAL_PRIVATE, owner_id=5),
    ]
    admin = UserContext(user_id=1, user_kind=UserKind.INTERNAL, user_role=UserRole.ADMIN)
    user = UserContext(user_id=1, user_kind=UserKind.INTERNAL, user_role=UserRole.USER)

    with qdrant_with(docs) as client, sqlite_with(docs) as session:
        assert sql_listing_ids(session, admin) == {1, 2}
        assert qdrant_search_ids(client, admin, len(docs)) == {2}
        assert sql_listing_ids(session, user) == set()
        assert qdrant_search_ids(client, user, len(docs)) == set()
