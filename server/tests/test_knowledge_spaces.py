"""Knowledge spaces select documents and pack rules; filling is limited to admitted groups.

A space is a selection, never an access condition: search, sources and cache stay inside
the question's space, while visibility alone decides what a user may read.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock

import pydantic
import pytest
from qdrant_client.local.payload_filters import check_filter
from sqlalchemy.dialects import postgresql

from application.chat import ChatService
from application.documents.space_processor import SpaceDocumentProcessor
from application.identity import UserContextFactory
from application.ingestion.sync import DocumentSyncService
from application.rag.dto import PreparedQuery
from application.rag.space_router import SpaceRagRouter, SpaceRequiredError
from application.spaces import (
    DomainPackNotLoadedError,
    KnowledgeSpaceService,
    SpaceChanges,
    resolve_space,
)
from domain.identity.roles import UserKind, UserRole
from domain.identity.user_context import UserContext
from domain.knowledge.document import Document
from domain.knowledge.enums import DocumentVisibility
from domain.knowledge.space import KnowledgeSpace, SpaceRef
from domain.rag.context import RagContext
from domain.rag.result import RagResult
from domain.shared.exceptions import (
    BusinessRuleViolationError,
    EntityNotFoundError,
    PermissionDeniedError,
    ValidationError,
)
from infrastructure.ml.cache.keys import CacheKeyBuilder
from infrastructure.ml.evidence_profiles.profiles import GenericEvidenceProfile, evidence_profile_scope
from infrastructure.ml.rag.rag_steps import RagOperations
from infrastructure.repositories.chunk.chunk_search import build_substring_search_stmt
from infrastructure.repositories.vector.acl import build_qdrant_filter, build_search_filter
from presentation.api.schemas.spaces import UpdateKnowledgeSpaceRequest
from tests.fakes import (
    DEFAULT_SPACE_ID,
    UPLOADERS_GROUP_ID,
    FakeChatRAGPort,
    FakePackCatalog,
    FakeUnitOfWork,
    FakeUnitOfWorkFactory,
    admit_all_users,
)
from tests.test_document_commands_characterization import harness, seed, upload


if TYPE_CHECKING:
    from application.documents.dto import ProcessDocumentCommand
    from application.ingestion.dto import RegistryRecord

SUPPORT_SPACE_ID = 2
ADMIN = UserContext(1, UserKind.INTERNAL, UserRole.ADMIN)


def add_space(uow: FakeUnitOfWork, space_id: int, pack_key: str = "generic", **kwargs: Any) -> None:
    uow.fake_knowledge_spaces._spaces[space_id] = KnowledgeSpace(
        id=space_id, name=f"Пространство {space_id}", pack_key=pack_key, **kwargs
    )


# --- Domain ---------------------------------------------------------------


@pytest.mark.parametrize(
    "user, admitted, allowed",
    [
        pytest.param(ADMIN, frozenset(), True, id="admin-fills-any-space"),
        pytest.param(
            UserContext(2, UserKind.INTERNAL, UserRole.CURATOR, group_ids=(5,)), {5}, True, id="member"
        ),
        pytest.param(
            UserContext(2, UserKind.INTERNAL, UserRole.CURATOR, group_ids=(6,)), {5}, False, id="other"
        ),
        pytest.param(
            UserContext(2, UserKind.INTERNAL, UserRole.CURATOR, group_ids=(5,)), set(), False, id="empty"
        ),
        pytest.param(UserContext(9, UserKind.CLIENT, UserRole.USER), {5}, False, id="client"),
    ],
)
def test_filling_a_space_requires_an_admitted_group(
    user: UserContext, admitted: set[int], allowed: bool
) -> None:
    space = KnowledgeSpace(name="Поддержка", pack_key="generic", upload_group_ids=frozenset(admitted))
    assert space.can_be_filled_by(user) is allowed


def test_space_name_is_required() -> None:
    with pytest.raises(ValidationError):
        KnowledgeSpace(name="  ", pack_key="generic")


def test_space_pack_cannot_be_changed_through_the_api() -> None:
    with pytest.raises(pydantic.ValidationError):
        UpdateKnowledgeSpaceRequest.model_validate({"pack_key": "belarus_law"})


# --- Resolution -----------------------------------------------------------


@pytest.mark.asyncio
async def test_the_only_space_is_used_without_an_explicit_one() -> None:
    uow = FakeUnitOfWork()
    assert (await resolve_space(uow, None)).id == DEFAULT_SPACE_ID


@pytest.mark.asyncio
async def test_several_spaces_require_an_explicit_choice() -> None:
    uow = FakeUnitOfWork()
    add_space(uow, SUPPORT_SPACE_ID)

    with pytest.raises(ValidationError):
        await resolve_space(uow, None)

    assert (await resolve_space(uow, SUPPORT_SPACE_ID)).id == SUPPORT_SPACE_ID

    with pytest.raises(EntityNotFoundError):
        await resolve_space(uow, 404)


# --- Management -----------------------------------------------------------


@pytest.mark.asyncio
async def test_space_uses_only_a_loaded_pack() -> None:
    service = KnowledgeSpaceService(FakeUnitOfWorkFactory(), FakePackCatalog())

    with pytest.raises(DomainPackNotLoadedError):
        await service.create(ADMIN, "Право", "belarus_law")

    created = await service.create(ADMIN, "Поддержка", "generic")
    assert created.pack_key == "generic"
    assert created.can_upload is True


@pytest.mark.asyncio
async def test_space_admits_only_existing_groups() -> None:
    uow = FakeUnitOfWork()
    uow.fake_groups.list_by_ids = AsyncMock(return_value=[SimpleNamespace(id=5, name="Поддержка")])  # type: ignore[method-assign]
    service = KnowledgeSpaceService(FakeUnitOfWorkFactory(uow), FakePackCatalog())

    with pytest.raises(ValidationError):
        await service.update(ADMIN, DEFAULT_SPACE_ID, SpaceChanges(upload_group_ids=[5, 6]))

    updated = await service.update(
        ADMIN, DEFAULT_SPACE_ID, SpaceChanges(name="Общая база", upload_group_ids=[5])
    )
    assert updated.name == "Общая база"
    assert updated.upload_group_ids == (5,)
    stored = await uow.knowledge_spaces.get_by_id(DEFAULT_SPACE_ID)
    assert stored is not None
    assert stored.pack_key == "generic"


@pytest.mark.asyncio
async def test_space_with_documents_cannot_be_deleted() -> None:
    uow = FakeUnitOfWork()
    await uow.documents.save(Document(filename="a.txt"))
    service = KnowledgeSpaceService(FakeUnitOfWorkFactory(uow), FakePackCatalog())

    with pytest.raises(BusinessRuleViolationError):
        await service.delete(DEFAULT_SPACE_ID)

    add_space(uow, SUPPORT_SPACE_ID)
    await service.delete(SUPPORT_SPACE_ID)
    assert await uow.knowledge_spaces.get_by_id(SUPPORT_SPACE_ID) is None


@pytest.mark.asyncio
async def test_admitted_groups_are_shown_only_to_admin() -> None:
    uow = FakeUnitOfWork()
    admit_all_users(uow)
    service = KnowledgeSpaceService(FakeUnitOfWorkFactory(uow), FakePackCatalog())

    [for_user] = await service.list_for_user(2, UserKind.INTERNAL, UserRole.CURATOR)
    [for_admin] = await service.list_for_user(1, UserKind.INTERNAL, UserRole.ADMIN)

    assert for_user.upload_group_ids is None
    assert for_user.can_upload is True
    assert for_admin.upload_group_ids == (UPLOADERS_GROUP_ID,)


# --- Filling a space (upload, replacement, manual documents, ingestion) -----


@pytest.mark.asyncio
async def test_upload_without_fill_right_is_rejected_before_storage() -> None:
    h = harness()
    h.uow.fake_groups.member_of = ()

    with pytest.raises(PermissionDeniedError):
        await upload(h, user_role="curator")

    h.storage.upload_file.assert_not_awaited()
    assert h.uow.background_jobs.jobs == {}


@pytest.mark.asyncio
async def test_upload_fills_the_chosen_space() -> None:
    h = harness()
    add_space(h.uow, SUPPORT_SPACE_ID, upload_group_ids=frozenset({UPLOADERS_GROUP_ID}))

    dto = await upload(h, user_role="curator", space_id=SUPPORT_SPACE_ID)

    assert dto.space_id == SUPPORT_SPACE_ID


@pytest.mark.asyncio
async def test_replacement_requires_the_fill_right() -> None:
    h = harness()
    old = await seed(h)
    h.uow.fake_groups.member_of = ()

    with pytest.raises(PermissionDeniedError):
        await upload(h, replaces_document_id=old.id, user_role="curator")

    assert await h.uow.documents.get_by_id(old.id) is not None
    h.storage.upload_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_replacement_stays_in_the_document_space() -> None:
    h = harness()
    add_space(h.uow, SUPPORT_SPACE_ID, upload_group_ids=frozenset({UPLOADERS_GROUP_ID}))
    old = await seed(h)

    with pytest.raises(BusinessRuleViolationError, match="same knowledge space"):
        await upload(h, replaces_document_id=old.id, space_id=SUPPORT_SPACE_ID)

    h.storage.upload_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_same_filename_in_another_space_is_a_new_document() -> None:
    h = harness()
    add_space(h.uow, SUPPORT_SPACE_ID, upload_group_ids=frozenset({UPLOADERS_GROUP_ID}))
    old = await seed(h)

    dto = await upload(h, space_id=SUPPORT_SPACE_ID)

    assert dto.filename == old.filename
    assert dto.replace_id is None
    assert dto.space_id == SUPPORT_SPACE_ID


@pytest.mark.asyncio
async def test_explicit_profile_must_belong_to_the_space_pack() -> None:
    h = harness()

    with pytest.raises(ValidationError):
        await upload(h, doc_domain="legal")

    assert (await upload(h, doc_domain="general")).doc_domain == "general"


def record(filename: str) -> RegistryRecord:
    return {"filename": filename, "hash": "h", "source": f"docs/{filename}", "chunks": 1, "chars": 1}


@pytest.mark.asyncio
async def test_storage_key_belongs_to_the_space_it_was_ingested_into() -> None:
    uow = FakeUnitOfWork()
    add_space(uow, SUPPORT_SPACE_ID)
    factory = FakeUnitOfWorkFactory(uow)
    await uow.documents.save(
        Document(filename="a.txt", ingestion_key="docs/a.txt", space_id=DEFAULT_SPACE_ID)
    )
    sync = DocumentSyncService(factory)

    with pytest.raises(BusinessRuleViolationError, match="already ingested"):
        await sync.ensure_document_rows({"docs/a.txt": record("a.txt")}, space_id=SUPPORT_SPACE_ID)

    rows = await sync.ensure_document_rows({"docs/b.txt": record("b.txt")}, space_id=SUPPORT_SPACE_ID)
    created = await uow.documents.get_by_id(rows["docs/b.txt"][0])
    assert created is not None
    assert created.space_id == SUPPORT_SPACE_ID


# --- Search and cache isolation --------------------------------------------


def point(space_id: int, visibility: str = DocumentVisibility.INTERNAL_PUBLIC.value) -> dict[str, Any]:
    return {"page_content": "text", "metadata": {"visibility": visibility, "space_id": space_id}}


@pytest.mark.parametrize("role", [UserRole.USER, UserRole.ADMIN, UserRole.CURATOR])
def test_qdrant_search_keeps_documents_of_other_spaces_out(role: UserRole) -> None:
    user = UserContext(1, UserKind.INTERNAL, role)
    search = build_search_filter(user, [], space_id=DEFAULT_SPACE_ID)

    assert check_filter(search.model_copy(deep=True), point(DEFAULT_SPACE_ID), 1)
    assert not check_filter(search.model_copy(deep=True), point(SUPPORT_SPACE_ID), 2)


@pytest.mark.parametrize("role", [UserRole.USER, UserRole.ADMIN, UserRole.CURATOR])
def test_space_does_not_widen_or_narrow_reading(role: UserRole) -> None:
    """Inside its space a point is readable exactly when the ACL filter alone allows it."""
    user = UserContext(1, UserKind.INTERNAL, role, group_ids=(4,))
    access = build_qdrant_filter(user, [4])
    search = build_search_filter(user, [4], space_id=DEFAULT_SPACE_ID)

    for visibility in DocumentVisibility:
        payload = point(DEFAULT_SPACE_ID, visibility.value)
        payload["metadata"].update({"owner_id": 1, "group_id": 4})
        assert check_filter(search.model_copy(deep=True), payload, 1) == check_filter(
            access.model_copy(deep=True), payload, 1
        )


def test_client_search_also_stays_in_its_space() -> None:
    client = UserContext(9, UserKind.CLIENT, UserRole.USER)
    search = build_search_filter(client, [], space_id=DEFAULT_SPACE_ID)
    own = point(SUPPORT_SPACE_ID, DocumentVisibility.CLIENT_PRIVATE.value)
    own["metadata"]["owner_id"] = 9

    assert not check_filter(search.model_copy(deep=True), own, 1)


def test_sql_search_selects_the_space_with_acl() -> None:
    stmt = build_substring_search_stmt(
        "rules", UserContext(1, UserKind.INTERNAL), "icontains", None, None, space_id=SUPPORT_SPACE_ID
    )
    sql = str(stmt.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))  # type: ignore[no-untyped-call]  # SQLAlchemy exposes an untyped dialect constructor.

    assert f"chunks.space_id = {SUPPORT_SPACE_ID}" in sql
    assert "chunks.visibility" in sql


@pytest.mark.asyncio
async def test_rag_retrieval_filter_selects_the_question_space() -> None:
    operations = RagOperations(MagicMock(), document_access=MagicMock())
    ctx = RagContext(UserContext(1, UserKind.INTERNAL), space=SpaceRef(SUPPORT_SPACE_ID, "generic"))

    with evidence_profile_scope(GenericEvidenceProfile()):
        work = operations.retrieval_work(PreparedQuery(operations.request("Q", [], ctx), "Q"))

    retrieval = cast("Any", work.retrieval_filter)
    assert check_filter(retrieval.model_copy(deep=True), point(SUPPORT_SPACE_ID), 1)
    assert not check_filter(retrieval.model_copy(deep=True), point(DEFAULT_SPACE_ID), 2)


@pytest.mark.asyncio
async def test_answer_cache_never_crosses_spaces() -> None:
    operations = RagOperations(MagicMock(), document_access=MagicMock())
    user = UserContext(1, UserKind.INTERNAL)

    def keys(space: SpaceRef) -> tuple[str, str]:
        request = operations.request("Q", [], RagContext(user, space=space))
        return CacheKeyBuilder.build(PreparedQuery(request, "Q"))

    with evidence_profile_scope(GenericEvidenceProfile()):
        general = keys(SpaceRef(DEFAULT_SPACE_ID, "generic"))
        support = keys(SpaceRef(SUPPORT_SPACE_ID, "generic"))
        again = keys(SpaceRef(DEFAULT_SPACE_ID, "generic"))

    assert general[0] != support[0]
    assert general == again
    # Visibility scope is unchanged: the space is a selection, not an access condition.
    assert general[1] == support[1]


# --- Pack routing ------------------------------------------------------------


@pytest.mark.asyncio
async def test_question_is_answered_by_its_space_pack() -> None:
    answers = {"generic": RagResult(answer="generic"), "law": RagResult(answer="law")}
    generic, law = MagicMock(), MagicMock()
    generic.invoke = AsyncMock(return_value=answers["generic"])
    law.invoke = AsyncMock(return_value=answers["law"])
    router = SpaceRagRouter({"generic": generic, "law": law})
    user = UserContext(1, UserKind.INTERNAL)

    assert await router.invoke("Q", [], RagContext(user, space=SpaceRef(3, "law"))) is answers["law"]
    generic.invoke.assert_not_awaited()

    with pytest.raises(SpaceRequiredError):
        await router.invoke("Q", [], RagContext(user))

    with pytest.raises(DomainPackNotLoadedError):
        await router.invoke("Q", [], RagContext(user, space=SpaceRef(3, "missing")))


@pytest.mark.asyncio
async def test_document_is_processed_by_its_space_pack() -> None:
    uow = FakeUnitOfWork()
    add_space(uow, SUPPORT_SPACE_ID, pack_key="law")
    document = await uow.documents.save(Document(filename="a.txt", space_id=SUPPORT_SPACE_ID))
    processed: list[str] = []

    class Recording:
        def __init__(self, key: str) -> None:
            self.key = key

        async def process(self, command: ProcessDocumentCommand) -> None:
            del command
            processed.append(self.key)

    router = SpaceDocumentProcessor(
        FakeUnitOfWorkFactory(uow), {"generic": Recording("generic"), "law": Recording("law")}
    )
    await router.process(cast("ProcessDocumentCommand", SimpleNamespace(document_id=document.id)))
    assert processed == ["law"]

    other = await uow.documents.save(Document(filename="b.txt", space_id=SUPPORT_SPACE_ID))
    with pytest.raises(DomainPackNotLoadedError):
        await SpaceDocumentProcessor(FakeUnitOfWorkFactory(uow), {"generic": Recording("generic")}).process(
            cast("ProcessDocumentCommand", SimpleNamespace(document_id=other.id))
        )


# --- Chat ----------------------------------------------------------------------


def chat_service(uow: FakeUnitOfWork, rag: FakeChatRAGPort) -> ChatService:
    return ChatService(
        uow_factory=FakeUnitOfWorkFactory(uow),
        rag_service=rag,
        chat_settings=SimpleNamespace(
            history_window=8, rolling_summary_enabled=False, curator_scope_max_ids=100
        ),
        chat_log_service=MagicMock(create=AsyncMock()),
        conversation_service=MagicMock(update_rolling_summary=AsyncMock()),
        pii_redactor=MagicMock(redact=lambda text: text),
        user_ctx_factory=UserContextFactory(),
    )


@pytest.mark.asyncio
async def test_conversation_keeps_the_space_of_its_first_question() -> None:
    uow = FakeUnitOfWork()
    add_space(uow, SUPPORT_SPACE_ID)
    service = chat_service(uow, FakeChatRAGPort())

    setup = await service.prepare_chat(
        "Q", None, 1, UserKind.INTERNAL, UserRole.USER, depth=None, as_of_date=None, space_id=SUPPORT_SPACE_ID
    )
    assert setup.ctx.space == SpaceRef(SUPPORT_SPACE_ID, "generic")
    assert setup.conv.id is not None

    follow_up = await service.prepare_chat(
        "Q2", setup.conv.id, 1, UserKind.INTERNAL, UserRole.USER, depth=None, as_of_date=None
    )
    assert follow_up.ctx.space == SpaceRef(SUPPORT_SPACE_ID, "generic")

    with pytest.raises(ValidationError):
        await service.prepare_chat(
            "Q3",
            setup.conv.id,
            1,
            UserKind.INTERNAL,
            UserRole.USER,
            depth=None,
            as_of_date=None,
            space_id=DEFAULT_SPACE_ID,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("same_space", [False, True])
async def test_concurrent_first_questions_use_one_conversation_space(same_space: bool) -> None:
    uow = FakeUnitOfWork()
    add_space(uow, SUPPORT_SPACE_ID)
    conversation = await uow.conversations.create(1)
    assert conversation.id is not None
    # Both requests have read the unbound conversation before either writes.
    snapshots = [replace(conversation), replace(conversation)]
    choices = [DEFAULT_SPACE_ID, DEFAULT_SPACE_ID if same_space else SUPPORT_SPACE_ID]
    results = await asyncio.gather(
        *(
            ChatService.conversation_space(uow, conv, choice)
            for conv, choice in zip(snapshots, choices, strict=True)
        ),
        return_exceptions=True,
    )
    stored = await uow.conversations.get_by_id(conversation.id)
    assert stored is not None
    assert stored.space_id == DEFAULT_SPACE_ID
    assert isinstance(results[0], KnowledgeSpace)
    assert results[0].id == stored.space_id

    if same_space:
        assert isinstance(results[1], KnowledgeSpace)
        assert results[1].id == stored.space_id
    else:
        assert isinstance(results[1], ValidationError)


@pytest.mark.asyncio
async def test_admin_assigns_client_and_listing_only_returns_assigned_space() -> None:
    uow = FakeUnitOfWork()
    add_space(uow, SUPPORT_SPACE_ID)
    uow.fake_users.add_user(9, kind=UserKind.CLIENT)
    service = KnowledgeSpaceService(FakeUnitOfWorkFactory(uow), FakePackCatalog())
    assert await service.list_for_user(9, UserKind.CLIENT, UserRole.USER) == []
    await service.assign_client(ADMIN, 9, SUPPORT_SPACE_ID)
    [assigned] = await service.list_for_user(9, UserKind.CLIENT, UserRole.USER)
    assert assigned.id == SUPPORT_SPACE_ID
    assert not assigned.can_upload
    await service.assign_client(ADMIN, 9, None)
    assert await service.list_for_user(9, UserKind.CLIENT, UserRole.USER) == []


@pytest.mark.asyncio
async def test_client_assignment_rejects_non_admin_non_client_and_unknown_ids() -> None:
    uow = FakeUnitOfWork()
    uow.fake_users.add_user(9, kind=UserKind.CLIENT)
    uow.fake_users.add_user(10)
    service = KnowledgeSpaceService(FakeUnitOfWorkFactory(uow), FakePackCatalog())
    with pytest.raises(BusinessRuleViolationError):
        await service.assign_client(UserContext(2, UserKind.INTERNAL, UserRole.CURATOR), 9, DEFAULT_SPACE_ID)
    with pytest.raises(BusinessRuleViolationError):
        await service.assign_client(ADMIN, 10, DEFAULT_SPACE_ID)
    with pytest.raises(EntityNotFoundError):
        await service.assign_client(ADMIN, 404, DEFAULT_SPACE_ID)
    with pytest.raises(EntityNotFoundError):
        await service.assign_client(ADMIN, 9, 404)


@pytest.mark.asyncio
async def test_client_chat_uses_current_assignment_and_rejects_old_conversation_after_reassignment() -> None:
    uow = FakeUnitOfWork()
    add_space(uow, SUPPORT_SPACE_ID)
    uow.fake_users.add_user(9, kind=UserKind.CLIENT)
    spaces = KnowledgeSpaceService(FakeUnitOfWorkFactory(uow), FakePackCatalog())
    service = chat_service(uow, FakeChatRAGPort())

    async def ask(conversation_id: int | None = None, space_id: int | None = None) -> Any:
        return await service.prepare_chat(
            "Q",
            conversation_id,
            9,
            UserKind.CLIENT,
            UserRole.USER,
            depth=None,
            as_of_date=None,
            space_id=space_id,
        )

    with pytest.raises(ValidationError):
        await ask()
    await spaces.assign_client(ADMIN, 9, SUPPORT_SPACE_ID)
    setup = await ask()
    assert setup.ctx.space == SpaceRef(SUPPORT_SPACE_ID, "generic")
    assert (await ask(setup.conv.id)).ctx.space == setup.ctx.space
    with pytest.raises(ValidationError):
        await ask(space_id=DEFAULT_SPACE_ID)

    await spaces.assign_client(ADMIN, 9, DEFAULT_SPACE_ID)
    with pytest.raises(ValidationError):
        await ask(setup.conv.id)
    assert (await ask()).ctx.space == SpaceRef(DEFAULT_SPACE_ID, "generic")
    await spaces.assign_client(ADMIN, 9, None)
    with pytest.raises(ValidationError):
        await ask(setup.conv.id)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "role,allowed", [(UserRole.ADMIN, True), (UserRole.CURATOR, True), (UserRole.USER, False)]
)
async def test_upload_client_public_checks_capability_before_storage(role: UserRole, allowed: bool) -> None:
    h = harness()

    if allowed:
        dto = await upload(h, visibility=DocumentVisibility.CLIENT_PUBLIC, user_role=role)
        assert dto.visibility == DocumentVisibility.CLIENT_PUBLIC
    else:
        with pytest.raises(BusinessRuleViolationError):
            await upload(h, visibility=DocumentVisibility.CLIENT_PUBLIC, user_role=role)
        h.storage.upload_file.assert_not_awaited()
