"""Stale cache/index data must not bypass the primary database ACL."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document as Chunk

from application.services.document_access_service import DocumentAccessService
from domain.entities.document import Document
from domain.value_objects.chat_context import ChatContext
from domain.value_objects.llm_provider import Breadth
from domain.value_objects.roles import UserKind, UserRole
from domain.value_objects.user_context import UserContext
from domain.value_objects.visibility import DocumentVisibility
from infrastructure.ml.rag.document_access import cache_is_accessible, filter_scored_documents
from infrastructure.ml.rag.rag_steps import step_check_cache
from fakes import FakeUnitOfWorkFactory
from test_rag_improvements import state_for
from test_rag_pipeline import _make_rag


@pytest.fixture
def database_access():
    factory = FakeUnitOfWorkFactory()
    factory._uow.documents._documents = {
        1: Document(id=1, visibility=DocumentVisibility.INTERNAL_PUBLIC),
        2: Document(id=2, visibility=DocumentVisibility.INTERNAL_PRIVATE, owner_id=99),
        3: Document(id=3, visibility=DocumentVisibility.INTERNAL_GROUP, group_id=10),
        4: Document(id=4, visibility=DocumentVisibility.CLIENT_PRIVATE, owner_id=99),
    }
    factory.create = MagicMock(wraps=factory.create)
    return DocumentAccessService(factory), factory


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'user,expected',
    [
        (UserContext(1, UserKind.INTERNAL, UserRole.USER), {1}),
        (UserContext(1, UserKind.INTERNAL, UserRole.USER, group_ids=(10,)), {1, 3}),
        (UserContext(99, UserKind.INTERNAL, UserRole.USER), {1, 2}),
        (UserContext(99, UserKind.CLIENT, UserRole.USER), {4}),
        (UserContext(1, UserKind.INTERNAL, UserRole.ADMIN), {1, 2, 3}),
        (
            UserContext(
                1,
                UserKind.INTERNAL,
                UserRole.CURATOR,
                managed_internal_ids=(99,),
                managed_client_ids=(99,),
                managed_group_ids=(10,),
            ),
            {1, 2, 3, 4},
        ),
    ],
)
async def test_primary_database_search_acl(database_access, user, expected):
    access, factory = database_access
    assert await access.allowed_document_ids([1, 2, 3, 4, 999], user) == expected
    factory.create.assert_called_once_with(master=True)


@pytest.mark.asyncio
async def test_stale_index_ignores_deleted_and_inaccessible_documents(database_access):
    access, _ = database_access
    user = UserContext(1, UserKind.INTERNAL, UserRole.USER)
    # Index claims all documents are public; DB ACL is authoritative.
    docs = [
        (
            Chunk(
                page_content=str(did),
                metadata={'document_id': did, 'doc_visibility': DocumentVisibility.INTERNAL_PUBLIC.value},
            ),
            0.9,
        )
        for did in [1, 2, 3, 4, 999]
    ]
    docs += [(Chunk(page_content='legacy without ID'), 0.8)]
    permitted = await filter_scored_documents(docs, access, user)
    assert [doc.metadata['document_id'] for doc, _ in permitted] == [1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'dependencies,source_ids,expected',
    [
        ([1], [1], True),
        ([1, 2], [1], False),
        ([1, 999], [1], False),
        ([1], [2], False),
        ([], [], False),
        (None, [1], False),
        ([True], [], False),
    ],
)
async def test_cache_checks_all_dependencies_not_only_citations(
    database_access, dependencies, source_ids, expected
):
    access, _ = database_access
    cached = {
        'answer': 'sensitive',
        'document_ids': dependencies,
        'sources': [{'document_id': did} for did in source_ids],
    }
    user = UserContext(1, UserKind.INTERNAL, UserRole.USER)
    assert await cache_is_accessible(cached, access, user) is expected


@pytest.mark.asyncio
@pytest.mark.parametrize('deleted', [False, True])
async def test_cache_hit_becomes_miss_without_redis_invalidation(database_access, monkeypatch, deleted):
    from infrastructure.ml.rag import rag_steps

    access, factory = database_access
    if deleted:
        del factory._uow.documents._documents[1]
    else:
        factory._uow.documents._documents[1].visibility = DocumentVisibility.INTERNAL_PRIVATE
        factory._uow.documents._documents[1].owner_id = 99
    cached = {'answer': 'must not escape', 'document_ids': [1], 'sources': [{'document_id': 1}]}
    monkeypatch.setattr(rag_steps, 'get_corpus_revision', AsyncMock(return_value='unchanged'))
    monkeypatch.setattr(rag_steps, 'check_cache', AsyncMock(return_value=cached))
    rag = _make_rag()
    rag = replace(rag, features=replace(rag.features, cache_enabled=True))
    state = state_for([], rag=rag)
    _, events = await step_check_cache(state, document_access=access)
    assert events is None and not state.terminal


@pytest.mark.asyncio
async def test_database_failure_never_emits_cached_answer(monkeypatch):
    from infrastructure.ml.rag import rag_steps

    monkeypatch.setattr(rag_steps, 'get_corpus_revision', AsyncMock(return_value='0'))
    monkeypatch.setattr(
        rag_steps,
        'check_cache',
        AsyncMock(return_value={'answer': 'secret', 'document_ids': [1], 'sources': [{'document_id': 1}]}),
    )
    access = SimpleNamespace(allowed_document_ids=AsyncMock(side_effect=RuntimeError('DB unavailable')))
    rag = _make_rag()
    state = state_for([], rag=replace(rag, features=replace(rag.features, cache_enabled=True)))
    with pytest.raises(RuntimeError, match='DB unavailable'):
        await step_check_cache(state, document_access=access)
    assert not state.terminal


@pytest.mark.asyncio
async def test_neighbor_expansion_cannot_reintroduce_forbidden_evidence(database_access, monkeypatch):
    from infrastructure.ml.rag import helpers

    access, _ = database_access
    visible = Chunk(page_content='allowed', metadata={'document_id': 1})
    hidden = Chunk(page_content='secret', metadata={'document_id': 2})
    reranker = AsyncMock(return_value=[(visible, 0.9)])
    monkeypatch.setattr(helpers, 'rerank_documents', reranker)
    monkeypatch.setattr(
        helpers, 'enrich_with_neighbors', AsyncMock(return_value=[(visible, 0.9), (hidden, 0.9)])
    )
    ml = SimpleNamespace(reranker_semaphore=__import__('asyncio').Semaphore(1), reranker=lambda: None)
    ctx = ChatContext(1, UserKind.INTERNAL, UserRole.USER)
    docs, _, _ = await helpers.rerank_and_enrich(
        'question',
        [visible, hidden],
        _make_rag(),
        ml,
        Breadth.NARROW,
        'general',
        4,
        None,
        ctx,
        [],
        8,
        None,
        False,
        visibility_conditions=[],
        user_id=1,
        user_group_ids=[],
        document_access=access,
    )
    assert reranker.call_args.args[1] == [visible]
    assert docs == [(visible, 0.9)]
