"""Document citations and regressions from the RAG pipeline review."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import date
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock

import fakeredis.aioredis
import httpx
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
import pytest
import pytest_asyncio

from application.chat import ConversationService
from application.rag.dto import PreparedQuery
from composition.service_providers import create_rag_service
from config import settings
from domain.chat.message import MessageRole
from domain.identity.roles import UserKind, UserRole
from domain.identity.user_context import CuratorScope, UserContext
from domain.knowledge.index_sync import OutboxOperation, OutboxStatus, VectorOutboxEntry
from domain.rag.breadth import Breadth
from domain.rag.context import RagContext
from domain.rag.exceptions import ContextBudgetExceededError
from infrastructure.domain_profile.answer_policy import policy_for
from infrastructure.ml.cache import answers as answer_cache, keys as cache_keys
from infrastructure.ml.cache.answers import compute_question_hash
from infrastructure.ml.clients.tei_clients import TEIRerankerClient
from infrastructure.ml.rag.prompt_budget import estimate_message_tokens
from infrastructure.ml.rag.rag_formatting import format_docs_with_selection
from infrastructure.ml.rag.rag_prompts import update_rolling_summary
from infrastructure.ml.rag.rag_relevance import filter_cited_documents
from infrastructure.ml.rag.rag_reranking import rerank_documents
from infrastructure.ml.rag.rag_sources import extract_sources
from infrastructure.redis.redis_client import redis_client
from infrastructure.repositories.vector.outbox_dispatcher import OutboxDispatcher
from tests.fakes import FakeUnitOfWorkFactory
from tests.rag_stage_fixtures import (
    build_context as step_build_context,
    check_cache as step_check_cache,
    reject_ood as step_reject_ood,
    stage_fixture,
)
from tests.test_chunk_service import _make_service, _make_uow
from tests.test_rag_pipeline import make_rag


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from langchain_core.prompt_values import PromptValue

    from infrastructure.ml.rag.rag_steps import RagOperations


def doc(text: str, document_id: int, source: str = "same.pdf") -> Document:
    return Document(page_content=text, metadata={"document_id": document_id, "source": source})


def state_for(docs: list[tuple[Document, float]], **kwargs: Any) -> SimpleNamespace:
    ctx = kwargs.pop(
        "ctx", RagContext(user=UserContext(user_id=1, user_kind=UserKind.INTERNAL, user_role=UserRole.USER))
    )
    return stage_fixture(
        rag=kwargs.pop("rag", make_rag()),
        t_pipeline_start=0,
        question=kwargs.pop("question", "Как оформить договор?"),
        ctx=ctx,
        user=ctx.to_user_context(),
        access_filter=None,
        retrieval_filter=None,
        breadth=Breadth.NARROW,
        docs=docs,
        **kwargs,
    )


def test_prompt_groups_fragments_by_document_and_sources_keep_the_same_numbers() -> None:
    docs = [
        (doc("First condition", 10), 0.9),
        (doc("Other document", 20), 0.95),
        (doc("Second condition", 10), 0.8),
    ]
    context, selected = format_docs_with_selection(docs)
    assert context.count("[1]") == context.count("[2]") == 1
    assert (
        context.index("Other document") < context.index("First condition") < context.index("Second condition")
    )
    assert [item[0].metadata["citation_id"] for item in selected] == [1, 2, 2]
    # Selection metadata does not mutate the retrieved evidence.
    assert all("citation_id" not in item[0].metadata for item in docs)
    cited = filter_cited_documents("Условие согласно [2].", selected)
    sources = extract_sources(cited)
    assert [(s["citation_id"], s["document_id"]) for s in sources] == [(2, 10)]
    assert filter_cited_documents("Условие согласно [999].", selected) == []


def test_source_identity_does_not_merge_same_basename_without_document_ids() -> None:
    docs = [
        Document(page_content="A", metadata={"source": "a/rules.pdf"}),
        Document(page_content="B", metadata={"source": "b/rules.pdf"}),
    ]
    _, selected = format_docs_with_selection(docs)
    sources = extract_sources(selected)
    assert len(sources) == 2
    assert [s["citation_id"] for s in sources] == [1, 2]


@pytest.mark.asyncio
async def test_tei_sorted_response_selects_the_actually_relevant_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_client = httpx.AsyncClient
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(200, json=[{"index": 1, "score": 0.99}, {"index": 0, "score": 0.01}])
    )
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: real_client(**{**kwargs, "transport": transport})
    )
    reranker = TEIRerankerClient("http://tei")
    try:
        selected = await rerank_documents("Q", [doc("irrelevant", 1), doc("relevant", 2)], 1, reranker)
        assert selected[0][0].metadata["document_id"] == 2
        assert selected[0][1] == 0.99
    finally:
        await reranker.close()


@pytest.mark.parametrize(
    "response",
    [
        [{"index": 0, "score": 0.5}, {"index": 0, "score": 0.9}],
        [{"index": 2, "score": 0.5}],
        [{"index": 0, "score": 0.5}],
        [{"index": 0, "score": float("nan")}, {"index": 1, "score": 0.5}],
    ],
)
def test_tei_malformed_response_fails_instead_of_misranking(
    response: list[dict[str, int | float]] | list[dict[str, int] | dict[str, int | float]],
) -> None:
    with pytest.raises(ValueError, match="TEI rerank"):
        TEIRerankerClient.ordered_scores(response, 2)


def test_complete_prompt_budget_trims_old_history_and_keeps_summary() -> None:
    ctx = RagContext(
        user=UserContext(user_id=1, user_kind=UserKind.INTERNAL, user_role=UserRole.USER),
        summary="Заказчик — организация.",
    )
    state = state_for(
        [(doc("Документ подтверждает условия.", 1), 0.9)],
        ctx=ctx,
        history_messages=[HumanMessage(content="old " * 10000), AIMessage(content="old answer")],
    )
    messages, _ = step_build_context(state, None)
    assert (
        estimate_message_tokens(messages) + state.rag.llm_num_predict_narrow <= state.rag.llm_num_ctx_narrow
    )
    assert state._prompt_docs
    assert "Заказчик — организация." in "\n".join(message.content for message in messages)
    assert not state.history_messages


def test_prompt_does_not_force_context_into_an_exhausted_window() -> None:
    state = state_for([(doc("Evidence", 1), 0.9)], rag=make_rag(llm_num_ctx_narrow=512))
    with pytest.raises(ContextBudgetExceededError):
        step_build_context(state, None)


def test_exact_token_counter_is_applied_to_the_complete_formatted_prompt() -> None:
    state = state_for([(doc("Evidence", 1), 0.9)])
    counter = MagicMock(return_value=state.rag.llm_num_ctx_narrow)
    with pytest.raises(ContextBudgetExceededError):
        step_build_context(state, None, token_counter=counter)
    messages = counter.call_args.args[0]
    assert any("Как оформить договор?" in message.content for message in messages)
    assert len(messages[0].content) > 1000


def test_all_search_paths_can_receive_the_full_curator_snapshot() -> None:
    scope = CuratorScope(managed_internal_ids=(10,), managed_group_ids=(20,), managed_client_ids=(30,))
    ctx = RagContext(
        user=UserContext(
            user_id=1, user_kind=UserKind.INTERNAL, user_role=UserRole.CURATOR, group_ids=(2,)
        ).with_curator_scope(scope)
    )
    service = create_rag_service(MagicMock(), document_access=MagicMock())
    state = cast("RagOperations", service.preparation).retrieval_work(
        PreparedQuery(service.preparation.request("Q", [], ctx), "Q")
    )
    assert state.user.group_ids == (2,)
    assert state.user.managed_internal_ids == (10,)
    assert state.user.managed_group_ids == (20,)
    assert state.user.managed_client_ids == (30,)


@pytest.mark.asyncio
async def test_topic_hint_does_not_override_document_evidence() -> None:
    state = state_for(
        [(doc("Алгоритм согласования договора", 1), 0.9)], query_for_search="Algorithm for LeetCode problem"
    )
    state, events = await step_reject_ood(state)
    assert not state.terminal
    assert events is None


@pytest.mark.asyncio
async def test_summary_bootstrap_includes_the_history_before_it_leaves_the_window() -> None:
    factory = FakeUnitOfWorkFactory()
    async with factory.create(master=True) as uow:
        conv = await uow.conversations.create(1)
    updater = SimpleNamespace(update=AsyncMock(return_value="summary"))
    service = ConversationService(
        factory, updater, SimpleNamespace(rolling_summary_enabled=True, history_window=2)
    )
    history = [
        {"role": MessageRole.USER, "content": "Initial constraints"},
        {"role": MessageRole.ASSISTANT, "content": "Initial answer"},
    ]
    assert conv.id is not None
    await service.update_rolling_summary(conv.id, "Q", "A", history)
    await asyncio.gather(*service._background_tasks)
    turns = updater.update.await_args.args[1]
    assert [turn["content"] for turn in turns] == ["Initial constraints", "Initial answer", "Q", "A"]
    await service.update_rolling_summary(conv.id, "Next Q", "Next A", history)
    await asyncio.gather(*service._background_tasks)
    assert [turn["content"] for turn in updater.update.await_args.args[1]] == ["Next Q", "Next A"]


@pytest.mark.asyncio
async def test_long_summary_input_preserves_tails_and_batches_within_the_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    monkeypatch.setattr(settings, "llm_num_ctx_narrow", 1024)
    monkeypatch.setattr(settings, "llm_num_predict_narrow", 400)
    calls = []

    async def summarize(prompt: PromptValue) -> AIMessage:
        messages = prompt.to_messages()
        assert estimate_message_tokens(messages) <= 624
        assert isinstance(messages[-1].content, str)
        calls.append(messages[-1].content)
        return AIMessage(content="Краткое резюме.")

    tail = "Ключевое условие в конце длинного сообщения."
    result = await update_rolling_summary(
        RunnableLambda(summarize),
        None,
        [{"role": MessageRole.USER, "content": "Начало. " * 1000 + tail}],
    )
    assert result == "Краткое резюме."
    assert len(calls) > 1
    assert tail in "".join(calls)


@pytest_asyncio.fixture
async def cache_redis(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(redis_client, "_redis", redis)
    try:
        yield redis
    finally:
        await redis.aclose()


@pytest.mark.asyncio
async def test_late_generation_cannot_repopulate_cache_after_document_mutation(
    cache_redis: fakeredis.aioredis.FakeRedis,
) -> None:
    revision = await answer_cache.get_corpus_revision()
    await answer_cache.store_cached_answer(
        "Q", "old-key", "old answer", [], "scope", document_ids=[1], expected_revision=revision
    )
    await answer_cache.invalidate_by_document_ids([1])
    await answer_cache.store_cached_answer(
        "Q", "old-key", "late old answer", [], "scope", document_ids=[1], expected_revision=revision
    )
    assert await answer_cache.find_cached_answer("old-key", "scope") is None
    current = await answer_cache.get_corpus_revision()
    await answer_cache.store_cached_answer(
        "Q", "new-key", "new answer", [], "scope", document_ids=[1], expected_revision=current
    )
    cached = await answer_cache.find_cached_answer("new-key", "scope", expected_revision=current)
    assert cached is not None
    assert cached["answer"] == "new answer"


@pytest.mark.asyncio
async def test_new_documents_and_config_changes_make_old_answers_miss(
    cache_redis: fakeredis.aioredis.FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> None:

    rag = make_rag()
    rag = replace(rag, features=replace(rag.features, cache_enabled=True))
    first = state_for([], rag=rag, query_for_search="Q")
    await step_check_cache(first)
    await answer_cache.store_cached_answer(
        "Q", first.q_hash, "answer", [], first.vis_hash, expected_revision=first.cache_revision
    )
    # A new document need not occur in any old reverse index to change answers.
    await answer_cache.invalidate_by_document_ids([999], cache_enabled=False)
    second = state_for([], rag=rag, query_for_search="Q")
    await step_check_cache(second)
    assert first.q_hash != second.q_hash
    assert not second.terminal
    monkeypatch.setattr(settings, "llm_model", "changed-model")
    third = state_for([], rag=rag, query_for_search="Q")
    await step_check_cache(third)
    assert second.q_hash != third.q_hash


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["add_chunk", "edit_chunk", "delete_chunk"])
async def test_chunk_mutations_invalidate_answers_after_the_database_transaction(method: str) -> None:

    uow = _make_uow()
    invalidator = SimpleNamespace(invalidate_by_document_ids=AsyncMock())
    service = _make_service(uow, cache_invalidator=invalidator)
    transaction = cast("MagicMock", service._uow_factory.create).return_value

    async def invalidate(document_ids: list[int]) -> None:
        transaction.__aexit__.assert_awaited_once()
        assert document_ids == [1]

    invalidator.invalidate_by_document_ids.side_effect = invalidate
    params = {"document_id": 1, "user_id": 1, "user_role": UserRole.ADMIN}

    if method != "delete_chunk":
        params["content"] = "Условия обработки документа. " * 10

    if method != "add_chunk":
        params["chunk_id"] = 10

    await getattr(service, method)(**params)
    invalidator.invalidate_by_document_ids.assert_awaited_once_with([1])


@pytest.mark.asyncio
async def test_cache_invalidation_failure_keeps_outbox_pending_for_retry() -> None:

    factory = FakeUnitOfWorkFactory()
    async with factory.create(master=True) as uow:
        entry = await uow.vector_outbox.enqueue(
            VectorOutboxEntry(
                operation=OutboxOperation.DELETE_CHUNKS, aggregate_id=42, payload={"chunk_ids": [7]}
            )
        )
    assert entry.id is not None
    vector_store = AsyncMock()
    invalidator = SimpleNamespace(invalidate_by_document_ids=AsyncMock(side_effect=[OSError("offline"), 1]))
    dispatcher = OutboxDispatcher(factory, vector_store, invalidator)
    assert await dispatcher.run_once() == 1
    async with factory.create() as uow:
        assert uow.fake_vector_outbox._entries[entry.id].status == OutboxStatus.FAILED
    assert await dispatcher.run_once() == 1
    async with factory.create() as uow:
        assert uow.fake_vector_outbox._entries[entry.id].status == OutboxStatus.DONE
    assert vector_store.delete_by_ids.await_count == 2
    assert invalidator.invalidate_by_document_ids.await_count == 2


@pytest.mark.asyncio
async def test_split_answer_policies_preserve_cache_prompt_scope(
    cache_redis: fakeredis.aioredis.FakeRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    rag = make_rag()
    rag = replace(rag, features=replace(rag.features, cache_enabled=True))
    state = state_for([], rag=rag, query_for_search="Q")
    state.ctx = replace(state.ctx, as_of_date=date(2026, 1, 1))
    contexts = []
    original_hash = compute_question_hash

    def capture_hash(question: str, *, context: dict[str, Any]) -> str:
        contexts.append(context)
        return original_hash(question, context=context)

    monkeypatch.setattr(cache_keys, "compute_question_hash", capture_hash)
    keys = ("general", "legal", "decree")
    await step_check_cache(state, keys)
    first_hash = state.q_hash
    assert contexts[-1]["domain_prompts"] == {
        key: [policy_for(key).prompt_addendum(state.ctx.as_of_date) for breadth in ("narrow", "broad")]
        for key in keys
    }
    await step_check_cache(state, tuple(reversed(keys)))
    assert state.q_hash == first_hash
    await step_check_cache(state, ("general", "legal"))
    assert state.q_hash != first_hash
    await step_check_cache(state, None)
    assert contexts[-1]["domain_prompts"] == {}
    await step_check_cache(state, ("unknown",))
    assert contexts[-1]["domain_prompts"] == {"unknown": [None, None]}
