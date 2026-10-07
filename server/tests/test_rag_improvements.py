"""Document citations and regressions from the RAG pipeline review."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import fakeredis.aioredis
import httpx
import pytest
import pytest_asyncio
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from application.services.conversation_service import ConversationService
from domain.exceptions import ContextBudgetExceededError
from domain.value_objects.chat_context import ChatContext
from domain.value_objects.curator_scope import CuratorScope
from domain.value_objects.llm_provider import Breadth
from domain.value_objects.message_role import MessageRole
from domain.value_objects.roles import UserKind, UserRole
from infrastructure.ml import answer_cache
from infrastructure.ml.clients.tei_clients import TEIRerankerClient
from infrastructure.ml.rag.prompt_budget import estimate_message_tokens
from infrastructure.ml.rag.rag_formatting import format_docs_with_selection
from infrastructure.ml.rag.rag_relevance import filter_cited_documents
from infrastructure.ml.rag.rag_reranking import rerank_documents
from infrastructure.ml.rag.rag_sources import extract_sources
from infrastructure.ml.rag.rag_steps import step_build_context, step_check_cache, step_reject_ood
from infrastructure.ml.rag_pipeline import RagPipelineState
from infrastructure.ml.rag_service import RagService
from fakes import FakeUnitOfWorkFactory
from test_rag_pipeline import _make_rag


def doc(text, document_id, source="same.pdf"):
    return Document(page_content=text, metadata={"document_id": document_id, "source": source})


def state_for(docs, **kwargs):
    ctx = kwargs.pop("ctx", ChatContext(1, UserKind.INTERNAL, UserRole.USER))
    return RagPipelineState(
        rag=kwargs.pop("rag", _make_rag()),
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


def test_prompt_groups_fragments_by_document_and_sources_keep_the_same_numbers():
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


def test_source_identity_does_not_merge_same_basename_without_document_ids():
    docs = [
        Document(page_content="A", metadata={"source": "a/rules.pdf"}),
        Document(page_content="B", metadata={"source": "b/rules.pdf"}),
    ]
    _, selected = format_docs_with_selection(docs)
    sources = extract_sources(selected)
    assert len(sources) == 2
    assert [s["citation_id"] for s in sources] == [1, 2]


@pytest.mark.asyncio
async def test_tei_sorted_response_selects_the_actually_relevant_document():
    reranker = TEIRerankerClient("http://tei")
    await reranker.close()
    reranker._client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, json=[{"index": 1, "score": 0.99}, {"index": 0, "score": 0.01}]
            )
        )
    )
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
def test_tei_malformed_response_fails_instead_of_misranking(response):
    with pytest.raises(ValueError):
        TEIRerankerClient._ordered_scores(response, 2)


def test_complete_prompt_budget_trims_old_history_and_keeps_summary():
    ctx = ChatContext(1, UserKind.INTERNAL, UserRole.USER, summary="Заказчик — организация.")
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


def test_prompt_does_not_force_context_into_an_exhausted_window():
    state = state_for([(doc("Evidence", 1), 0.9)], rag=_make_rag(llm_num_ctx_narrow=512))
    with pytest.raises(ContextBudgetExceededError):
        step_build_context(state, None)


def test_exact_token_counter_is_applied_to_the_complete_formatted_prompt():
    state = state_for([(doc("Evidence", 1), 0.9)])
    counter = MagicMock(return_value=state.rag.llm_num_ctx_narrow)
    with pytest.raises(ContextBudgetExceededError):
        step_build_context(state, None, token_counter=counter)
    messages = counter.call_args.args[0]
    assert any("Как оформить договор?" in message.content for message in messages)
    assert len(messages[0].content) > 1000


def test_all_search_paths_can_receive_the_full_curator_snapshot():
    scope = CuratorScope(managed_internal_ids=(10,), managed_group_ids=(20,), managed_client_ids=(30,))
    ctx = ChatContext(1, UserKind.INTERNAL, UserRole.CURATOR, [2], scope)
    service = RagService(MagicMock(), document_access=MagicMock())
    state = service._init_state("Q", [], ctx)
    assert state.user.group_ids == (2,)
    assert state.user.managed_internal_ids == (10,)
    assert state.user.managed_group_ids == (20,)
    assert state.user.managed_client_ids == (30,)


@pytest.mark.asyncio
async def test_topic_hint_does_not_override_document_evidence():
    state = state_for(
        [(doc("Алгоритм согласования договора", 1), 0.9)], query_for_search="Algorithm for LeetCode problem"
    )
    state, events = await step_reject_ood(state)
    assert not state.terminal and events is None


@pytest.mark.asyncio
async def test_summary_bootstrap_includes_the_history_before_it_leaves_the_window():
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
    await service.update_rolling_summary(conv.id, "Q", "A", history)
    await asyncio.gather(*service._background_tasks)
    turns = updater.update.await_args.args[1]
    assert [turn["content"] for turn in turns] == ["Initial constraints", "Initial answer", "Q", "A"]
    await service.update_rolling_summary(conv.id, "Next Q", "Next A", history)
    await asyncio.gather(*service._background_tasks)
    assert [turn["content"] for turn in updater.update.await_args.args[1]] == ["Next Q", "Next A"]


@pytest.mark.asyncio
async def test_long_summary_input_preserves_tails_and_batches_within_the_window(monkeypatch):
    from langchain_core.runnables import RunnableLambda
    from infrastructure.ml.rag.rag_prompts import settings, update_rolling_summary

    monkeypatch.setattr(settings, "llm_num_ctx_narrow", 1024)
    monkeypatch.setattr(settings, "llm_num_predict_narrow", 400)
    calls = []

    async def summarize(prompt):
        messages = prompt.to_messages()
        assert estimate_message_tokens(messages) <= 624
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
async def cache_redis(monkeypatch):
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(answer_cache.redis_client, "_redis", redis)
    try:
        yield redis
    finally:
        await redis.aclose()


@pytest.mark.asyncio
async def test_late_generation_cannot_repopulate_cache_after_document_mutation(cache_redis):
    revision = await answer_cache.get_corpus_revision()
    await answer_cache.store_cached_answer(
        "Q", "old-key", "old answer", [], "scope", [1], expected_revision=revision
    )
    await answer_cache.invalidate_by_document_ids([1])
    await answer_cache.store_cached_answer(
        "Q", "old-key", "late old answer", [], "scope", [1], expected_revision=revision
    )
    assert await answer_cache.find_cached_answer("old-key", "scope") is None
    current = await answer_cache.get_corpus_revision()
    await answer_cache.store_cached_answer(
        "Q", "new-key", "new answer", [], "scope", [1], expected_revision=current
    )
    assert (await answer_cache.find_cached_answer("new-key", "scope", expected_revision=current))[
        "answer"
    ] == "new answer"


@pytest.mark.asyncio
async def test_new_documents_and_config_changes_make_old_answers_miss(cache_redis, monkeypatch):
    from infrastructure.ml.rag.rag_steps import settings

    rag = _make_rag()
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
    assert first.q_hash != second.q_hash and not second.terminal
    monkeypatch.setattr(settings, "llm_model", "changed-model")
    third = state_for([], rag=rag, query_for_search="Q")
    await step_check_cache(third)
    assert second.q_hash != third.q_hash


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["add_chunk", "edit_chunk", "delete_chunk"])
async def test_chunk_mutations_invalidate_answers_after_the_database_transaction(method):
    from test_chunk_service import _make_service, _make_uow

    uow = _make_uow()
    invalidator = SimpleNamespace(invalidate_by_document_ids=AsyncMock())
    service = _make_service(uow, cache_invalidator=invalidator)
    transaction = service._uow_factory.create.return_value

    async def invalidate(document_ids):
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
async def test_cache_invalidation_failure_keeps_outbox_pending_for_retry():
    from domain.entities.vector_outbox_entry import OutboxOperation, OutboxStatus, VectorOutboxEntry
    from infrastructure.repositories.vector.outbox_dispatcher import OutboxDispatcher

    factory = FakeUnitOfWorkFactory()
    async with factory.create(master=True) as uow:
        entry = await uow.vector_outbox.enqueue(
            VectorOutboxEntry(
                operation=OutboxOperation.DELETE_CHUNKS, aggregate_id=42, payload={"chunk_ids": [7]}
            )
        )
    vector_store = AsyncMock()
    invalidator = SimpleNamespace(invalidate_by_document_ids=AsyncMock(side_effect=[OSError("offline"), 1]))
    dispatcher = OutboxDispatcher(factory, vector_store, invalidator)
    assert await dispatcher.run_once() == 1
    async with factory.create() as uow:
        assert uow.vector_outbox._entries[entry.id].status == OutboxStatus.FAILED
    assert await dispatcher.run_once() == 1
    async with factory.create() as uow:
        assert uow.vector_outbox._entries[entry.id].status == OutboxStatus.DONE
    assert vector_store.delete_by_ids.await_count == 2
    assert invalidator.invalidate_by_document_ids.await_count == 2
