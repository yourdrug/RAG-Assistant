"""Exact RAG probes use structural references within retrieved documents."""

from datetime import date
from unittest.mock import AsyncMock

from langchain_core.documents import Document
import pytest

from domain.identity.roles import UserKind, UserRole
from domain.identity.user_context import UserContext
from domain.knowledge.chunk import ChunkRecord
from domain.knowledge.enums import SearchMode
from domain.rag.classification import has_exact_reference
from domain.rag.context import RagContext
from infrastructure.ml.rag.retrieval_pipeline import apply_exact_search
from infrastructure.repositories.chunk.chunk_search import SQLAlchemyChunkSearchRepository


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "question,reference",
    [
        ("Что предусматривает пункт 21.2?", "Пункт 21.2"),
        ("Что предусмотрено п. 21.2?", "Пункт 21.2"),
        ("Требования подпункта 21.2.3?", "Подпункт 21.2.3"),
        ("Что предусматривает пункт 2?", "Пункт 2"),
        ("Что установлено статьей 15?", "Статья 15"),
    ],
)
async def test_exact_probe_extracts_reference_and_preserves_document_acl_and_date(
    question: str, reference: str
) -> None:
    assert has_exact_reference(question)
    user = UserContext(7, UserKind.INTERNAL, UserRole.CURATOR, group_ids=(4,), managed_client_ids=(8,))
    ctx = RagContext(user, as_of_date=date(2024, 5, 1))
    candidates = [Document(page_content="Related provision", metadata={"document_id": 42})]
    match = ChunkRecord(1, 42, "rules.txt", f"{reference}. Представить заявление.", 2)
    repository = AsyncMock()
    repository.search_substring.return_value = [match]

    await apply_exact_search(question, candidates, user, ctx, repository)

    repository.search_substring.assert_awaited_once_with(
        query=reference,
        user=user,
        limit=5,
        mode=SearchMode.PROVISION.value,
        document_ids=[42],
        as_of_date=ctx.as_of_date,
    )
    assert candidates[-1].page_content == match.content
    assert candidates[-1].metadata["document_id"] == 42


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "question,candidates",
    [
        ("Что предусматривает пункт 21.2?", []),
        ("Что предусматривает пункт 21.2?", [Document(page_content="No document identity")]),
        ("Как подать заявление?", [Document(page_content="Related", metadata={"document_id": 42})]),
    ],
)
async def test_exact_probe_requires_a_reference_and_document_scope(
    question: str, candidates: list[Document]
) -> None:
    user = UserContext(7, UserKind.INTERNAL, UserRole.USER)
    repository = AsyncMock()

    await apply_exact_search(question, candidates, user, RagContext(user), repository)

    repository.search_substring.assert_not_awaited()


@pytest.mark.asyncio
async def test_each_reference_is_searched_once_across_retrieved_documents() -> None:
    user = UserContext(7, UserKind.INTERNAL, UserRole.USER)
    candidates = [
        Document(page_content="First", metadata={"document_id": 42}),
        Document(page_content="Second", metadata={"document_id": 43}),
        Document(page_content="Third", metadata={"document_id": 42}),
    ]
    repository = AsyncMock()
    repository.search_substring.return_value = []

    await apply_exact_search(
        "Сравни пункт 21.2 с п. 5 и пункт 21.2", candidates, user, RagContext(user), repository
    )

    assert [
        (call.kwargs["query"], call.kwargs["document_ids"])
        for call in repository.search_substring.await_args_list
    ] == [("Пункт 21.2", [42, 43]), ("Пункт 5", [42, 43])]
    assert len(candidates) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("reference", ["2", "21"])
async def test_short_numeric_reference_is_searchable_only_in_exact_document_scope(reference: str) -> None:
    user = UserContext(7, UserKind.INTERNAL, UserRole.USER)
    session = AsyncMock()
    session.execute.return_value.all = list
    repository = SQLAlchemyChunkSearchRepository(session)

    assert await repository.search_substring(reference, user, document_id=42) == []
    session.execute.assert_not_awaited()
    session.execute.reset_mock()

    assert await repository.search_substring(reference, user) == []
    assert await repository.search_substring(reference, user, document_id=42, mode="fuzzy") == []
    assert await repository.search_substring_page(reference, user, document_id=42) == ([], 0)
    session.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_probe_keeps_results_of_other_probes() -> None:
    user = UserContext(7, UserKind.INTERNAL, UserRole.USER)
    candidates = [
        Document(page_content="First", metadata={"document_id": 42}),
        Document(page_content="Second", metadata={"document_id": 43}),
    ]
    match = ChunkRecord(1, 43, "rules.txt", "Пункт 2. Текст.", 2)
    repository = AsyncMock()
    repository.search_substring.side_effect = [RuntimeError("db"), [match]]

    await apply_exact_search("Сравни пункт 1 и пункт 2", candidates, user, RagContext(user), repository)

    assert candidates[-1].page_content == match.content


@pytest.mark.asyncio
async def test_multi_document_probe_is_one_query_with_per_document_acl_and_limit() -> None:
    user = UserContext(7, UserKind.INTERNAL, UserRole.USER)
    session = AsyncMock()
    session.execute.return_value.all = list
    repository = SQLAlchemyChunkSearchRepository(session)

    assert await repository.search_substring("Пункт 21.2", user, limit=5, document_ids=[42, 43, 42]) == []

    session.execute.assert_awaited_once()
    sql = str(session.execute.await_args.args[0].compile())
    assert sql.count("UNION ALL") == 1
    assert sql.count("chunks.document_id = ") == 2
    assert sql.count("LIMIT") == 2
    assert sql.count("chunks.owner_id = ") == 2


@pytest.mark.asyncio
async def test_empty_document_scope_skips_query_and_rejects_mixed_scope() -> None:
    user = UserContext(7, UserKind.INTERNAL, UserRole.USER)
    session = AsyncMock()
    repository = SQLAlchemyChunkSearchRepository(session)

    assert await repository.search_substring("Пункт 21.2", user, document_ids=[]) == []
    session.execute.assert_not_awaited()

    with pytest.raises(ValueError, match="document_id"):
        await repository.search_substring("Пункт 21.2", user, document_id=1, document_ids=[2])
