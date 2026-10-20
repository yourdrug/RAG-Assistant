"""Search compatibility, current access context and retrieval edge cases."""

import importlib
from unittest.mock import AsyncMock, Mock

from langchain_core.documents import Document
import pytest

from application.search import HybridRetriever, SearchService
from domain.identity.roles import UserKind, UserRole
from domain.identity.user_context import UserContext
from domain.knowledge.enums import SearchMode


@pytest.mark.parametrize(
    "public,implementation,name",
    [
        ("application.search", "application.search.service", "SearchService"),
        ("application.search", "application.search.hybrid", "HybridRetriever"),
    ],
)
def test_public_exports_share_canonical_objects(public: str, implementation: str, name: str) -> None:
    assert getattr(importlib.import_module(public), name) is getattr(
        importlib.import_module(implementation), name
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("role", [UserRole.USER, UserRole.ADMIN, UserRole.CURATOR])
async def test_search_refreshes_scope_and_preserves_page(role: UserRole) -> None:
    uow = AsyncMock()
    uow.groups.get_user_group_ids.return_value = [2]
    uow.assignments.get_managed_client_ids.return_value = [10]
    uow.assignments.get_managed_internal_ids.return_value = [11]
    uow.assignments.get_managed_group_ids.return_value = [12]
    uow.chunks.search_substring_page.return_value = ([], 7)
    factory = Mock()
    factory.create.return_value.__aenter__ = AsyncMock(return_value=uow)
    factory.create.return_value.__aexit__ = AsyncMock(return_value=False)
    user = UserContext(1, UserKind.INTERNAL, role, group_ids=(99,), managed_client_ids=(99,))

    result = await SearchService(factory).exact_search(
        "needle", user, limit=2, offset=4, document_id=5, mode=SearchMode.EXACT.value
    )

    curator = role == UserRole.CURATOR
    uow.chunks.search_substring_page.assert_awaited_once_with(
        query="needle",
        user=UserContext(
            1,
            UserKind.INTERNAL,
            role,
            group_ids=(2,),
            managed_client_ids=(10,) if curator else (),
            managed_internal_ids=(11,) if curator else (),
            managed_group_ids=(12,) if curator else (),
        ),
        limit=2,
        offset=4,
        document_id=5,
        mode=SearchMode.EXACT.value,
    )
    assert result == ([], 7)
    factory.create.assert_called_once_with()


@pytest.mark.asyncio
async def test_scope_load_failure_prevents_search() -> None:
    uow = AsyncMock()
    factory = Mock()
    factory.create.return_value.__aenter__ = AsyncMock(return_value=uow)
    factory.create.return_value.__aexit__ = AsyncMock(return_value=False)
    contexts = Mock(build=AsyncMock(side_effect=RuntimeError("scope unavailable")))
    with pytest.raises(RuntimeError, match="scope unavailable"):
        await SearchService(factory, contexts).exact_search(
            "needle", UserContext(1, UserKind.INTERNAL, UserRole.USER)
        )
    uow.chunks.search_substring_page.assert_not_awaited()
    assert factory.create.return_value.__aexit__.await_args.args[0] is RuntimeError


def test_dense_only_merge_skips_missing_hashes_and_duplicate_content() -> None:
    first = Document(page_content="same")
    duplicate = Document(page_content="same")
    last = Document(page_content="other")
    result = HybridRetriever().merge_and_dedup(
        [("missing", 1), ("a", 0.9), ("a", 0.8), ("b", 0.7), ("c", 0.6)],
        [],
        {"a": (0.9, first), "b": (0.7, duplicate), "c": (0.6, last)},
        fetch_k=2,
        rrf_k=60,
        dense_weight=1,
        sparse_weight=1,
    )
    assert result == [first, last]


@pytest.mark.parametrize(
    "minimum,gap,expected",
    [(None, None, [1, 0.5, 0]), (0, None, [1, 0.5, 0]), (0.5, 0.5, [1, 0.5]), (2, 0.5, [])],
)
def test_rerank_thresholds_preserve_boundaries(
    minimum: float | None, gap: float | None, expected: list[float]
) -> None:
    ranked = [(Document(page_content=str(score)), score) for score in [1, 0.5, 0]]
    assert [score for _, score in HybridRetriever().apply_rerank_filters(ranked, minimum, gap)] == expected
