"""Tests for frozen RAG stages and the existing domain and adapter helpers.

Covers:
- Frozen request and prepared-query DTOs
- _is_not_found_answer helper
- _build_rag_settings helper
- _prepare_history_dicts
- _resolve_breadth
- _compute_effective_weights
- _resolve_fetch_top_k
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from application.rag.dto import PreparedQuery, RagRequest
from composition.service_providers import create_rag_service
from domain.identity.roles import UserKind
from domain.identity.user_context import UserContext
from domain.rag.breadth import BREADTH_ALIASES, Breadth
from domain.rag.classification import classify_question_breadth
from domain.rag.context import RagContext
from domain.rag.retrieval import compute_retrieval_params
from domain.rag.settings import (
    FeatureToggles,
    HybridSearchConfig,
    RagSettings,
    RerankConfig,
    RetrieverConfig,
)
from infrastructure.ml.rag.prompt_blocks import NOT_FOUND_PATTERNS
from infrastructure.ml.rag.rag_config import build_rag_settings as _build_rag_settings
from infrastructure.ml.rag.rag_postprocess import is_not_found_answer as _is_not_found_answer
from infrastructure.ml.rag.rag_steps import RagOperations


if TYPE_CHECKING:
    from application.rag.service import RagService


def make_rag(**overrides: Any) -> RagSettings:
    defaults: dict[str, Any] = {
        "retriever": RetrieverConfig(
            fetch_k=25,
            top_k=4,
            fetch_k_broad=40,
            top_k_broad=10,
        ),
        "hybrid_search": HybridSearchConfig(
            enabled=True,
            bm25_fetch_k=25,
            rrf_k=30,
            dense_weight=1.5,
            sparse_weight=0.5,
        ),
        "rerank": RerankConfig(min_score=0.15, score_gap_ratio=0.1),
        "features": FeatureToggles(
            citation_filter_enabled=False,
            relevance_gate_enabled=False,
            condense_enabled=True,
            decomposition_enabled=False,
            rolling_summary_enabled=True,
            cache_enabled=False,
        ),
        "source_min_score": 0.3,
    }
    defaults.update(overrides)
    return RagSettings(**defaults)


def _make_service() -> RagService:
    mock_ml = MagicMock()
    return create_rag_service(ml_clients=mock_ml, document_access=MagicMock())


# ---------------------------------------------------------------------------
# Frozen stage values
# ---------------------------------------------------------------------------


def test_request_and_stage_results_are_frozen() -> None:

    request = RagRequest(
        "question",
        (),
        RagContext(user=UserContext(user_id=1, user_kind=UserKind.INTERNAL)),
        make_rag(),
        0,
        "req",
        "model",
    )
    prepared = PreparedQuery(request, "query")
    field_name = "query"

    with pytest.raises(FrozenInstanceError):
        setattr(prepared, field_name, "changed")
    assert not hasattr(request, "docs")
    assert not hasattr(prepared, "terminal")


# ---------------------------------------------------------------------------
# _is_not_found_answer
# ---------------------------------------------------------------------------


class TestIsNotFoundAnswer:
    def test_canonical_not_found(self) -> None:
        assert _is_not_found_answer("Информация не найдена в документах.") is True

    def test_partial_match(self) -> None:
        assert _is_not_found_answer("К сожалению, информация не найдена.") is True

    def test_not_found_in_answer(self) -> None:
        assert _is_not_found_answer("Статья 14 регулирует порядок.") is False

    def test_case_insensitive(self) -> None:
        assert _is_not_found_answer("ИНФОРМАЦИЯ НЕ НАЙДЕНА") is True

    def test_all_patterns_covered(self) -> None:
        for pattern in NOT_FOUND_PATTERNS:
            # Use short answers so the pattern-check branch is triggered (< 50 chars)
            answer = f"Ответ: {pattern}"
            assert _is_not_found_answer(answer) is True, f"Pattern '{pattern}' not detected"

    def test_empty_string(self) -> None:
        assert _is_not_found_answer("") is False

    def test_whitespace_only(self) -> None:
        assert _is_not_found_answer("   ") is False


# ---------------------------------------------------------------------------
# _build_rag_settings
# ---------------------------------------------------------------------------


class TestBuildRagSettings:
    def test_returns_rag_settings(self) -> None:
        result = _build_rag_settings()
        assert isinstance(result, RagSettings)

    def test_retriever_config(self) -> None:
        result = _build_rag_settings()
        assert result.retriever.fetch_k > 0
        assert result.retriever.top_k > 0
        assert result.retriever.fetch_k_broad > 0
        assert result.retriever.top_k_broad > 0

    def test_hybrid_config(self) -> None:
        result = _build_rag_settings()
        assert isinstance(result.hybrid_search, HybridSearchConfig)
        assert result.hybrid_search.dense_weight > 0

    def test_rerank_config(self) -> None:
        result = _build_rag_settings()
        assert isinstance(result.rerank, RerankConfig)

    def test_feature_toggles(self) -> None:
        result = _build_rag_settings()
        assert isinstance(result.features, FeatureToggles)
        assert isinstance(result.features.condense_enabled, bool)
        assert isinstance(result.features.cache_enabled, bool)


# ---------------------------------------------------------------------------
# _prepare_history_dicts
# ---------------------------------------------------------------------------


class TestPrepareHistoryDicts:
    def test_empty_history(self) -> None:
        result = RagOperations.prepare_history_dicts([])
        assert result == []

    def test_dict_history_passthrough(self) -> None:
        history = [{"role": "user", "content": "hello"}]
        result = RagOperations.prepare_history_dicts(history)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_filters_short_content(self) -> None:
        history = [
            {"role": "user", "content": "hi"},  # < 3 chars after strip
            {"role": "user", "content": "hello world"},
        ]
        result = RagOperations.prepare_history_dicts(history)
        # Short messages may be filtered
        assert all(len(m.get("content", "")) >= 3 for m in result)


# ---------------------------------------------------------------------------
# classify_question_breadth
# ---------------------------------------------------------------------------


class TestResolveBreadth:
    def test_alias_short_is_narrow(self) -> None:

        raw = BREADTH_ALIASES.get("short") or classify_question_breadth("обычный вопрос")
        assert raw in ("short", Breadth.NARROW)

    def test_alias_detailed_is_broad(self) -> None:

        raw = BREADTH_ALIASES.get("detailed") or classify_question_breadth("широкий вопрос")
        assert raw in ("detailed", Breadth.BROAD)

    def test_auto_classification(self) -> None:

        result = classify_question_breadth("расскажи подробно обо всех аспектах")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# compute_retrieval_params
# ---------------------------------------------------------------------------


class TestComputeEffectiveWeights:
    def test_default_weights(self) -> None:

        rag = make_rag(exact_ref_sparse_boost=2.0)
        params = compute_retrieval_params(Breadth.NARROW, rag, "обычный вопрос")
        assert params["effective_dense_weight"] == 1.5
        assert params["effective_sparse_weight"] == 0.5
        assert params["use_exact_ref_boost"] is False

    def test_exact_ref_boost(self) -> None:

        rag = make_rag(exact_ref_sparse_boost=2.0)
        params = compute_retrieval_params(Breadth.NARROW, rag, "статья 14 пункт 3")
        assert params["use_exact_ref_boost"] is True
        assert params["effective_sparse_weight"] > 0.5


# ---------------------------------------------------------------------------
# compute_retrieval_params — fetch_k / top_k
# ---------------------------------------------------------------------------


class TestResolveFetchTopK:
    def test_narrow_params(self) -> None:

        rag = make_rag(exact_ref_sparse_boost=2.0)
        params = compute_retrieval_params(Breadth.NARROW, rag, "обычный вопрос")
        assert params["fetch_k"] == 25

    def test_broad_params(self) -> None:

        rag = make_rag(exact_ref_sparse_boost=2.0)
        params = compute_retrieval_params(Breadth.BROAD, rag, "широкий вопрос")
        assert params["fetch_k"] == 40
