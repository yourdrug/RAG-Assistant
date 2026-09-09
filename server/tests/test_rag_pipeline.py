"""Tests for the RAG pipeline state and extracted helpers from rag_service.py.

Covers:
- RagPipelineState dataclass
- _is_not_found_answer helper
- _build_rag_settings helper
- _prepare_history_dicts
- _resolve_breadth
- _compute_effective_weights
- _resolve_fetch_top_k
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from domain.value_objects.chat_context import ChatContext  # noqa: E402
from domain.value_objects.llm_provider import Breadth  # noqa: E402
from domain.value_objects.rag_settings import (  # noqa: E402
    FeatureToggles,
    HybridSearchConfig,
    RagSettings,
    RerankConfig,
    RetrieverConfig,
)
from domain.value_objects.not_found_patterns import NOT_FOUND_PATTERNS  # noqa: E402
from infrastructure.ml.rag_pipeline import RagPipelineState  # noqa: E402
from infrastructure.ml.rag_service import (  # noqa: E402
    RagService,
    _is_not_found_answer,
    _build_rag_settings,
)


def _make_rag(**overrides):
    defaults = {
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


def _make_service():
    mock_ml = MagicMock()
    return RagService(ml_clients=mock_ml)


# ---------------------------------------------------------------------------
# RagPipelineState
# ---------------------------------------------------------------------------


class TestRagPipelineState:
    def _make_state(self, **overrides):
        defaults = {
            "rag": _make_rag(),
            "t_pipeline_start": 0.0,
            "query_for_search": "test query",
            "question": "test question",
            "history_messages": [],
            "ctx": ChatContext(user_id=1, user_kind="individual"),
            "user": {"id": 1, "kind": "individual"},
            "access_filter": None,
            "retrieval_filter": None,
        }
        defaults.update(overrides)
        return RagPipelineState(**defaults)

    def test_creation_with_required_fields(self):
        state = self._make_state()
        assert state.query_for_search == "test query"
        assert state.breadth is None
        assert state.fetch_k == 0
        assert state.top_k == 0
        assert state.docs == []
        assert state.sources == []
        assert state.terminal is False

    def test_defaults_for_optional_fields(self):
        state = self._make_state()
        assert state.effective_dense_weight == 0.0
        assert state.effective_sparse_weight == 0.0
        assert state.use_exact_ref_boost is False
        assert state.query_domain == ""
        assert state.avg_sim == 0.0
        assert state.full_answer == ""
        assert state.confidence == 0.0
        assert state.usage_report is None
        assert state.q_hash == ""
        assert state.vis_hash == ""

    def test_mutable_fields_can_be_updated(self):
        state = self._make_state()
        state.breadth = Breadth.BROAD
        state.fetch_k = 40
        state.docs = [("doc1", 0.9)]
        state.terminal = True
        assert state.breadth == Breadth.BROAD
        assert state.fetch_k == 40
        assert len(state.docs) == 1
        assert state.terminal is True

    def test_independent_lists(self):
        state1 = self._make_state()
        state2 = self._make_state()
        state1.docs.append(("doc", 0.5))
        assert state2.docs == []


# ---------------------------------------------------------------------------
# _is_not_found_answer
# ---------------------------------------------------------------------------


class TestIsNotFoundAnswer:
    def test_canonical_not_found(self):
        assert _is_not_found_answer("Информация не найдена в документах.") is True

    def test_partial_match(self):
        assert _is_not_found_answer("К сожалению, информация не найдена.") is True

    def test_not_found_in_answer(self):
        assert _is_not_found_answer("Статья 14 регулирует порядок.") is False

    def test_case_insensitive(self):
        assert _is_not_found_answer("ИНФОРМАЦИЯ НЕ НАЙДЕНА") is True

    def test_all_patterns_covered(self):
        for pattern in NOT_FOUND_PATTERNS:
            answer = f"Ответ: {pattern} в данном контексте."
            assert _is_not_found_answer(answer) is True, f"Pattern '{pattern}' not detected"

    def test_empty_string(self):
        assert _is_not_found_answer("") is False

    def test_whitespace_only(self):
        assert _is_not_found_answer("   ") is False


# ---------------------------------------------------------------------------
# _build_rag_settings
# ---------------------------------------------------------------------------


class TestBuildRagSettings:
    def test_returns_rag_settings(self):
        result = _build_rag_settings()
        assert isinstance(result, RagSettings)

    def test_retriever_config(self):
        result = _build_rag_settings()
        assert result.retriever.fetch_k > 0
        assert result.retriever.top_k > 0
        assert result.retriever.fetch_k_broad > 0
        assert result.retriever.top_k_broad > 0

    def test_hybrid_config(self):
        result = _build_rag_settings()
        assert isinstance(result.hybrid_search, HybridSearchConfig)
        assert result.hybrid_search.dense_weight > 0

    def test_rerank_config(self):
        result = _build_rag_settings()
        assert isinstance(result.rerank, RerankConfig)

    def test_feature_toggles(self):
        result = _build_rag_settings()
        assert isinstance(result.features, FeatureToggles)
        assert isinstance(result.features.condense_enabled, bool)
        assert isinstance(result.features.cache_enabled, bool)


# ---------------------------------------------------------------------------
# _prepare_history_dicts
# ---------------------------------------------------------------------------


class TestPrepareHistoryDicts:
    def test_empty_history(self):
        result = RagService._prepare_history_dicts([])
        assert result == []

    def test_dict_history_passthrough(self):
        history = [{"role": "user", "content": "hello"}]
        result = RagService._prepare_history_dicts(history)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_filters_short_content(self):
        history = [
            {"role": "user", "content": "hi"},  # < 3 chars after strip
            {"role": "user", "content": "hello world"},
        ]
        result = RagService._prepare_history_dicts(history)
        # Short messages may be filtered
        assert all(len(m.get("content", "")) >= 3 for m in result)


# ---------------------------------------------------------------------------
# _resolve_breadth
# ---------------------------------------------------------------------------


class TestResolveBreadth:
    def test_alias_short_is_narrow(self):
        svc = _make_service()
        ctx = ChatContext(user_id=1, user_kind="individual", depth="short")
        result = svc._resolve_breadth(ctx, "обычный вопрос")
        assert result == Breadth.NARROW

    def test_alias_detailed_is_broad(self):
        svc = _make_service()
        ctx = ChatContext(user_id=1, user_kind="individual", depth="detailed")
        result = svc._resolve_breadth(ctx, "широкий вопрос")
        assert result == Breadth.BROAD

    def test_auto_classification(self):
        svc = _make_service()
        ctx = ChatContext(user_id=1, user_kind="individual")
        result = svc._resolve_breadth(ctx, "расскажи подробно обо всех аспектах")
        assert isinstance(result, Breadth)


# ---------------------------------------------------------------------------
# _compute_effective_weights
# ---------------------------------------------------------------------------


class TestComputeEffectiveWeights:
    def test_default_weights(self):
        svc = _make_service()
        rag = _make_rag()
        dense, sparse, boost = svc._compute_effective_weights(rag, "обычный вопрос")
        assert dense == 1.5
        assert sparse == 0.5
        assert boost is False

    def test_exact_ref_boost(self):
        svc = _make_service()
        rag = _make_rag()
        dense, sparse, boost = svc._compute_effective_weights(rag, "статья 14 пункт 3")
        assert boost is True
        assert sparse > 0.5


# ---------------------------------------------------------------------------
# _resolve_fetch_top_k
# ---------------------------------------------------------------------------


class TestResolveFetchTopK:
    def test_narrow_params(self):
        svc = _make_service()
        rag = _make_rag()
        fetch_k, top_k = svc._resolve_fetch_top_k(rag, Breadth.NARROW)
        assert fetch_k == 25
        assert top_k == 4

    def test_broad_params(self):
        svc = _make_service()
        rag = _make_rag()
        fetch_k, top_k = svc._resolve_fetch_top_k(rag, Breadth.BROAD)
        assert fetch_k == 40
        assert top_k == 10
