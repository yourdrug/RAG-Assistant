"""Retrieval keeps every compared norm: named act kinds, timing wording and reranked timing clauses."""

from __future__ import annotations

from langchain_core.documents import Document
import pytest
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from domain.rag.ranking import MAX_RETAINED_REQUESTED_PROVISIONS, retain_requested_provisions
from infrastructure.domain_packs.belarus_law.evidence.profile import BelarusEvidenceProfile
from infrastructure.domain_packs.belarus_law.evidence.queries import (
    TIMING_WORDING_PROBE,
    evidence_search_queries,
)
from infrastructure.domain_profile.answer_policy import classify_query_domain, search_domains
from infrastructure.repositories.vector.acl import with_domain_filter
from tests.law_pack import law_registry


pytestmark = pytest.mark.belarus

MIXED = "Как соотносятся требования к ЭЦП в Законе N 113-З и в постановлении МНС N 11?"
FREQUENCY = (
    "Чем отличается частота отправки данных о прослеживаемых товарах от частоты отправки данных о прочих?"
)


def test_question_naming_a_law_and_a_decree_searches_both_domains() -> None:
    registry = law_registry()
    domain = classify_query_domain(MIXED, registry)

    assert domain == "decree"
    assert search_domains(MIXED, domain, registry) == ["decree", "legal"]


@pytest.mark.parametrize(
    "question",
    [
        "Изменилась ли периодичность передачи сведений постановлением N 11/49/30/26?",
        "Что требует законодательство о постановлении N 11?",
        "Какие условия договора поставки установлены постановлением N 11?",
    ],
)
def test_domain_filter_is_not_widened_without_another_named_act_kind(question: str) -> None:
    registry = law_registry()
    domain = classify_query_domain(question, registry)

    assert search_domains(question, domain, registry) == [domain]


def test_domain_filter_keeps_acl_and_matches_any_requested_domain() -> None:
    acl = Filter(must=[FieldCondition(key="metadata.owner_id", match=MatchValue(value=1))])

    single = with_domain_filter(acl, "decree")
    several = with_domain_filter(acl, ["decree", "legal"])

    assert isinstance(single.must, list)
    assert isinstance(several.must, list)
    assert single.must == [
        acl,
        FieldCondition(key="metadata.doc_domain", match=MatchValue(value="decree")),
    ]
    assert several.must == [
        acl,
        FieldCondition(key="metadata.doc_domain", match=MatchAny(any=["decree", "legal"])),
    ]


def test_frequency_question_searches_act_timing_wording() -> None:
    assert TIMING_WORDING_PROBE in evidence_search_queries(FREQUENCY)
    assert TIMING_WORDING_PROBE not in evidence_search_queries("Какие сведения передаются в ПК СПТ?")


def test_requested_timing_clauses_follow_the_window_without_displacing_it() -> None:
    items = ["a", "b", "timing-1", "c", "timing-2", "timing-3"]

    retained, excluded = retain_requested_provisions(items, 2, lambda item: item.startswith("timing"))

    assert retained == ["a", "b", "timing-1", "timing-2"]
    assert excluded == ["c", "timing-3"]
    assert len(retained) - 2 == MAX_RETAINED_REQUESTED_PROVISIONS


def test_nothing_is_retained_past_the_window_for_other_questions() -> None:
    profile = BelarusEvidenceProfile()
    timing = Document(
        page_content="передаются EDI-провайдером не реже одного раза в сутки путем взаимодействия",
        metadata={},
    )
    question = "Изменилась ли периодичность передачи сведений EDI-провайдером?"

    assert profile.requested_provision(question, timing.page_content)
    assert not profile.requested_provision("Какие сведения передаются в ПК СПТ?", timing.page_content)
    assert not profile.requested_provision(question, "При передаче в ПК СПТ сведений об отмене:")
