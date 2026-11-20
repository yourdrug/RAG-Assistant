"""Found evidence reaches the answer: cited points are packed first, both conditional branches are quoted."""

from __future__ import annotations

from langchain_core.documents import Document
import pytest

from domain.rag.act_references import cites_referenced_point
from infrastructure.domain_packs.belarus_law.evidence.focus import conditional_branch_anchors
from infrastructure.domain_packs.belarus_law.evidence.profile import BelarusEvidenceProfile
from infrastructure.domain_packs.belarus_law.evidence.reading import has_reading_rule
from infrastructure.ml.evidence_profiles.profiles import evidence_profile_scope
from infrastructure.ml.rag.prompt_budget import estimate_text_tokens
from infrastructure.ml.rag.rag_formatting import format_docs_with_selection
from infrastructure.ml.rag.rag_steps import cited_point_preference


pytestmark = pytest.mark.belarus

POINT_QUESTION = (
    "Какие сведения передаются при автоматической отмене согласно приложению к постановлению МНС N 11 (п. 5)?"
)
WORDING_QUESTION = "Какая новая формулировка «Пояснения N 1» установлена постановлением N 8/91/45/27?"
PRESENCE = (
    "3) в отношении пищевых продуктов при наличии у субъектов необходимых программных и технических средств."
)
ABSENCE = (
    "При отсутствии у субъектов необходимых программных и технических средств в реквизите "
    '"Примечание" (номер поля "28") указывается информация "Товар маркирован средствами идентификации".'
)


def chunk(index: int, text: str, points: list[str], title: str = "Постановление № 11") -> Document:
    return Document(
        page_content=text,
        metadata={
            "document_id": 25 if "11" in title else 24,
            "chunk_index": index,
            "doc_title": title,
            "section": f"п. {points[0]}",
            "point_numbers": points,
            "citation_id": index,
        },
    )


@pytest.mark.parametrize(
    "points, title, cited",
    [
        (["4", "5"], "Постановление № 11", True),
        (["4"], "Постановление № 11", False),
        (["5"], "Постановление № 13", False),
    ],
)
def test_cited_point_is_matched_by_packed_points_and_named_act(
    points: list[str], title: str, *, cited: bool
) -> None:
    assert cites_referenced_point(chunk(9, "текст", points, title).metadata, {"5"}, ("11",)) == cited


def test_chunk_with_cited_point_is_packed_before_higher_scored_evidence() -> None:
    cited = (
        chunk(9, "4. При передаче сведений о передаче.\n5. При передаче сведений об отмене:", ["4", "5"]),
        0.90,
    )
    others = [
        (chunk(i, f"{i}. Моментом создания накладной считается дата " * 20, [str(i)], "№ 9/75"), 0.99)
        for i in range(30, 33)
    ]
    docs = [*others, cited]
    budget = estimate_text_tokens("\n\n".join(doc.page_content for doc, _ in others[:2])) + 60
    profile = BelarusEvidenceProfile()

    with evidence_profile_scope(profile):
        # Exempt every document from the share limit: score order alone must cut the point.
        plain, _ = format_docs_with_selection(docs, budget, named=lambda _item: True)
        preferred, _ = format_docs_with_selection(
            docs, budget, preferred=cited_point_preference(POINT_QUESTION, profile)
        )

    assert "5. При передаче" not in plain
    assert "5. При передаче" in preferred


def test_question_without_point_reference_keeps_score_order() -> None:
    assert cited_point_preference("Какие сведения передаются в ПК СПТ?", BelarusEvidenceProfile()) is None


def test_wording_question_pairs_presence_with_the_longest_absence_branch() -> None:
    truncated = ABSENCE.split(" указывается", 1)[0]
    context = f"[1] Акт\n{PRESENCE}\n{truncated}\n\n---\n\n[2] Акт\n{ABSENCE}"

    assert conditional_branch_anchors(context, WORDING_QUESTION) == [PRESENCE + "\n" + ABSENCE]
    assert has_reading_rule(WORDING_QUESTION, "conditional_wording")


def test_branch_of_another_condition_or_question_is_not_paired() -> None:
    other = "При отсутствии договора поставки в реквизите указывается номер заказа."

    assert conditional_branch_anchors(f"{PRESENCE}\n{other}", WORDING_QUESTION) == []
    assert conditional_branch_anchors(f"{PRESENCE}\n{ABSENCE}", "Какие сведения передаются?") == []
    assert not has_reading_rule("Какие сведения передаются?", "conditional_wording")
