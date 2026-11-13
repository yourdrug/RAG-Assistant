"""Token budgets, sentence overlap and fallback behavior through the ingestion entry point."""

from pathlib import Path

from langchain.schema import Document
import pytest

from config import ChunkingSection, process_section
from domain.ingestion.embed_text import MAX_EMBED_TOKENS
from domain.ingestion.profiles.decree import DecreeDomainProfile
from domain.ingestion.profiles.legal import LegalDomainProfile
from domain.ingestion.raw_document import RawDocument
from domain.ingestion.text_budget import overlap_tail, sentence_fragments
from infrastructure.domain_profile.settings_adapter import DomainSettingsAdapter
from infrastructure.ml.ingestion.splitting import TreeChunkingAdapter, split_documents
from infrastructure.ml.ingestion.text_chunks import text_tree_chunks
from infrastructure.ml.langchain_document_parser import LangchainDocumentSplitter
from infrastructure.ml.tokenization import CharEstimateTokenCounter, HfTokenCounter, build_token_counter


TOKENIZER = Path(__file__).resolve().parents[2] / "data/tei_cache/bge-m3/tokenizer.json"


@pytest.fixture(scope="module")
def real_counter() -> HfTokenCounter:
    if not TOKENIZER.is_file():
        pytest.skip("Local BAAI/bge-m3 tokenizer is not installed")

    return HfTokenCounter(TOKENIZER)


@pytest.mark.parametrize("domain, maximum", [("general", 512), ("legal", 800), ("decree", 800)])
@pytest.mark.parametrize("table", [False, True])
def test_every_final_chunk_fits_real_bge_tokens(
    real_counter: HfTokenCounter,
    domain: str,
    maximum: int,
    *,
    table: bool,
) -> None:
    body = "Заявитель обязан представить сведения. " * 150 + "資料申請 " * 100
    text = "1. Условия:\n1.1. " + body + "\n1.2. Заключительное правило."
    metadata = {"owner_id": 42, "visibility": "client_private", "group_id": 7, "doc_title": "Название" * 50}

    if table:
        text = "| Код | Условие |\n|---|---|\n| A42 | " + body + " |\n| B52 | Последняя строка |"
        metadata["content_type"] = "table"

    chunks = LangchainDocumentSplitter(token_counter=real_counter).split(
        [RawDocument(text, metadata)], domain
    )
    assert len(chunks) > 1
    assert all(real_counter.count(chunk.page_content) <= maximum for chunk in chunks)
    assert all(
        real_counter.count(chunk.embed_text or chunk.page_content) <= MAX_EMBED_TOKENS for chunk in chunks
    )
    assert all(chunk.metadata["owner_id"] == 42 and chunk.metadata["group_id"] == 7 for chunk in chunks)
    assert all((chunk.embed_text or chunk.page_content).endswith(chunk.page_content) for chunk in chunks)
    assert any(
        "Заключительное правило" in chunk.page_content or "Последняя строка" in chunk.page_content
        for chunk in chunks
    )


def test_fallback_counter_needs_no_tokenizer_file(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    counter = build_token_counter(str(tmp_path / "missing.json"))
    assert isinstance(counter, CharEstimateTokenCounter)
    assert counter.count("12345") == 2
    assert counter.count("") == 0
    assert "character estimate" in caplog.text
    chunks = LangchainDocumentSplitter(token_counter=counter).split([RawDocument("Длинный текст. " * 1000)])
    assert all(counter.count(chunk.page_content) <= 512 for chunk in chunks)


def test_only_complete_sentences_overlap_and_pages_follow_source() -> None:
    first = "Первое предложение. Второе предложение. "
    second = "Третье предложение. Четвёртое предложение. Пятое предложение."
    text = first + second
    chunks = text_tree_chunks(
        Document(
            page_content=text, metadata={"_page_spans": [(0, len(first), 1), (len(first), len(text), 2)]}
        ),
        55,
        25,
        target=45,
    )
    assert len(chunks) > 1
    assert chunks[0].metadata["pages"] == [1]
    assert chunks[-1].metadata["pages"] == [2]
    starts = {sentence.strip().split()[0] for sentence in sentence_fragments(text)}
    assert all(chunk.page_content.split()[0] in starts for chunk in chunks)
    assert overlap_tail(first, 25).strip() == "Второе предложение."
    assert overlap_tail(first, 5) == ""


def test_legal_abbreviations_survive_overflow() -> None:
    sentence = "В соответствии со ст. 5 и п. 3 т.е. по ст. 12 действует правило. "
    assert sentence_fragments(sentence) == [sentence]
    chunks = text_tree_chunks(Document(page_content=sentence * 10), 140, 50)
    assert all(chunk.page_content.startswith("В соответствии") for chunk in chunks)


@pytest.mark.parametrize("profile_type", [LegalDomainProfile, DecreeDomainProfile])
def test_structured_domains_ignore_retired_character_limits_and_overlap(
    profile_type: type[LegalDomainProfile | DecreeDomainProfile],
    real_counter: HfTokenCounter,
) -> None:
    settings = DomainSettingsAdapter()
    profile = profile_type(settings)
    values = {(item.key, profile.key): item.value for item in profile.config_defaults()}
    values["max_unit_chars", profile.key] = "1"
    values[f"{profile.key}_chunk_overlap", profile.key] = "200"
    settings.replace_snapshot(values)
    text = "1. " + "Содержательное правило. " * 120
    chunks = split_documents(
        [Document(page_content=text)], profile.key, profile, settings, token_counter=real_counter
    )
    bodies = " ".join(chunk.page_content for chunk in chunks)
    assert bodies.count("Содержательное правило.") == 120
    assert TreeChunkingAdapter(profile.key, profile, settings, real_counter).chunk_overlap == 0


def test_invalid_token_budget_is_rejected() -> None:
    section = process_section(ChunkingSection).model_copy(
        update={"chunk_target_tokens": 800, "chunk_max_tokens": 100}
    )
    assert any("TARGET_TOKENS" in error for error in section.check_chunking())


def test_inserted_article_number_survives_final_chunk_metadata(real_counter: HfTokenCounter) -> None:
    text = "Статья 26. Сертификат\nСтатья 26-1. Атрибутный сертификат\nСтатья 27. Следующая статья"
    chunks = LangchainDocumentSplitter(token_counter=real_counter).split([RawDocument(text)], "legal")
    assert [chunk.metadata["article_number"] for chunk in chunks] == ["26", "26-1", "27"]
    assert "26-1" in chunks[1].metadata["section"]
    assert chunks[1].metadata["domain_metadata"]["article"] == "26-1"
