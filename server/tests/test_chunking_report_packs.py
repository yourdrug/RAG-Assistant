"""Offline benchmark tooling uses the selected pack, never a union of profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
from typing import Any

import pytest

from composition.domain_packs import PACK_REGISTRATIONS, selected_pack
from domain.benchmark.question import validate_annotations


pytestmark = pytest.mark.skipif(
    "support" not in PACK_REGISTRATIONS or "internal" not in PACK_REGISTRATIONS,
    reason="Optional packs are disabled in the core-only run",
)


@pytest.fixture(scope="module")
def report_tool() -> dict[str, Any]:
    script = Path(__file__).resolve().parents[2] / "scripts/benchmark/chunking_report.py"
    return runpy.run_path(str(script))


@pytest.mark.parametrize(
    "pack_key,profile_key",
    [
        ("support", "support_guide"),
        ("internal", "internal_instruction"),
    ],
)
def test_offline_splitter_uses_only_selected_pack(
    report_tool: dict[str, Any], tmp_path: Path, pack_key: str, profile_key: str
) -> None:
    (tmp_path / "guide.txt").write_text(
        "Инструкция по подключению\n" + "Откройте настройки и выберите подключение. " * 10
    )
    docs = report_tool["split_corpus"](tmp_path, pack_key=pack_key)
    assert len(docs) == 1
    assert docs[0]["domain"] == profile_key
    assert docs[0]["chunks"]


def test_missing_acceptance_sets_fail_explicitly(
    report_tool: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["chunking_report.py", "--pack", "generic"])
    with pytest.raises(SystemExit) as exc:
        report_tool["parse_args"]()
    assert exc.value.code == 2
    assert "No benchmark sets" in capsys.readouterr().err


def test_custom_dataset_does_not_compare_support_with_law_baseline(
    report_tool: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "chunking_report.py",
            "--pack",
            "support",
            "--datasets",
            str(tmp_path),
        ],
    )
    args = report_tool["parse_args"]()
    assert args.pack == "support"
    assert args.datasets == [tmp_path]
    assert args.baseline is None


@pytest.mark.parametrize("pack_key", ["support", "internal"])
def test_pack_defaults_to_its_bundled_corpus_and_questions(
    report_tool: dict[str, Any], monkeypatch: pytest.MonkeyPatch, pack_key: str
) -> None:
    monkeypatch.setattr("sys.argv", ["chunking_report.py", "--pack", pack_key])
    args = report_tool["parse_args"]()
    base = selected_pack(pack_key).benchmark_sets[0].parent
    assert args.corpus == base / "corpus"
    assert args.datasets == [base / "chunking"]


def test_default_offline_run_keeps_law_pack(
    report_tool: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sys.argv", ["chunking_report.py"])
    args = report_tool["parse_args"]()
    registry, _settings, _parser, _splitter = report_tool["build_pipeline"](args.pack)
    assert args.pack == "belarus_law"
    assert {profile.key for profile in registry.all()} == {"general", "legal", "decree", "contract"}


def test_wrong_pack_baseline_is_rejected_before_reading_corpus(
    report_tool: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"pack": "internal"}))
    main = report_tool["main"]
    monkeypatch.setitem(
        main.__globals__, "parse_args", lambda: argparse.Namespace(pack="support", baseline=baseline)
    )
    with pytest.raises(SystemExit, match="different domain pack"):
        main()


@pytest.fixture(scope="module", params=["support", "internal"])
def synthetic_measurement(
    request: pytest.FixtureRequest, report_tool: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    pack_key = request.param
    base = selected_pack(pack_key).benchmark_sets[0].parent
    specification = json.loads((base / "acceptance.json").read_text())
    documents = report_tool["split_corpus"](base / "corpus", pack_key=pack_key)
    chunks = [chunk for document in documents for chunk in document["chunks"]]
    report = {"pack": pack_key, "documents": documents, "domains": {}}
    by_source = {document["file"]: document for document in documents}
    all_questions = []

    for domain in specification["minimums"]:
        questions = report_tool["labelled_questions"]([base / "chunking"], domain)
        all_questions.extend(questions)
        rankings = report_tool["bm25_rankings"](questions, chunks)
        report["domains"][domain] = {
            "dataset_questions": len(questions),
            **report_tool["reachability"](questions, by_source),
            "retrieval": {"bm25": report_tool["retrieval_metrics"](questions, rankings, chunks)},
        }

    rankings = report_tool["bm25_rankings"](all_questions, chunks)
    report["overall"] = {
        "dataset_questions": len(all_questions),
        **report_tool["reachability"](all_questions, by_source),
        "retrieval": {"bm25": report_tool["retrieval_metrics"](all_questions, rankings, chunks)},
    }

    return report, specification


def test_synthetic_corpus_passes_provisional_offline_floors(
    report_tool: dict[str, Any], synthetic_measurement: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    report, specification = synthetic_measurement
    assert specification["provenance"] == "synthetic"
    assert specification["status"] == "provisional"
    assert report_tool["acceptance_failures"](report, specification) == []


@pytest.mark.parametrize("broken_field", ["pack", "documents", "domains", "overall"])
def test_acceptance_rejects_wrong_pack_corpus_or_missing_metrics(
    report_tool: dict[str, Any],
    synthetic_measurement: tuple[dict[str, Any], dict[str, Any]],
    broken_field: str,
) -> None:
    report, specification = synthetic_measurement
    broken = {**report, broken_field: {}}
    assert report_tool["acceptance_failures"](broken, specification)


def test_acceptance_rejects_retrieval_regression(
    report_tool: dict[str, Any], synthetic_measurement: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    report, specification = synthetic_measurement
    broken = json.loads(json.dumps(report))
    broken["overall"]["retrieval"]["bm25"]["recall_at_5"] = 0.0
    failures = report_tool["acceptance_failures"](broken, specification)
    assert any("overall.bm25.recall_at_5" in failure for failure in failures)


@pytest.mark.parametrize("pack_key", ["support", "internal"])
def test_synthetic_questions_have_valid_annotations_and_verbatim_evidence(pack_key: str) -> None:
    base = selected_pack(pack_key).benchmark_sets[0].parent
    questions = json.loads((base / "questions.json").read_text())
    validator = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/benchmark/validate_v2.py"))
    validator["validate_cases"](questions)
    assert len({question["case_id"] for question in questions}) == len(questions)
    assert any(question["annotations"]["expected_refusal"] for question in questions)
    offline = {
        question["case_id"]: question
        for path in (base / "chunking").glob("*.json")
        for question in json.loads(path.read_text())
    }
    assert offline == {
        question["case_id"]: question
        for question in questions
        if question["annotations"]["expected_fragments"]
    }

    for question in questions:
        annotations = question["annotations"]
        validate_annotations(annotations)
        assert "synthetic" in question["tags"]
        assert question["dataset"].startswith(f"synthetic-{pack_key}-")

        for fragment in annotations["expected_fragments"]:
            source = (base / "corpus" / fragment["source"]).read_text()
            assert "СИНТЕТИЧЕСКИЙ" in source
            assert fragment["text"] in source
