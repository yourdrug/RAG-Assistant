"""Dataset acceptance: thresholds, noise, composition, regression on unchanged questions, judge agreement."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest

from application.benchmark.quality import BenchmarkQualityService
from application.benchmark.regression import comparable_rows, regression_report
from application.benchmark.result_service import BenchmarkResultService
from domain.benchmark.agreement import HumanLabel, LabelMetric, agreement, judge_agreement, ranks
from domain.benchmark.composition import dataset_composition
from domain.benchmark.policy import (
    AcceptanceThresholds,
    DatasetPolicy,
    GateStatus,
    JudgeIdentity,
    NoiseEstimate,
    allowed_drops,
    metric_noise,
    question_weights,
)
from domain.benchmark.run import BenchmarkRun
from domain.shared.exceptions import EntityNotFoundError, ValidationError
from infrastructure.benchmark.evaluation.run_manifest import source_digest
from infrastructure.benchmark.reporting.benchmark_history_adapter import BenchmarkHistoryAdapter
from tests.fakes import FakeUnitOfWorkFactory


if TYPE_CHECKING:
    from pathlib import Path

    from domain.benchmark.repositories import BenchmarkRunRepository


JUDGE = {"judge_model": "judge-a"}
FRAGMENT = {"source": "a.docx", "text": "ставка"}


def result(
    question_id: int, *, digest: str = "h", correctness: float = 9.0, recall: float = 1.0
) -> dict[str, Any]:
    return {
        "id": question_id,
        "question": f"q{question_id}",
        "run": 1,
        "question_hash": digest,
        "answer": "a",
        "outcome": "generated",
        "annotations": {"expected_fragments": [FRAGMENT]},
        "evidence_metrics": {"fragment_recall_at_k": recall, "context_fragment_recall": recall},
        "generator_metrics": {"correctness": correctness, "faithfulness": 9.0, "relevancy": 9.0},
        "retriever_metrics": {"hit_rate": 1, "mrr": 1.0, "avg_similarity": 0.0},
        "context_metrics": {},
        "latency_sec": 1.0,
        "judge_rubric_version": "r1",
    }


def run(run_id: int, rows: list[dict[str, Any]], *, days_ago: int = 0, **summary: Any) -> BenchmarkRun:
    return BenchmarkRun(
        id=run_id,
        dataset="belarus_law-main",
        config_json=dict(JUDGE),
        summary_metrics=summary,
        per_question_results=rows,
        creation_date=datetime.now(UTC) - timedelta(days=days_ago),
    )


async def stored_id(runs: BenchmarkRunRepository, benchmark_run: BenchmarkRun) -> int:
    saved = await runs.create(benchmark_run)

    if saved.id is None:
        message = "fake repository assigns ids"
        raise AssertionError(message)

    return saved.id


def checks_by_metric(report: Any, kind: str) -> dict[str, dict[str, Any]]:
    return {check["metric"]: check for check in report.checks if check["kind"] == kind}


def test_changed_questions_are_excluded_from_baseline() -> None:
    baseline = [result(1), result(2, correctness=9.0), result(3)]
    current = [result(1), result(2, digest="edited", correctness=2.0), result(4)]

    kept_current, kept_baseline, excluded = comparable_rows(current, baseline) or ([], [], [])

    assert [row["id"] for row in kept_current] == [1]
    assert [row["id"] for row in kept_baseline] == [1]
    assert {(item["key"], item["reason"]) for item in excluded} == {
        ("2#1", "changed"),
        ("4#1", "added"),
        ("3#1", "removed"),
    }


def test_regression_ignores_drop_on_edited_question() -> None:
    previous = run(1, [result(1), result(2)], days_ago=1, avg_correctness=9.0)
    # The edited question scores badly, but it is not comparable with its old version.
    current = run(2, [result(1), result(2, digest="edited", correctness=1.0)], avg_correctness=5.0)

    report = regression_report(current, previous, None, BenchmarkHistoryAdapter())

    assert report.passed
    assert report.compared_questions == 1
    assert checks_by_metric(report, "baseline")["correctness"]["current"] == 9.0
    assert [item["reason"] for item in report.excluded_questions] == ["changed"]


def test_regression_fails_on_unchanged_question_drop() -> None:
    previous = run(1, [result(1), result(2)], days_ago=1, avg_correctness=9.0)
    current = run(2, [result(1), result(2, correctness=1.0)], avg_correctness=5.0)

    report = regression_report(current, previous, None, BenchmarkHistoryAdapter())

    assert not report.passed
    assert checks_by_metric(report, "baseline")["correctness"]["failed"]


def test_legacy_baseline_with_other_questions_is_not_compared() -> None:
    previous = BenchmarkRun(id=1, config_json={"dataset_hash": "old"}, summary_metrics={"hit_rate": 0.9})
    current = BenchmarkRun(id=2, config_json={"dataset_hash": "new"}, summary_metrics={"hit_rate": 0.1})

    report = regression_report(current, previous, None, BenchmarkHistoryAdapter())

    assert not report.passed
    assert report.status is GateStatus.INSUFFICIENT_DATA
    assert not report.baseline_comparable
    assert not checks_by_metric(report, "baseline")


@pytest.mark.parametrize(
    "thresholds,status",
    [
        ({"min_correctness": 8.0}, GateStatus.PASSED),
        ({"min_correctness": 9.5}, GateStatus.FAILED),
        ({"min_fragment_recall_at_k": 1.0}, GateStatus.PASSED),
        # A configured threshold whose metric is absent is not a pass.
        ({"max_false_answer_rate": 0.0}, GateStatus.INSUFFICIENT_DATA),
    ],
)
def test_acceptance_thresholds(thresholds: dict[str, float], status: GateStatus) -> None:
    current = run(2, [result(1)], avg_correctness=9.0, avg_fragment_recall_at_k=1.0)
    previous = run(1, [result(1)], days_ago=1, avg_correctness=9.0, avg_fragment_recall_at_k=1.0)
    policy = DatasetPolicy(dataset="belarus_law-main", thresholds=AcceptanceThresholds.from_dict(thresholds))

    report = regression_report(current, previous, policy, BenchmarkHistoryAdapter())

    assert report.status is status
    assert report.passed is (status is GateStatus.PASSED)
    assert checks_by_metric(report, "acceptance")


def test_empty_run_without_baseline_is_insufficient() -> None:
    policy = DatasetPolicy(
        dataset="belarus_law-main", thresholds=AcceptanceThresholds.from_dict({"min_correctness": 8.0})
    )

    report = regression_report(run(2, []), None, policy, BenchmarkHistoryAdapter())

    assert report.status is GateStatus.INSUFFICIENT_DATA
    assert not report.passed
    assert checks_by_metric(report, "acceptance")["min_correctness"]["insufficient"]


def test_all_questions_changed_is_insufficient() -> None:
    previous = run(1, [result(1), result(2)], days_ago=1, avg_correctness=9.0)
    current = run(2, [result(1, digest="x"), result(2, digest="y")], avg_correctness=9.0)

    report = regression_report(current, previous, None, BenchmarkHistoryAdapter())

    assert report.status is GateStatus.INSUFFICIENT_DATA
    assert report.compared_questions == 0


def test_baseline_of_another_judge_rubric_is_not_compared() -> None:
    previous = run(1, [{**result(1), "judge_rubric_version": "r0"}], days_ago=1, avg_correctness=9.0)
    current = run(2, [result(1)], avg_correctness=9.0)
    previous.llm_evaluated = current.llm_evaluated = True

    report = regression_report(current, previous, None, BenchmarkHistoryAdapter())

    assert report.status is GateStatus.INSUFFICIENT_DATA
    assert not checks_by_metric(report, "baseline")
    assert "rubric r0" in report.warnings[0]

    same = regression_report(
        current, run(1, [result(1)], days_ago=1, avg_correctness=9.0), None, BenchmarkHistoryAdapter()
    )
    assert same.baseline_comparable


def test_thresholds_are_bounded() -> None:
    with pytest.raises(ValidationError):
        AcceptanceThresholds(min_correctness=11)

    with pytest.raises(ValidationError):
        AcceptanceThresholds(max_false_answer_rate=-0.1)


def test_noise_widens_allowed_drop_only_for_the_same_judge() -> None:
    judge = JudgeIdentity(model="judge-a", rubric_version="r1")
    noise = NoiseEstimate(sigma={"correctness": 0.8}, judge=judge, run_ids=[1, 2, 3])

    drops, warnings = allowed_drops(noise, judge)
    other, other_warnings = allowed_drops(noise, JudgeIdentity(model="judge-b", rubric_version="r1"))

    assert drops["correctness"] == -1.6
    assert not warnings
    assert other["correctness"] == -1.0
    assert "judge-a" in other_warnings[0]


def test_zero_noise_still_allows_one_question_to_change() -> None:
    judge = JudgeIdentity(model="judge-a", rubric_version="r1")
    noise = NoiseEstimate(sigma={"correctness": 0.0, "hit_rate": 0.0}, judge=judge, run_ids=[1, 2, 3])
    weights = question_weights({"correctness_evaluated_count": 40, "hit_rate_evaluated_count": 20})

    drops, _ = allowed_drops(noise, judge, weights)

    # One whole question plus one rounding step of the stored summary.
    assert weights == {"correctness": 0.35, "hit_rate": 0.0501}
    assert drops["correctness"] == -0.35
    assert drops["hit_rate"] == -0.0501
    # Measured noise wider than one question is kept.
    wide = NoiseEstimate(sigma={"correctness": 0.5}, judge=judge, run_ids=[1, 2, 3])
    assert allowed_drops(wide, judge, weights)[0]["correctness"] == -1.0
    # A drop of exactly one question passes despite float error in the delta.
    assert question_weights({"hit_rate_evaluated_count": 33})["hit_rate"] > 0.9091 - 0.8788
    # Without counts (old summaries) no floor is invented.
    assert question_weights({"correctness_evaluated_count": 0}) == {}


def test_metric_noise_is_sample_standard_deviation() -> None:
    noise = metric_noise([{"correctness": 7.0}, {"correctness": 8.0}, {"correctness": 9.0}])

    assert noise == {"correctness": 1.0}
    # A metric missing from one run is not estimated from the remaining runs.
    assert metric_noise([{"hit_rate": 0.5}, {}]) == {}


def test_composition_reports_missing_case_types() -> None:
    factual = {"annotations": {"expected_fragments": [FRAGMENT], "required_facts": ["a", "b"]}}
    abstain = {"annotations": {"answer_type": "not_in_corpus"}, "history": [{"role": "user", "content": "x"}]}
    questions = [factual] * 30 + [abstain] * 10

    report = dataset_composition(questions, pack_key="belarus_law")

    assert report.total == 40
    assert report.shares["factual_with_fragments"] == 0.75
    assert report.shares["abstention"] == 0.25
    assert report.failures == ["dated: 0% < 10%"]
    assert dataset_composition(questions, pack_key="support").passed
    assert "active questions" in dataset_composition(questions[:5]).failures[0]


def test_agreement_metrics() -> None:
    assert ranks([3.0, 1.0, 3.0]) == [2.5, 1.0, 2.5]

    perfect = agreement([(9.0, 9.0), (2.0, 3.0), (8.0, 7.0), (4.0, 5.0)])

    assert perfect["n"] == 4
    assert perfect["mae"] == 0.75
    assert perfect["spearman"] == 1.0
    assert perfect["kappa"] == 1.0
    assert agreement([])["mae"] is None


def test_judge_agreement_averages_labelers() -> None:
    rows = [result(1, correctness=9.0), result(2, correctness=3.0)]
    labels = [
        HumanLabel(run_id=1, question_key="1#1", metric=LabelMetric.CORRECTNESS, score=8.0, labeler_id=1),
        HumanLabel(run_id=1, question_key="1#1", metric=LabelMetric.CORRECTNESS, score=10.0, labeler_id=2),
        HumanLabel(run_id=1, question_key="2#1", metric=LabelMetric.CORRECTNESS, score=2.0, labeler_id=1),
    ]

    report = judge_agreement(rows, labels)

    assert report["correctness"]["n"] == 2
    assert report["correctness"]["mae"] == 0.5
    assert report["faithfulness"]["n"] == 0


def test_label_validation() -> None:
    with pytest.raises(ValidationError):
        HumanLabel(run_id=1, question_key="1#1", metric=LabelMetric.CORRECTNESS, score=11, labeler_id=1)

    with pytest.raises(ValueError, match="relevancy"):
        HumanLabel(run_id=1, question_key="1#1", metric="relevancy", score=5, labeler_id=1)  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        HumanLabel(
            run_id=1,
            question_key="1#1",
            metric=LabelMetric.CORRECTNESS,
            score=5,
            labeler_id=1,
            note="x" * 1001,
        )


@pytest.mark.asyncio
async def test_labels_are_upserted_per_labeler_and_bound_to_run_cases() -> None:
    factory = FakeUnitOfWorkFactory()
    stored_run = await factory._uow.benchmark_runs.create(run(0, [result(1)]))
    stored = stored_run.id or 0
    service = BenchmarkQualityService(factory)

    await service.save_label(
        run_id=stored, question_key="1#1", metric="correctness", score=4, note=None, user_id=5
    )
    await service.save_label(
        run_id=stored, question_key="1#1", metric="correctness", score=6, note=None, user_id=5
    )

    labels = await service.list_labels(stored)

    assert [(label.score, label.labeler_id) for label in labels] == [(6, 5)]

    with pytest.raises(EntityNotFoundError):
        await service.save_label(
            run_id=stored, question_key="9#1", metric="correctness", score=6, note=None, user_id=5
        )

    with pytest.raises(EntityNotFoundError):
        await service.save_label(
            run_id=999, question_key="1#1", metric="correctness", score=6, note=None, user_id=5
        )

    assert await service.delete_label(run_id=stored, question_key="1#1", metric="correctness", user_id=5)
    assert not await service.delete_label(run_id=stored, question_key="1#1", metric="correctness", user_id=5)


MANIFEST = {"configuration_version": "c1", "code": {"revision": "abc"}, "limitations": ["x"]}


def manifested(*, manifest: dict[str, Any] = MANIFEST, **summary: Any) -> BenchmarkRun:
    return run(0, [{**result(1), "run_manifest": manifest}], **summary)


@pytest.mark.asyncio
async def test_noise_requires_repeated_runs_of_one_configuration() -> None:
    factory = FakeUnitOfWorkFactory()
    runs = factory._uow.benchmark_runs
    ids = [
        await stored_id(runs, manifested(avg_correctness=value, hit_rate=0.9)) for value in (8.0, 9.0, 10.0)
    ]
    # Descriptive manifest fields do not make otherwise identical runs differ.
    described = await stored_id(
        runs, manifested(manifest={**MANIFEST, "limitations": []}, avg_correctness=9.0)
    )
    other_judge = await stored_id(
        runs, BenchmarkRun(dataset="belarus_law-main", config_json={"judge_model": "judge-b"})
    )
    other_config = await stored_id(
        runs, manifested(manifest={**MANIFEST, "configuration_version": "c2"}, avg_correctness=9.0)
    )
    unverifiable = await stored_id(runs, run(0, [result(1)], avg_correctness=9.0))
    # Recorded responses repeat earlier answers and scores; a run that missed every lookup is still real.
    judge_replayed = await stored_id(
        runs, manifested(avg_correctness=9.0, llm_replay={"mode": "record", "hits": 3, "misses": 0})
    )
    answer_replayed = await stored_id(
        runs, run(0, [{**result(1), "run_manifest": MANIFEST, "llm_replayed": True}], avg_correctness=9.0)
    )
    recorded_only = await stored_id(
        runs, manifested(avg_correctness=9.0, llm_replay={"mode": "record", "hits": 0, "misses": 5})
    )
    service = BenchmarkQualityService(factory)

    policy = await service.measure_noise("belarus_law-main", ids, None)

    assert policy.noise is not None
    assert policy.noise.sigma["correctness"] == 1.0
    assert policy.noise.judge == JudgeIdentity(model="judge-a", rubric_version="r1")
    assert (await service.measure_noise("belarus_law-main", [*ids[:2], described], None)).noise

    assert (await service.measure_noise("belarus_law-main", [*ids[:2], recorded_only], None)).noise

    for rejected in (other_judge, other_config, unverifiable, judge_replayed, answer_replayed):
        with pytest.raises(ValidationError):
            await service.measure_noise("belarus_law-main", [*ids[:2], rejected], None)

    with pytest.raises(ValidationError):
        await service.measure_noise("support-main", ids, None)

    with pytest.raises(ValidationError):
        await service.measure_noise("belarus_law-main", ids[:2], None)


@pytest.mark.asyncio
async def test_thresholds_survive_noise_updates() -> None:
    factory = FakeUnitOfWorkFactory()
    service = BenchmarkQualityService(factory)

    await service.update_thresholds("belarus_law-main", {"min_correctness": 7.5}, user_id=1)
    policy = await service.get_policy("belarus_law-main")

    assert policy.thresholds.min_correctness == 7.5
    assert (await service.get_policy("support-main")).thresholds == AcceptanceThresholds()


@pytest.mark.asyncio
async def test_calibrations_are_kept_as_history() -> None:
    factory = FakeUnitOfWorkFactory()
    first = await stored_id(factory._uow.benchmark_runs, run(0, [result(1, correctness=9.0)]))
    second = await stored_id(factory._uow.benchmark_runs, run(0, [result(1, correctness=3.0)]))
    service = BenchmarkQualityService(factory)

    with pytest.raises(ValidationError):
        await service.save_calibration(first, user_id=5)

    policy = None

    for run_id in (first, second):
        await service.save_label(
            run_id=run_id, question_key="1#1", metric="correctness", score=8, note=None, user_id=5
        )
        policy = await service.save_calibration(run_id, user_id=5)

    history = await service.list_calibrations("belarus_law-main")

    assert [item.run_id for item in history] == [second, first]
    assert history[0].metrics["correctness"]["mae"] == 5.0
    assert history[1].metrics["correctness"]["mae"] == 1.0
    assert all(item.created_by == 5 for item in history)
    # The policy holds only the latest calibration; earlier ones stay in the history.
    assert policy is not None
    assert policy.calibration is not None
    assert policy.calibration.run_id == second
    assert await service.list_calibrations("support-main") == []


def test_source_digest_tracks_python_modules_only(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    module = tmp_path / "pkg" / "rag.py"
    module.write_text("TOP_K = 5\n")
    (tmp_path / "notes.txt").write_text("not code")
    before = source_digest(tmp_path)

    (tmp_path / "notes.txt").write_text("changed, still not code")
    assert source_digest(tmp_path) == before

    module.write_text("TOP_K = 6\n")
    assert source_digest(tmp_path) != before

    module.write_text("TOP_K = 5\n")
    module.rename(tmp_path / "pkg" / "renamed.py")
    assert source_digest(tmp_path) != before


@pytest.mark.asyncio
async def test_stored_noise_from_replayed_runs_does_not_widen_regression_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = FakeUnitOfWorkFactory()
    uow = factory._uow
    judge = JudgeIdentity(model="judge-a", rubric_version="r1")
    clean = [await stored_id(uow.benchmark_runs, manifested(avg_correctness=v)) for v in (8.0, 9.0)]
    replayed = await stored_id(
        uow.benchmark_runs, manifested(avg_correctness=9.0, llm_replay={"mode": "record", "hits": 3})
    )
    previous = run(0, [result(1, correctness=9.0)], correctness=9.0)
    current = await stored_id(uow.benchmark_runs, run(0, [result(1, correctness=7.5)], correctness=7.5))
    monkeypatch.setattr(uow.benchmark_runs, "get_baseline", AsyncMock(return_value=previous))
    service = BenchmarkResultService(factory)

    async def check(run_ids: list[int]) -> tuple[float, list[str]]:
        # sigma=1.0 allows a 2.0 drop; the default allows 1.0.
        noise = NoiseEstimate(sigma={"correctness": 1.0}, judge=judge, run_ids=run_ids)
        await uow.benchmark_quality.save_policy(DatasetPolicy(dataset="belarus_law-main", noise=noise))
        output = await service.check_regression(current, BenchmarkHistoryAdapter())
        threshold = next(
            r.threshold for r in output.results if r.kind == "baseline" and r.metric == "correctness"
        )
        return threshold, output.warnings

    threshold, warnings = await check(clean)
    assert threshold == -2.0
    assert not any("recorded LLM responses" in w for w in warnings)

    threshold, warnings = await check([*clean, replayed])
    assert threshold == -1.0
    assert any(f"runs [{replayed}] with recorded LLM responses" in w for w in warnings)
    assert not any(w.startswith("noise not measured") for w in warnings)
