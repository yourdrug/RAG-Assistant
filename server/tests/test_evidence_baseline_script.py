"""Offline checks for checkpointed evidence validation phases."""

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def baseline_script() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "benchmark/validate_evidence_baseline.py"
    spec = importlib.util.spec_from_file_location("evidence_baseline_script", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_completed_cases_resume_without_generation_or_judging(
    baseline_script: ModuleType, tmp_path: Path
) -> None:
    args = SimpleNamespace(concurrency=2, out=tmp_path / "report.json")
    cases = [{"id": 7}]
    report = {"generated": {"7": {}}, "results": [{"id": 7}]}
    await baseline_script.generate_cases(cases, None, args, report)
    await baseline_script.evaluate_cases(cases, args, report)
    assert report == {"generated": {"7": {}}, "results": [{"id": 7}]}
    assert not args.out.exists()


@pytest.mark.asyncio
async def test_phase_failure_waits_for_successful_siblings(baseline_script: ModuleType) -> None:
    completed = []

    async def succeed() -> None:
        completed.append("saved")

    async def fail() -> None:
        raise ValueError

    with pytest.raises(ExceptionGroup) as error:
        await baseline_script.complete_phase([fail(), succeed()])
    assert completed == ["saved"]
    assert len(error.value.exceptions) == 1
    assert isinstance(error.value.exceptions[0], ValueError)
