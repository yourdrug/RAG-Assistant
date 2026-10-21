"""Status classification stays consistent across domain lifecycles."""

import pytest

from domain.benchmark.sweep import BenchmarkSweepStatus
from domain.knowledge.enums import DocumentStatus
from domain.knowledge.index_sync import OutboxStatus
from domain.runtime.jobs import BackgroundJobStatus
from domain.shared.enums import LifecycleStatus


@pytest.mark.parametrize(
    "status_type, terminal, active",
    [
        (DocumentStatus, {"done", "failed"}, {"pending", "processing", "indexing"}),
        (BackgroundJobStatus, {"done", "failed"}, {"pending", "running"}),
        (BenchmarkSweepStatus, {"done", "failed", "cancelled"}, {"pending", "running"}),
        (OutboxStatus, {"done", "dead_letter"}, {"pending", "in_progress"}),
    ],
)
def test_status_classification(
    status_type: type[DocumentStatus | BackgroundJobStatus | BenchmarkSweepStatus | OutboxStatus],
    terminal: set[str],
    active: set[str],
) -> None:
    for status in status_type:
        assert status.is_terminal is (status.value in terminal)
        assert status.is_active is (status.value in active)


def test_lifecycle_defaults_are_neutral() -> None:
    class NeutralStatus(LifecycleStatus):
        UNKNOWN = "unknown"

    assert not NeutralStatus.UNKNOWN.is_terminal
    assert not NeutralStatus.UNKNOWN.is_active


def test_lifecycle_inherits_classification_and_keeps_overrides_local() -> None:
    class ConfiguredStatus(LifecycleStatus, terminal=frozenset({"done"}), active=frozenset({"waiting"})):
        pass

    class InheritedStatus(ConfiguredStatus):
        WAITING = "waiting"
        DONE = "done"

    class ResetStatus(ConfiguredStatus, terminal=frozenset(), active=frozenset()):
        DONE = "done"

    assert InheritedStatus.WAITING.is_active
    assert InheritedStatus.DONE.is_terminal
    assert not ResetStatus.DONE.is_terminal
    assert not ResetStatus.DONE.is_active
