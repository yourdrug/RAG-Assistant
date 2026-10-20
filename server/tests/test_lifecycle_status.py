"""Status classification stays consistent across domain lifecycles."""

from enum import StrEnum

import pytest

from domain.entities.vector_outbox_entry import OutboxStatus
from domain.value_objects.document_status import DocumentStatus
from domain.value_objects.job_status import BackgroundJobStatus
from domain.value_objects.lifecycle_status import LifecycleStatusMixin
from domain.value_objects.sweep_status import BenchmarkSweepStatus


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
    class NeutralStatus(LifecycleStatusMixin, StrEnum):
        UNKNOWN = "unknown"

    assert not NeutralStatus.UNKNOWN.is_terminal
    assert not NeutralStatus.UNKNOWN.is_active
