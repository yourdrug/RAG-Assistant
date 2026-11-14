"""Each replica must prove its own event loop is still updating its heartbeat."""

import socket

import pytest

from infrastructure import worker_health as health


def test_replica_health_keys_are_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "gethostname", lambda: "replica-one")
    first = health.health_check_key("background_tasks")
    monkeypatch.setattr(socket, "gethostname", lambda: "replica-two")
    assert health.health_check_key("background_tasks") != first
    assert health.health_check_key("batch_tasks") != first
