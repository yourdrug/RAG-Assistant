"""Architecture tests for the ML client provider port."""

from __future__ import annotations

from application.ports.ml_clients import AsyncSemaphorePort, MLClientPort
from infrastructure.ml.clients.client_registry import MLClientRegistry
from fakes import FakeMLClientRegistry


def test_registry_implements_ml_client_port() -> None:
    registry = MLClientRegistry()

    assert isinstance(registry, MLClientPort)
    assert isinstance(registry.generation_semaphore, AsyncSemaphorePort)


def test_test_double_implements_ml_client_port() -> None:
    assert isinstance(FakeMLClientRegistry(), MLClientPort)
