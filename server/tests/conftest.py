"""conftest.py -- Shared fixtures and mocks for all tests.

Mocks heavy optional dependencies that aren't installed in dev/test.
Provides common fixtures for testing application services without
Postgres, Qdrant, Redis, or LLM.
"""

import sys
from unittest.mock import MagicMock

import pytest

# Mock surya before any test module imports domain.ingestion
_surya_mock = MagicMock()
sys.modules.setdefault("surya", _surya_mock)
sys.modules.setdefault("surya.detection", _surya_mock)
sys.modules.setdefault("surya.recognition", _surya_mock)


# ---------------------------------------------------------------------------
# Common fakes for application-layer unit tests
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_uow_factory():
    """Fake UnitOfWorkFactory for testing application services."""
    from tests.fakes import FakeUnitOfWorkFactory

    return FakeUnitOfWorkFactory()


@pytest.fixture
def fake_chat_rag_port():
    """Fake ChatRAGPort for testing ChatService."""
    from tests.fakes import FakeChatRAGPort

    return FakeChatRAGPort()


@pytest.fixture
def fake_event_bus():
    """Fake EventBus for testing ConfigService."""
    from tests.fakes import FakeEventBus

    return FakeEventBus()


@pytest.fixture
def fake_ml_clients():
    """Fake MLClientRegistry for testing ML-dependent services."""
    from tests.fakes import FakeMLClientRegistry

    return FakeMLClientRegistry()


@pytest.fixture
def mock_vector_store():
    """Mock VectorStoreRepository."""
    mock = MagicMock()
    mock.generate_embeddings.return_value = [0.1] * 128
    return mock


@pytest.fixture
def mock_file_storage():
    """Mock FileStorage."""
    mock = MagicMock()
    mock.supported_extensions = (".pdf", ".docx", ".doc", ".txt", ".md")
    return mock
