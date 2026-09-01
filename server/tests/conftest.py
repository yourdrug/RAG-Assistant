"""conftest.py -- Shared fixtures and mocks for all tests.

Mocks heavy optional dependencies that aren't installed in dev/test.
Provides common fixtures for testing application services without
Postgres, Qdrant, Redis, or LLM.
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Ensure required env vars are set before any module imports Settings().
# In CI there is no .env file, so Settings() at config.py:258 would fail
# with 12 missing-field errors.  setdefault keeps local .env values intact.
# ---------------------------------------------------------------------------
_TEST_DEFAULTS = {
    "DB_HOST": "localhost",
    "DB_PORT": "5432",
    "DB_USER": "test",
    "DB_PASSWORD": "test",
    "DB_NAME": "test",
    "QDRANT_URL": "http://localhost:6333",
    "COLLECTION_NAME": "test_collection",
    "ALLOWED_ORIGINS": "*",
    "ML_PROVIDER": "tei",
    "LLM_PROVIDER": "ollama",
    "LLM_MODEL": "qwen2.5:7b",
    "JWT_SECRET_KEY": "test-secret-key",
    "DATA_DIR": "/tmp/rag_test",
}
for _key, _val in _TEST_DEFAULTS.items():
    os.environ.setdefault(_key, _val)

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
