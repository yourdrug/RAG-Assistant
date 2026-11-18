"""conftest.py -- Shared fixtures and mocks for all tests.

Mocks heavy optional dependencies that aren't installed in dev/test.
Provides common fixtures for testing application services without
Postgres, Qdrant, Redis, or LLM.
"""

import ast
from collections.abc import Iterator
from functools import cache
import importlib
import os
from pathlib import Path
import sys
from tempfile import gettempdir
from unittest.mock import MagicMock

import pytest

from tests.fakes import FakeChatRAGPort, FakeEventBus, FakeMLClientRegistry, FakeUnitOfWorkFactory


# ---------------------------------------------------------------------------
# Ensure app directory is on sys.path for all tests
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

# ---------------------------------------------------------------------------
# Ensure required env vars are set before any module imports Settings().
# In CI there is no .env file, so the module-level Settings() in config/__init__.py would fail
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
    "EMBEDDING_API_KEY": "test-embedding-key",
    "RERANK_API_KEY": "test-rerank-key",
    "LLM_MODEL": "qwen/qwen-2.5-7b-instruct",
    "LLM_API_KEY": "test-llm-key",
    "JWT_SECRET_KEY": "test-secret-key-for-unit-tests-only-0000",
    "DATA_DIR": str(Path(gettempdir()) / "rag_test"),
    "REDIS_PASSWORD": "test-redis-password",
    "S3_ACCESS_KEY": "test-access-key",
    "S3_SECRET_KEY": "test-secret-key-value",
    "RATE_LIMIT_ENABLED": "false",
}

for _key, _val in _TEST_DEFAULTS.items():
    os.environ.setdefault(_key, _val)


# Modules whose import makes a test module a test of the Belarusian law pack.
LAW_PACK_MODULES = (
    "domain.domain_packs",
    "application.domain_packs",
    "presentation.domain_packs",
    "infrastructure.domain_packs",
    "composition.domain_packs.belarus_law",
    "tests.law_pack",
)
LAW_PACK_PLUGIN = "tests.law_pack_plugin"
# Why the law pack is not importable in this run; None when it is.
law_pack_import_error: ImportError | None = None


def pytest_configure(config: pytest.Config) -> None:
    """Register the law pack fixtures only when the pack imports, so the core runs without it."""
    global law_pack_import_error  # noqa: PLW0603 -- set once per session before collection

    try:
        # Imported here, not at module level: the pack is optional for the core suite.
        plugin = importlib.import_module(LAW_PACK_PLUGIN)
    except ImportError as exc:
        law_pack_import_error = exc
    else:
        config.pluginmanager.register(plugin, LAW_PACK_PLUGIN)
        return

    if "not belarus" not in (config.option.markexpr or ""):
        message = (
            f"Belarusian law pack is not importable ({law_pack_import_error}); "
            "run the core suite with -m 'not belarus'"
        )
        raise pytest.UsageError(message)

    # Imported here, not at module level: the composition package loads ``config`` on import,
    # which needs the test env defaults above (CI has no .env file).
    from composition.domain_packs import PACK_REGISTRATIONS  # noqa: PLC0415

    # A core-only test installation registers no optional packs. Production always
    # loads the full registry, with no environment flag to select a subset.
    PACK_REGISTRATIONS.clear()


def pytest_report_header() -> str | None:
    if law_pack_import_error is None:
        return None

    return f"law pack: not importable ({law_pack_import_error}); its test modules are not collected"


def pytest_ignore_collect(collection_path: Path) -> bool | None:
    """Skip law pack test modules when the pack is unavailable instead of failing their import."""
    if law_pack_import_error is None or collection_path.suffix != ".py":
        return None

    return True if imports_law_pack(collection_path) else None


SERVER_DIR = Path(__file__).resolve().parents[1]


def helper_module_path(module: str) -> Path | None:
    """Source file of a ``tests.*`` module, so law pack imports are followed through helpers."""
    if module != "tests" and not module.startswith("tests."):
        return None

    base = SERVER_DIR.joinpath(*module.split("."))
    return next((path for path in (base.with_suffix(".py"), base / "__init__.py") if path.is_file()), None)


@cache
def imports_law_pack(path: Path, seen: frozenset[Path] = frozenset()) -> bool:
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        names = (
            [alias.name for alias in node.names]
            if isinstance(node, ast.Import)
            else [node.module or ""]
            if isinstance(node, ast.ImportFrom)
            else []
        )

        if any(
            name == module or name.startswith(f"{module}.") for name in names for module in LAW_PACK_MODULES
        ):
            return True

        for name in names:
            helper = helper_module_path(name)

            if helper is None or helper == path or helper in seen:
                continue

            if imports_law_pack(helper, seen | {path}):
                return True

    return False


@pytest.hookimpl(tryfirst=True)  # Before ``-m`` deselection reads the markers.
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    # Law pack tests are marked automatically, so ``-m "not belarus"`` is the core suite.
    for item in items:
        uses_fixture = "law_registry" in getattr(item, "fixturenames", ())

        if uses_fixture or imports_law_pack(item.path):
            item.add_marker(pytest.mark.belarus)

    if os.environ.get("RUN_INTEGRATION_TESTS") == "1":
        return

    skip_integration = pytest.mark.skip(reason="Set RUN_INTEGRATION_TESTS=1 to run integration tests")

    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


from infrastructure.ml.evidence_profiles.profiles import GenericEvidenceProfile, evidence_profile_scope  # noqa: E402
from infrastructure.resilience.circuit_breaker import init_breakers  # noqa: E402


# Initialize circuit breakers for all tests (required by step_generate, handle_relevance_gate, etc.)


init_breakers(fail_max=5, timeout_duration=300)


@pytest.fixture(autouse=True)
def reset_breakers_between_tests() -> Iterator[None]:
    """Reset circuit breaker state between tests to prevent cross-test contamination."""
    init_breakers(fail_max=5, timeout_duration=300)
    yield
    init_breakers(fail_max=5, timeout_duration=300)


@pytest.fixture(autouse=True)
def selected_evidence_profile(request: pytest.FixtureRequest) -> Iterator[None]:
    """Select evidence rules only where a test asks for them; no profile is selected otherwise.

    Production code must select a profile itself (a request, a sweep, a script), so an
    unscoped call fails here exactly as in production. ``@pytest.mark.generic_evidence``
    calls profile-dependent helpers directly with neutral rules; ``@pytest.mark.belarus``
    marks tests of the Belarusian law pack (``-m "not belarus"`` runs the core suite), whose
    evidence rules ``tests/law_pack_plugin.py`` selects.
    """
    if request.node.get_closest_marker("belarus") or not request.node.get_closest_marker("generic_evidence"):
        yield
        return

    with evidence_profile_scope(GenericEvidenceProfile()):
        yield


# Mock surya before any test module imports domain.ingestion
_surya_mock = MagicMock()
sys.modules.setdefault("surya", _surya_mock)
sys.modules.setdefault("surya.detection", _surya_mock)
sys.modules.setdefault("surya.recognition", _surya_mock)


# ---------------------------------------------------------------------------
# Common fakes for application-layer unit tests
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_uow_factory() -> FakeUnitOfWorkFactory:
    """Fake UnitOfWorkFactory for testing application services."""

    return FakeUnitOfWorkFactory()


@pytest.fixture
def fake_chat_rag_port() -> FakeChatRAGPort:
    """Fake ChatRAGPort for testing ChatService."""

    return FakeChatRAGPort()


@pytest.fixture
def fake_event_bus() -> FakeEventBus:
    """Fake EventBus for testing ConfigService."""

    return FakeEventBus()


@pytest.fixture
def fake_ml_clients() -> FakeMLClientRegistry:
    """Fake MLClientRegistry for testing ML-dependent services."""

    return FakeMLClientRegistry()


@pytest.fixture
def mock_vector_store() -> MagicMock:
    """Mock VectorSearch."""
    mock = MagicMock()
    mock.generate_embeddings.return_value = [0.1] * 128
    return mock


@pytest.fixture
def mock_file_storage() -> MagicMock:
    """Mock FileStorage."""
    mock = MagicMock()
    mock.supported_extensions = (".pdf", ".docx", ".txt", ".md")
    return mock
