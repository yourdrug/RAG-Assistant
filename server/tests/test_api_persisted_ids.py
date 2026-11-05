"""Admin list responses require the identities of persisted entities."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from application.act_versioning import ActVersioningService
from application.benchmark import BenchmarkRunService
from domain.benchmark.run import BenchmarkRun
from domain.knowledge.acts.models import ActVersion
from presentation.api.routes.admin_act_versions import list_act_versions
from presentation.api.routes.benchmark_admin import HistoryQuery, benchmark_history
from presentation.api.schemas.auth import CurrentUser
from tests.fakes import FakeUnitOfWorkFactory


@pytest.fixture
def admin() -> CurrentUser:
    return CurrentUser(id=1, email="admin@example.com", role="admin", kind="internal", is_active=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("run_id", [None, 42])
async def test_benchmark_history_requires_persisted_id(
    run_id: int | None, admin: CurrentUser, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = BenchmarkRunService(FakeUnitOfWorkFactory())
    monkeypatch.setattr(service, "list", AsyncMock(return_value=([BenchmarkRun(id=run_id)], 1)))

    if run_id is None:
        with pytest.raises(RuntimeError, match="BenchmarkRun saved with None id"):
            await benchmark_history(options=HistoryQuery(), _admin=admin, service=service)
    else:
        result = await benchmark_history(options=HistoryQuery(), _admin=admin, service=service)
        assert result.total == 1
        assert result.points[0].run_id == run_id


@pytest.mark.asyncio
@pytest.mark.parametrize("version_id", [None, 42])
@pytest.mark.parametrize("review", ["all", "pending"])
async def test_act_version_list_requires_persisted_id(
    version_id: int | None, review: str, admin: CurrentUser
) -> None:
    factory = FakeUnitOfWorkFactory()
    factory._uow.fake_act_versions._versions[42] = ActVersion(id=version_id, act_id=None, document_id=7)
    service = ActVersioningService(factory, MagicMock())

    if version_id is None:
        with pytest.raises(RuntimeError, match="ActVersion saved with None id"):
            await list_act_versions(review=review, limit=100, offset=0, _admin=admin, service=service)
    else:
        result = await list_act_versions(review=review, limit=100, offset=0, _admin=admin, service=service)
        assert result.total == 1
        assert result.versions[0].id == version_id
