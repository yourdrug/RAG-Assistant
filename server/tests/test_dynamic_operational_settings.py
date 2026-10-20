"""Operational settings must affect real consumers and expose safe applied values."""

from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path

import pytest
from fastapi import HTTPException, UploadFile
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from config import Settings, settings
from domain.runtime.config import ConfigParameter
from domain.runtime.dynamic_params import DYNAMIC_FIELDS_MAP
from domain.runtime.jobs import BackgroundJobStatus
from infrastructure.database.models import BackgroundJobModel
from infrastructure.events.in_process_event_bus import InProcessEventBus
from infrastructure.ml.config.provider_configuration import SettingsProviderConfiguration
from infrastructure.repositories.misc.sqlalchemy_background_job_repository import (
    SQLAlchemyBackgroundJobRepository,
)
from presentation.api.dependencies import create_upload_config
from presentation.api.helpers import read_upload_with_limit


def test_operational_limits_validate_and_are_dynamic():
    assert {"max_upload_size_mb", "job_cleanup_days"} <= DYNAMIC_FIELDS_MAP.keys()
    for key, value in [("max_upload_size_mb", 0), ("max_upload_size_mb", 64), ("job_cleanup_days", 0)]:
        with pytest.raises(ValueError):
            Settings(**{key: value})
    nginx = (Path(__file__).parents[2] / "client/nginx.conf").read_text()
    assert "client_max_body_size 64m;" in nginx


@pytest.mark.asyncio
async def test_upload_snapshot_and_real_bytes_boundary(monkeypatch):
    monkeypatch.setattr(settings, "max_upload_size_mb", 1)
    first = create_upload_config(None)
    monkeypatch.setattr(settings, "max_upload_size_mb", 2)
    second = create_upload_config(None)
    data = b'x' * (1024 * 1024 + 1)
    with pytest.raises(HTTPException) as error:
        await read_upload_with_limit(UploadFile(BytesIO(data)), first.max_upload_size_mb * 1024 * 1024)
    assert error.value.status_code == 413
    assert (
        await read_upload_with_limit(UploadFile(BytesIO(data)), second.max_upload_size_mb * 1024 * 1024)
        == data
    )
    boundary = data[:-1]
    assert await read_upload_with_limit(UploadFile(BytesIO(boundary)), len(boundary)) == boundary


@pytest.mark.asyncio
async def test_cleanup_preserves_active_and_recent_jobs():
    engine = create_engine("sqlite://")
    BackgroundJobModel.__table__.create(engine)
    now = datetime.now(UTC)
    with Session(engine) as session:
        for i, status in enumerate(BackgroundJobStatus, 1):
            session.add(
                BackgroundJobModel(
                    id=i, job_type="test", status=status.value, creation_date=now - timedelta(days=40)
                )
            )
        session.add(
            BackgroundJobModel(
                id=20, job_type="test", status=BackgroundJobStatus.DONE.value, creation_date=now
            )
        )
        session.commit()

        class AsyncSessionFacade:
            async def execute(self, query):
                return session.execute(query)

            async def flush(self):
                session.flush()

        repo = SQLAlchemyBackgroundJobRepository(AsyncSessionFacade())
        await repo.delete_old(days=30)
        remaining = list(session.scalars(select(BackgroundJobModel)))
        assert {row.status for row in remaining if row.id != 20} == {
            BackgroundJobStatus.PENDING.value,
            BackgroundJobStatus.RUNNING.value,
        }
        assert any(row.id == 20 for row in remaining)
    engine.dispose()


def test_effective_values_distinguish_saved_applied_and_secret_presence():
    model = Settings(max_upload_size_mb=50, jwt_secret_key="do-not-expose-this-secret")
    adapter = SettingsProviderConfiguration(model, bus=InProcessEventBus())
    rows = [ConfigParameter("max_upload_size_mb", "60", "int", "ingestion")]
    entries = {row.key: row for row in adapter.effective_parameters(rows)}
    assert entries["max_upload_size_mb"].configured_value == "60"
    assert entries["max_upload_size_mb"].applied_value == "50"
    assert entries["max_upload_size_mb"].source == "database"
    assert entries["jwt_secret_key"].applied_value is None
    assert entries["jwt_secret_key"].present
    assert "do-not-expose" not in repr(entries)
    assert not entries["curator_scope_max_ids"].editable
