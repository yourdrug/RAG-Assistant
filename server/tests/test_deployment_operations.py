"""Operational checks must report failures; restore must never overwrite live data."""

import json
import os
import runpy
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("healthy", [True, False])
def test_monitor_reports_readiness_and_uses_optional_webhook(tmp_path, monkeypatch, healthy):
    operations = tmp_path / "scripts/operations"
    operations.mkdir(parents=True)
    source = operations / "check.py"
    shutil.copy(ROOT / "scripts/operations/check.py", source)
    (tmp_path / "data").mkdir()
    (tmp_path / ".deploy").mkdir()
    (tmp_path / ".deploy/last-backup-success").write_text("1000")
    monkeypatch.setattr("time.time", lambda: 1001)
    monkeypatch.setattr("shutil.disk_usage", lambda _: SimpleNamespace(free=80, total=100))
    for name in ("OPERATIONS_READY_URL", "OPERATIONS_BACKUP_MAX_AGE_SECONDS", "OPERATIONS_MIN_FREE_RATIO"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPERATIONS_ALERT_WEBHOOK", "https://example.test/alerts")
    response = MagicMock()
    response.__enter__.return_value.read.return_value = json.dumps(
        {"status": "healthy" if healthy else "degraded"}
    ).encode()
    calls = []

    def urlopen(request, **kwargs):
        calls.append(request)
        return response

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    if healthy:
        runpy.run_path(str(source), run_name="__main__")
        assert len(calls) == 1
    else:
        with pytest.raises(SystemExit) as error:
            runpy.run_path(str(source), run_name="__main__")
        assert error.value.code == 1
        assert len(calls) == 2
        assert "degraded" in json.loads(calls[1].data)["text"]


def test_restore_refuses_existing_target_before_reading_archive(tmp_path):
    backup = tmp_path / "backup"
    backup.mkdir()
    target = tmp_path / "live-data"
    target.mkdir()
    sentinel = target / "keep"
    sentinel.write_text("production data")
    result = subprocess.run(  # noqa: S603 -- repository script, controlled temporary paths
        ["/bin/bash", str(ROOT / "scripts/operations/restore.sh"), str(backup), str(target)],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode != 0
    assert "refusing to overwrite" in result.stderr
    assert sentinel.read_text() == "production data"
