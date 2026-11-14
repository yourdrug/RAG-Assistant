"""Deployment refuses connection budgets that cannot accommodate rolling replacement."""

import json
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "client_limit,postgres_limit,accepted", [(25, 115, True), (24, 300, False), (1000, 114, False)]
)
def test_capacity_includes_proxy_backends_direct_sessions_and_surge(
    client_limit: int, postgres_limit: int, accepted: bool
) -> None:
    services = {
        name: {"deploy": {"replicas": 1}} for name in ("server", "worker", "worker-batch", "maintenance")
    }
    services["pgbouncer"] = {"environment": {"MAX_CLIENT_CONN": client_limit, "MAX_DB_CONNECTIONS": 60}}
    script = Path(__file__).resolve().parents[2] / "deploy/capacity.py"
    result = subprocess.run(  # noqa: S603 -- repository script, controlled input
        [sys.executable, str(script), str(postgres_limit)],
        input=json.dumps({"services": services}),
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert (result.returncode == 0) == accepted

    if accepted:
        budget = json.loads(result.stdout)
        assert budget["application_clients_with_surge"] == 25
        assert budget["direct_connections_with_surge"] == 25
    else:
        assert "Connection budget rejected" in result.stderr
