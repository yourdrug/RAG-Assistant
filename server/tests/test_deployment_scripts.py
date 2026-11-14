"""Execute deployment failures against Docker fakes, preserving the live stack."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
DOCKER_FAKE = r"""
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
root = Path(os.environ["DEPLOY_DIR"])
with (root / "commands.jsonl").open("a") as log:
    log.write(json.dumps({"args": args, "image": os.environ.get("SERVER_IMAGE"),
                          "profiles": os.environ.get("COMPOSE_PROFILES", "")}) + "\n")
roles = ["server", "worker", "worker-batch", "maintenance", "client"]
state_file = root / "docker-state.json"
state = json.loads(state_file.read_text()) if state_file.exists() else {
    "old-" + role: {"service": role,
                   "image": "sha256:old-client" if role == "client" else "sha256:old-server"}
    for role in roles
}
if args[0] == "network":
    print("172.18.0.1")
elif args[0] == "image":
    print(args[-1])
elif args[0] == "inspect":
    for name in args[3:]:
        if "PortBindings" in args[2]:
            print("legacy" if name == "old-client" and os.environ.get("LEGACY_FRONTEND") else "")
        else:
            print(state.get(name, {}).get("image", "sha256:old-server"))
elif args[0] == "stop":
    pass
elif args[0] == "rm":
    state.pop(args[-1], None)
elif args[0] == "exec":
    if os.environ.get("FAIL_STAGE") == "readiness":
        sys.exit(1)
elif "config" in args and "--services" in args:
    print("server\nworker")
    if "tei" in os.environ.get("COMPOSE_PROFILES", "").split(","):
        print("tei-embed\ntei-rerank")
elif "config" in args and "--format" in args:
    print(json.dumps({"services": {role: {"environment": {"ML_PROVIDER": "tei"}} for role in roles}}))
elif "ps" in args:
    print("\n".join(name for name, value in state.items() if value["service"] == args[-1]))
elif "pull" in args and os.environ.get("FAIL_STAGE") == "pull":
    sys.exit(1)
elif "exec" in args:
    if "SHOW max_connections" in args[-1]:
        print("300")
    else:
        changed = os.environ.get("SCHEMA_CHANGED") and (root / "migrated").exists()
        print("new-schema" if changed else "old-schema")
elif "run" in args:
    (root / "migrated").touch()
    if os.environ.get("FAIL_STAGE") == "migrate":
        sys.exit(1)
elif "up" in args:
    if ("server" in args and os.environ.get("FAIL_STAGE") == "up"
        and not os.environ.get("SERVER_IMAGE", "").startswith("sha256:")):
        sys.exit(1)
    for role in roles:
        if role not in args:
            continue
        members = [name for name, value in state.items() if value["service"] == role]
        desired = (int(args[args.index("--scale") + 1].split("=")[1])
                   if "--scale" in args else len(members) or 1)
        if "--force-recreate" in args:
            for name in members:
                state.pop(name)
            members = []
        for number in range(len(members), desired):
            name = "new-" + role + "-" + str(len(state))
            image = os.environ.get("CLIENT_IMAGE") if role == "client" else os.environ.get("SERVER_IMAGE")
            state[name] = {"service": role, "image": image}
state_file.write_text(json.dumps(state))
"""


@pytest.fixture
def deployment(tmp_path: Path) -> Any:
    shutil.copytree(ROOT / "scripts/operations", tmp_path / "scripts/operations")
    shutil.copytree(ROOT / "deploy", tmp_path / "deploy")
    (tmp_path / "VERSION").write_text("0.1.0")
    (tmp_path / "server").mkdir()
    (tmp_path / "server/.env").write_text("ML_PROVIDER=tei\n")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    docker = binaries / "docker"
    docker.write_text(f"#!{sys.executable}\n" + DOCKER_FAKE)
    docker.chmod(0o755)
    curl = binaries / "curl"
    curl.write_text('#!/bin/bash\n[[ "${FAIL_STAGE:-}" != smoke ]]\n')
    curl.chmod(0o755)
    env = {
        **os.environ,
        "DEPLOY_DIR": str(tmp_path),
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "COMPOSE_PROFILES": "tei",
        "DEPLOY_DNS_SETTLE_SECONDS": "0",
    }

    def run(**updates: Any) -> Any:
        result = subprocess.run(  # noqa: S603 -- only repository scripts and controlled test fakes
            ["/bin/bash", str(tmp_path / "deploy/deploy.sh"), "v0.6.0", "--gpu"],
            env={**env, **updates},
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        log = tmp_path / "commands.jsonl"
        commands = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        return result, commands

    return tmp_path, run


def test_success_preserves_infrastructure_and_enables_tei(deployment: Any) -> None:
    directory, run = deployment
    result, commands = run(DEPLOY_DIR="")
    assert result.returncode == 0, result.stderr
    assert not any("down" in command["args"] for command in commands)
    assert all(command["profiles"] == "tei" for command in commands)
    assert any("tei-embed" in command["args"] for command in commands)
    assert "production-gpu-0.6.0" in (directory / ".deploy/current.env").read_text()


def test_api_rollout_starts_a_healthy_replacement_before_stopping_old_replica(deployment: Any) -> None:
    _, run = deployment
    result, commands = run()
    assert result.returncode == 0, result.stderr
    assert not any(
        "up" in command["args"] and "server" in command["args"] and "--force-recreate" in command["args"]
        for command in commands
    )
    replacement = next(
        i
        for i, command in enumerate(commands)
        if "--scale" in command["args"] and "server=2" in command["args"]
    )
    stopped = next(
        i
        for i, command in enumerate(commands)
        if command["args"][:1] == ["stop"] and "old-server" in command["args"]
    )
    assert replacement < stopped
    assert any(
        command["args"][:1] == ["exec"] and "http://localhost:8001/ready" in command["args"]
        for command in commands[replacement:stopped]
    )


def test_legacy_frontend_keeps_its_port_until_the_replacement_is_healthy(deployment: Any) -> None:
    _, run = deployment
    result, commands = run(LEGACY_FRONTEND="1")
    assert result.returncode == 0, result.stderr
    stopped = next(
        i
        for i, command in enumerate(commands)
        if command["args"][:1] == ["stop"] and "old-client" in command["args"]
    )
    handoff = next(
        i
        for i, command in enumerate(commands)
        if "--force-recreate" in command["args"] and command["args"][-1] == "gateway"
    )
    assert handoff > stopped
    assert "deploy/gateway-bootstrap.compose.yml" not in commands[handoff]["args"]
    assert any("deploy/gateway-bootstrap.compose.yml" in command["args"] for command in commands[:stopped])


def test_migration_failure_leaves_application_containers_running(deployment: Any) -> None:
    _, run = deployment
    result, commands = run(FAIL_STAGE="migrate")
    assert result.returncode != 0
    assert not any("up" in command["args"] and "server" in command["args"] for command in commands)
    assert not any(command["args"][:1] == ["stop"] for command in commands)


@pytest.mark.parametrize("stage", ["up", "smoke"])
def test_failure_restores_previous_images_if_schema_unchanged(deployment: Any, stage: str) -> None:
    directory, run = deployment
    result, commands = run(FAIL_STAGE=stage)
    assert result.returncode != 0
    assert any(command["image"] == "sha256:old-server" and "up" in command["args"] for command in commands)
    assert not (directory / ".deploy/current.env").exists()


def test_changed_schema_blocks_unsafe_rollback(deployment: Any) -> None:
    _, run = deployment
    result, commands = run(FAIL_STAGE="up", SCHEMA_CHANGED="true")
    assert result.returncode != 0
    assert "automatic image rollback blocked" in result.stderr
    assert not any(
        command["image"] == "sha256:old-server" and "up" in command["args"] for command in commands
    )


def test_pull_failure_leaves_running_stack_alone(deployment: Any) -> None:
    _, run = deployment
    result, commands = run(FAIL_STAGE="pull")
    assert result.returncode != 0
    assert not any("up" in command["args"] or "run" in command["args"] for command in commands)


def test_deployment_lock_blocks_concurrent_mutations(deployment: Any) -> None:
    directory, run = deployment
    (directory / ".deploy").mkdir()
    with (directory / ".deploy/operations.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result, commands = run()
    assert result.returncode != 0
    assert "Another deploy/backup is running" in result.stderr
    assert not any("pull" in command["args"] or "up" in command["args"] for command in commands)


def test_cloud_deployment_does_not_start_tei_from_legacy_provider(deployment: Any) -> None:
    _, run = deployment
    result, commands = run(COMPOSE_PROFILES="", ML_PROVIDER="tei")
    assert result.returncode == 0, result.stderr
    assert not any("tei-embed" in command["args"] for command in commands)
