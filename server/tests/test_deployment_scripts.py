"""Execute deployment failures against Docker fakes, preserving the live stack."""

import fcntl
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCKER_FAKE = r'''
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
root = Path(os.environ["DEPLOY_DIR"])
with (root / "commands.jsonl").open("a") as log:
    log.write(json.dumps({"args": args, "image": os.environ.get("SERVER_IMAGE")}) + "\n")
if args[0] == "network":
    print("172.18.0.1")
elif args[0] == "inspect":
    print("sha256:old-client" if args[-1] == "old-client" else "sha256:old-server")
elif "config" in args and "--format" in args:
    print(json.dumps({"services": {"server": {"environment": {"ML_PROVIDER": "tei"}}}}))
elif "ps" in args:
    print("old-" + args[-1])
elif "pull" in args and os.environ.get("FAIL_STAGE") == "pull":
    sys.exit(1)
elif "exec" in args:
    print("new-schema" if os.environ.get("SCHEMA_CHANGED") and (root / "migrated").exists() else "old-schema")
elif "run" in args:
    (root / "migrated").touch()
    if os.environ.get("FAIL_STAGE") == "migrate":
        sys.exit(1)
elif (
    "up" in args and "server" in args and os.environ.get("FAIL_STAGE") == "up"
    and not os.environ.get("SERVER_IMAGE", "").startswith("sha256:")
):
    sys.exit(1)
'''


@pytest.fixture
def deployment(tmp_path):
    shutil.copy(ROOT / "deploy.sh", tmp_path)
    shutil.copytree(ROOT / "scripts/operations", tmp_path / "scripts/operations")
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
    env = {**os.environ, "DEPLOY_DIR": str(tmp_path), "PATH": f"{binaries}:{os.environ['PATH']}"}

    def run(**updates):
        result = subprocess.run(  # noqa: S603 -- only repository scripts and controlled test fakes
            ["/bin/bash", str(tmp_path / "deploy.sh"), "v0.6.0", "--gpu"],
            env={**env, **updates},
            capture_output=True,
            text=True,
            timeout=15,
        )
        commands = [json.loads(line) for line in (tmp_path / "commands.jsonl").read_text().splitlines()]
        return result, commands

    return tmp_path, run


def test_success_preserves_infrastructure_and_enables_tei(deployment):
    directory, run = deployment
    result, commands = run()
    assert result.returncode == 0, result.stderr
    assert not any("down" in command["args"] for command in commands)
    assert all("--profile" in command["args"] for command in commands if "up" in command["args"])
    assert "production-gpu-0.6.0" in (directory / ".deploy/current.env").read_text()


@pytest.mark.parametrize("stage", ["up", "smoke", "migrate"])
def test_failure_restores_previous_images_if_schema_unchanged(deployment, stage):
    directory, run = deployment
    result, commands = run(FAIL_STAGE=stage)
    assert result.returncode != 0
    assert any(command["image"] == "sha256:old-server" and "up" in command["args"] for command in commands)
    assert not (directory / ".deploy/current.env").exists()


def test_changed_schema_blocks_unsafe_rollback(deployment):
    _, run = deployment
    result, commands = run(FAIL_STAGE="up", SCHEMA_CHANGED="true")
    assert result.returncode != 0
    assert "automatic image rollback blocked" in result.stderr
    assert not any(
        command["image"] == "sha256:old-server" and "up" in command["args"] for command in commands
    )


def test_pull_failure_leaves_running_stack_alone(deployment):
    _, run = deployment
    result, commands = run(FAIL_STAGE="pull")
    assert result.returncode != 0
    assert not any("up" in command["args"] or "run" in command["args"] for command in commands)


def test_deployment_lock_blocks_concurrent_mutations(deployment):
    directory, run = deployment
    (directory / ".deploy").mkdir()
    with (directory / ".deploy/operations.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result, commands = run()
    assert result.returncode != 0
    assert "Another deploy/backup is running" in result.stderr
    assert not any("pull" in command["args"] or "up" in command["args"] for command in commands)
