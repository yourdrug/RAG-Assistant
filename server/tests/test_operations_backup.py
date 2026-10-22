"""Exercise backup failures with Docker fakes and real archives/checksums."""

import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Protocol

import pytest


ROOT = Path(__file__).resolve().parents[2]
DOCKER_FAKE = r"""
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with Path(os.environ['COMMAND_LOG']).open('a') as log:
    log.write(json.dumps(args) + '\n')
stage = os.environ.get('FAIL_STAGE', '')
if args[0] == 'compose':
    if 'ps' in args:
        if stage == 'ps': sys.exit(1)
        if stage != 'empty': print('postgres\nredis\nqdrant\nminio\nserver\nworker\nclient')
    elif 'pg_dump' in ' '.join(args):
        if stage == 'dump': sys.exit(1)
        print('logical dump fixture')
    elif 'stop' in args and 'postgres' in args and stage == 'stop': sys.exit(1)
    elif 'up' in args and stage == 'resume': sys.exit(1)
    elif 'config' in args: print('postgres:16.4-alpine')
elif 'pg_restore' in args and stage == 'restore': sys.exit(1)
elif 'psql' in args: print('schema-fixture')
"""


class BackupRunner(Protocol):
    def __call__(
        self, script: str = "backup.sh", args: tuple[Path, ...] = (), **updates: str
    ) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]: ...


type BackupStack = tuple[Path, BackupRunner]


@pytest.fixture
def backup_stack(tmp_path: Path) -> BackupStack:
    shutil.copytree(ROOT / "scripts/operations", tmp_path / "scripts/operations")
    (tmp_path / "server").mkdir()
    (tmp_path / "server/.env").touch()
    (tmp_path / "VERSION").write_text("0.1.0")

    for name in ["postgres", "qdrant_storage", "minio", "redis"]:
        directory = tmp_path / "data" / name
        directory.mkdir(parents=True)
        (directory / "fixture").write_text(name)

    binaries = tmp_path / "bin"
    binaries.mkdir()
    docker = binaries / "docker"
    docker.write_text(f"#!{sys.executable}\n" + DOCKER_FAKE)
    docker.chmod(0o755)
    rsync = binaries / "rsync"
    rsync.write_text("#!/bin/sh\nexit 1\n")
    rsync.chmod(0o755)
    log = tmp_path / "commands.jsonl"
    env = {
        **os.environ,
        "DEPLOY_DIR": str(tmp_path),
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "COMMAND_LOG": str(log),
        "BACKUP_DIR": "relative-backups",
        "BACKUP_REMOTE": "",
    }

    def run(
        script: str = "backup.sh", args: tuple[Path, ...] = (), **updates: str
    ) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
        result = subprocess.run(  # noqa: S603 -- repository scripts, isolated data and Docker fake
            ["/bin/bash", str(tmp_path / "scripts/operations" / script), *map(str, args)],
            env={**env, **updates},
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        commands = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        return result, commands

    return tmp_path, run


def test_backup_verifies_relative_destination_and_restores_archive(backup_stack: BackupStack) -> None:
    root, run = backup_stack
    result, commands = run()
    assert result.returncode == 0, result.stderr
    assert (root / ".deploy/last-backup-success").read_text().strip().isdigit()
    backup = next((root / "relative-backups").iterdir())
    assert any("pg_restore" in command for command in commands)
    assert any(command[:2] == ["rm", "-fv"] for command in commands)
    # Services resume before the disposable restoration check starts.
    resume = next(i for i, c in enumerate(commands) if "up" in c)
    restore = next(i for i, c in enumerate(commands) if c[0] == "run")
    assert resume < restore
    target = root / "restored"
    result, _ = run("restore.sh", (backup, target))
    assert result.returncode == 0, result.stderr
    assert (target / "postgres/fixture").read_text() == "postgres"
    result, _ = run()
    assert result.returncode == 0, result.stderr
    assert len(list((root / "relative-backups").iterdir())) == 2


@pytest.mark.parametrize("stage", ["dump", "stop", "resume", "restore", "remote"])
def test_failed_backup_resumes_stack_without_marking_success(backup_stack: BackupStack, stage: str) -> None:
    root, run = backup_stack
    state = root / ".deploy"
    state.mkdir()
    (state / "last-backup-success").write_text("123")
    result, commands = run(FAIL_STAGE=stage, BACKUP_REMOTE="fixture:/backup" if stage == "remote" else "")
    assert result.returncode != 0
    assert (state / "last-backup-success").read_text() == "123"
    assert any("up" in command for command in commands)

    if stage == "restore":
        assert any(command[:2] == ["rm", "-fv"] for command in commands)

    if stage == "resume":
        assert "Stack restart failed" in result.stderr


@pytest.mark.parametrize("stage", ["ps", "empty"])
def test_unavailable_stack_is_not_stopped_or_started(backup_stack: BackupStack, stage: str) -> None:
    root, run = backup_stack
    result, commands = run(FAIL_STAGE=stage)
    assert result.returncode != 0
    assert not any("stop" in c or "up" in c for c in commands)
    assert not (root / ".deploy/last-backup-success").exists()


def test_backup_lock_prevents_stack_mutations(backup_stack: BackupStack) -> None:
    root, run = backup_stack
    (root / ".deploy").mkdir()
    with (root / ".deploy/operations.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result, commands = run()
    assert result.returncode != 0
    assert not commands


def test_corrupted_backup_is_rejected_before_target_creation(backup_stack: BackupStack) -> None:
    root, run = backup_stack
    result, _ = run()
    assert result.returncode == 0, result.stderr
    backup = next((root / "relative-backups").iterdir())
    (backup / "data.tar.gz").write_bytes(b"corrupted")
    target = root / "restored"
    result, _ = run("restore.sh", (backup, target))
    assert result.returncode != 0
    assert not target.exists()
    log_before = (root / "commands.jsonl").read_text()
    result, _ = run("restore-check.sh", (backup,))
    assert result.returncode != 0
    assert (root / "commands.jsonl").read_text() == log_before
