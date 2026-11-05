"""Validate deployment templates in a disposable directory without touching live services."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

from dotenv import dotenv_values
import pytest

from application.runtime.setting_keys import SECRET_SETTING_KEYS, TARGET_DYNAMIC_SETTING_KEYS


ROOT = Path(__file__).resolve().parents[2]


def test_env_template_contains_only_bootstrap_and_static_settings() -> None:
    values = dotenv_values(ROOT / "server/.env.example")
    assert not {key.lower() for key in values} & (SECRET_SETTING_KEYS | TARGET_DYNAMIC_SETTING_KEYS)
    assert {"ALLOWED_ORIGINS", "REDIS_USER", "COMPOSE_PROFILES", "S3_ENDPOINT"} <= values.keys()
    assert set(dotenv_values(ROOT / "server/.env.secrets.example")) == {
        key.upper() for key in SECRET_SETTING_KEYS
    }


@pytest.fixture
def deployment_templates(tmp_path: Path) -> Any:
    for directory in ("server", "client"):
        (tmp_path / directory).mkdir()
        shutil.copy(ROOT / directory / ".env.example", tmp_path / directory / ".env.example")

    shutil.copy(ROOT / "server/.env.secrets.example", tmp_path / "server/.env.secrets.example")
    shutil.copy(ROOT / "Taskfile.yml", tmp_path)
    shutil.copy(ROOT / "VERSION", tmp_path)
    return tmp_path


def run_tool(binary: Any, args: Any, directory: Any, env: Any = None) -> subprocess.CompletedProcess[str]:
    executable = shutil.which(binary)

    if executable is None:
        pytest.skip(f"{binary} is unavailable")

    return subprocess.run(  # noqa: S603 -- known tools, disposable configuration only
        [executable, *args], cwd=directory, env=env, capture_output=True, text=True, timeout=20, check=False
    )


def test_task_env_creates_private_secrets_and_preserves_existing_files(deployment_templates: Any) -> None:
    directory = deployment_templates
    secret = directory / "server/.env.secrets"

    for _ in range(2):
        result = run_tool("task", ["env"], directory)
        assert result.returncode == 0, result.stderr
        assert secret.stat().st_mode & 0o777 == 0o600

        if _ == 0:
            secret.write_text("JWT_SECRET_KEY=keep-existing\n")
            secret.chmod(0o644)
            (directory / "server/.env").write_text("DB_POOL_SIZE=73\n")

    assert secret.read_text() == "JWT_SECRET_KEY=keep-existing\n"
    assert (directory / "server/.env").read_text() == "DB_POOL_SIZE=73\n"
    result = run_tool("task", ["pull-model"], directory)
    assert result.returncode != 0
    assert 'Task "pull-model" does not exist' in result.stderr


@pytest.mark.parametrize("profiles", [None, "", "tei"])
@pytest.mark.parametrize("gpu", [False, True])
def test_compose_config_with_disposable_credentials(
    deployment_templates: Any, profiles: str | None, gpu: bool
) -> None:
    directory = deployment_templates

    for name in ("docker-compose.yml", "docker-compose.override.yml", "docker-compose.gpu.yml"):
        shutil.copy(ROOT / name, directory)

    shutil.copy(directory / "server/.env.example", directory / "server/.env")
    (directory / "server/.env.secrets").write_text(
        "\n".join(f"{key.upper()}=disposable-test-credential" for key in SECRET_SETTING_KEYS) + "\n"
    )
    env = {
        "PATH": os.environ["PATH"],
        "HOME": os.environ["HOME"],
        "VERSION": "0.0.0",
    }

    if profiles is not None:
        env["COMPOSE_PROFILES"] = profiles

    args = [
        "compose",
        "--env-file",
        "server/.env",
        "--env-file",
        "server/.env.secrets",
        "-f",
        "docker-compose.yml",
        "-f",
        "docker-compose.override.yml",
    ]

    if gpu:
        args += ["-f", "docker-compose.gpu.yml"]

    result = run_tool("docker", [*args, "config", "--quiet"], directory, env)
    assert result.returncode == 0, result.stderr
    result = run_tool("docker", [*args, "config", "--services"], directory, env)
    assert result.returncode == 0, result.stderr
    assert "ollama" not in result.stdout.splitlines()
    assert ("tei-embed" in result.stdout.splitlines()) == (profiles == "tei")
