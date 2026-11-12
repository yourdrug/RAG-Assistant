"""Migration contracts using a transactional in-memory repository and real files."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import copy
import json
import logging
import shutil
import subprocess
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

from dotenv import dotenv_values
import pytest
from typer.testing import CliRunner

from application.runtime.config import ConfigParameter, ConfigValueType
from application.runtime.setting_keys import TARGET_DYNAMIC_SETTING_KEYS
from composition.cli.commands import config as config_command
from domain.knowledge.enums import DocDomain
from infrastructure import config_migration
from infrastructure.config_migration import LEGACY_POOL_SETTING_KEYS
from infrastructure.database.database import database
from tests.fakes import FakeConfigParameterRepository, FakeUnitOfWork, FakeUnitOfWorkFactory, require_present


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from infrastructure.config_migration import EnvSnapshot


class MigrationRepository(FakeConfigParameterRepository):
    async def delete_by_key_and_domain(self, key: str, domain_key: str | None = None) -> None:
        self._params.pop((key, domain_key), None)


class MigrationFactory(FakeUnitOfWorkFactory):
    def __init__(self) -> None:
        super().__init__()
        self.repository = MigrationRepository()
        self.fail_commit = False

    @asynccontextmanager
    async def create(self, master: bool = False) -> AsyncIterator[FakeUnitOfWork]:
        _ = master
        repository = copy.deepcopy(self.repository)
        uow = FakeUnitOfWork()
        uow.config_parameters = repository
        yield uow

        if self.fail_commit:
            msg = "commit failed"
            raise RuntimeError(msg)

        self.repository = repository


@pytest.fixture
def migration_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, MigrationFactory]:
    for key in (
        TARGET_DYNAMIC_SETTING_KEYS
        | LEGACY_POOL_SETTING_KEYS
        | {"s3_access_key", "s3_secret_key", "file_backend", "ml_provider"}
    ):
        monkeypatch.delenv(key.upper(), raising=False)
        monkeypatch.delenv(key.lower(), raising=False)

    path = tmp_path / ".env"
    path.write_text(
        '# preserved comment\nOPENROUTER_MODEL="legacy model"\nLEGAL_CHUNK_MAX_TOKENS=1600\n'
        "ML_PROVIDER=tei\nRATE_LIMIT_ENABLED=false\nHTTP_POOL_MAX_CONNECTIONS=42\n"
        "HTTP_POOL_MAX_KEEPALIVE=18\nS3_ACCESS_KEY=private-access\nS3_SECRET_KEY=private-secret\n"
    )
    return path, MigrationFactory()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "binding,value",
    [
        (r"S3_SECRET_KEY=ab\cd", r"ab\cd"),
        (r'S3_SECRET_KEY="ab\\cd"', r"ab\cd"),
        (r"S3_SECRET_KEY='ab\cd'", r"ab\cd"),
        (r"S3_SECRET_KEY='ab\\cd'", r"ab\cd"),
        (r"s3_secret_key=ab\cd", r"ab\cd"),
        (r"export S3_SECRET_KEY=ab\cd # retained comment", r"ab\cd"),
    ],
)
async def test_secret_transfer_preserves_original_dotenv_binding(
    migration_env: tuple[Path, MigrationFactory], binding: str, value: str
) -> None:
    path, factory = migration_env
    path.write_text("S3_ACCESS_KEY=private-access\n" + binding + "\n")
    await config_migration.migrate_env_configuration(factory, path, apply=True)
    secret_path = path.with_name(".env.secrets")
    assert dotenv_values(secret_path)["S3_SECRET_KEY"] == value
    await config_migration.migrate_env_configuration(factory, path, apply=True)
    assert dotenv_values(secret_path)["S3_SECRET_KEY"] == value


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "binding,expected",
    [
        (r"S3_SECRET_KEY=ab\cd", r"ab\cd"),
        (r'S3_SECRET_KEY="ab\\cd"', r"ab\cd"),
        (r"S3_SECRET_KEY='ab\cd'", r"ab\cd"),
        (r"S3_SECRET_KEY='ab\\cd'", r"ab\\cd"),
        (r"S3_SECRET_KEY='ab\cd\'ef'", r"ab\cd'ef"),
    ],
)
async def test_secret_transfer_preserves_compose_credentials(
    migration_env: tuple[Path, MigrationFactory], binding: str, expected: str
) -> None:
    path, factory = migration_env
    docker = shutil.which("docker")

    if docker is None:
        pytest.skip("Docker Compose parser is unavailable")

    version = await asyncio.to_thread(
        subprocess.run, [docker, "compose", "version"], capture_output=True, timeout=10
    )

    if version.returncode:
        pytest.skip("Docker Compose parser is unavailable")

    path.write_text("S3_ACCESS_KEY=private-access\n" + binding + "\n")
    await config_migration.migrate_env_configuration(factory, path, apply=True)
    compose_file = path.with_name("compose.yaml")
    compose_file.write_text("services:\n  probe:\n    image: busybox\n    env_file: [.env, .env.secrets]\n")
    # config only invokes the real env parser; no daemon, images or services are used.
    result = await asyncio.to_thread(
        subprocess.run,
        [docker, "compose", "--env-file", "/dev/null", "-f", str(compose_file), "config", "--format", "json"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    assert json.loads(result.stdout)["services"]["probe"]["environment"]["S3_SECRET_KEY"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("exported_key", ["OPENROUTER_MODEL", "openrouter_model"])
async def test_dynamic_environment_override_is_migrated(
    migration_env: tuple[Path, MigrationFactory], monkeypatch: pytest.MonkeyPatch, exported_key: str
) -> None:
    path, factory = migration_env
    monkeypatch.setenv(exported_key, "exported-model")
    await config_migration.migrate_env_configuration(factory, path, apply=True)
    assert require_present(await factory.repository.get_by_key("openrouter_model")).value == "exported-model"
    assert dotenv_values(path)["OPENROUTER_MODEL"] == "legacy model"


@pytest.mark.asyncio
async def test_environment_only_dynamic_override_is_migrated(
    migration_env: tuple[Path, MigrationFactory], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, factory = migration_env
    monkeypatch.setenv("RATE_LIMIT_CHAT_PER_MINUTE", "100")
    await config_migration.migrate_env_configuration(factory, path, apply=True)
    assert require_present(await factory.repository.get_by_key("rate_limit_chat_per_minute")).value == "100"


@pytest.mark.asyncio
async def test_preview_limits_move_to_database_and_are_trimmed(
    migration_env: tuple[Path, MigrationFactory],
) -> None:
    path, factory = migration_env
    path.write_text(path.read_text() + "PREVIEW_MAX_SIZE_MB=7\nPREVIEW_CACHE_TTL_SECONDS=90\n")
    await config_migration.migrate_env_configuration(factory, path, apply=True, trim_env=True)
    assert require_present(await factory.repository.get_by_key("preview_max_size_mb")).value == "7"
    assert require_present(await factory.repository.get_by_key("preview_cache_ttl_seconds")).value == "90"
    assert "PREVIEW_MAX_SIZE_MB" not in dotenv_values(path)
    assert "PREVIEW_CACHE_TTL_SECONDS" not in dotenv_values(path)


@pytest.mark.asyncio
async def test_legacy_pool_environment_overrides_are_migrated(
    migration_env: tuple[Path, MigrationFactory], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, factory = migration_env
    monkeypatch.setenv("HTTP_POOL_MAX_CONNECTIONS", "81")
    monkeypatch.setenv("HTTP_POOL_MAX_KEEPALIVE", "31")
    await config_migration.migrate_env_configuration(factory, path, apply=True)
    rows = {row.key: row.value for row in await factory.repository.get_all()}
    assert rows["embedding_http_max_connections"] == "81"
    assert rows["rerank_http_max_connections"] == "40"
    assert rows["embedding_http_max_keepalive_connections"] == "31"
    assert rows["rerank_http_max_keepalive_connections"] == "15"


@pytest.mark.asyncio
async def test_secret_file_dynamic_value_has_priority_over_env(
    migration_env: tuple[Path, MigrationFactory],
) -> None:
    path, factory = migration_env
    path.with_name(".env.secrets").write_text("OPENROUTER_MODEL=secret-file-model\n")
    await config_migration.migrate_env_configuration(factory, path, apply=True)
    assert (
        require_present(await factory.repository.get_by_key("openrouter_model")).value == "secret-file-model"
    )


@pytest.mark.asyncio
async def test_existing_db_value_survives_environment_override(
    migration_env: tuple[Path, MigrationFactory], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, factory = migration_env
    await add_row(factory, "openrouter_model", "db-model")
    monkeypatch.setenv("OPENROUTER_MODEL", "exported-model")
    await config_migration.migrate_env_configuration(factory, path, apply=True)
    assert require_present(await factory.repository.get_by_key("openrouter_model")).value == "db-model"


async def add_row(factory: MigrationFactory, key: str, value: str, domain_key: str | None = None) -> None:
    await factory.repository.save(
        ConfigParameter(key, value, ConfigValueType.STR, "test", domain_key=domain_key)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("apply", [False, True])
@pytest.mark.parametrize("backend", ["", "FILE_BACKEND=local\n", "FILE_BACKEND=s3\n"])
async def test_s3_requires_credentials_before_mutations(
    migration_env: tuple[Path, MigrationFactory], apply: bool, backend: str
) -> None:
    path, factory = migration_env
    path.write_text(backend + "OPENROUTER_MODEL=legacy\n")
    original = path.read_text()
    with pytest.raises(ValueError, match=r"S3_ACCESS_KEY.*S3_SECRET_KEY"):
        await config_migration.migrate_env_configuration(factory, path, apply=apply)
    assert path.read_text() == original
    assert not path.with_name(".env.secrets").exists()
    assert not list(path.parent.glob("*.backup-*"))
    assert await factory.repository.get_all() == []


@pytest.mark.asyncio
async def test_dry_run_does_not_change_files_or_rows(
    migration_env: tuple[Path, MigrationFactory], caplog: pytest.LogCaptureFixture
) -> None:
    path, factory = migration_env
    before = path.read_bytes()
    caplog.set_level(logging.INFO)
    await config_migration.migrate_env_configuration(factory, path)
    assert path.read_bytes() == before
    assert await factory.repository.get_all() == []
    assert sorted(p.name for p in path.parent.iterdir()) == [".env"]
    assert "OPENROUTER_MODEL" in caplog.text
    assert "private-secret" not in caplog.text
    assert "private-access" not in caplog.text
    assert "legacy model" not in caplog.text


@pytest.mark.asyncio
async def test_apply_preserves_database_values_scopes_and_pending_env(
    migration_env: tuple[Path, MigrationFactory],
) -> None:
    path, factory = migration_env
    await add_row(factory, "openrouter_model", "db model")
    await config_migration.migrate_env_configuration(factory, path, apply=True)
    rows = {(r.key, r.domain_key): r.value for r in await factory.repository.get_all()}
    assert rows[("openrouter_model", None)] == "db model"
    assert rows[("legal_chunk_max_tokens", DocDomain.LEGAL.value)] == "1600"
    assert ("legal_chunk_max_tokens", None) not in rows
    assert rows[("embedding_http_max_connections", None)] == "42"
    assert rows[("rerank_http_max_connections", None)] == "21"
    assert rows[("rerank_http_max_keepalive_connections", None)] == "9"
    assert rows[("rate_limit_enabled", None)] == "false"
    assert not any(key.startswith("s3_") for key, scope in rows)
    env = dotenv_values(path)
    assert env["COMPOSE_PROFILES"] == "tei"
    assert env["OPENROUTER_MODEL"] == "legacy model"
    assert env["HTTP_POOL_MAX_CONNECTIONS"] == "42"
    assert "S3_SECRET_KEY" not in env
    secret_path = path.with_name(".env.secrets")
    assert dotenv_values(secret_path)["S3_SECRET_KEY"] == "private-secret"  # noqa: S105 - Synthetic test data used to verify authentication/redaction.
    assert secret_path.stat().st_mode & 0o777 == 0o600
    backups = list(path.parent.glob("*.backup-*"))
    assert backups
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in backups)
    before = copy.deepcopy(rows)
    await config_migration.migrate_env_configuration(factory, path, apply=True)
    assert {(r.key, r.domain_key): r.value for r in await factory.repository.get_all()} == before


@pytest.mark.asyncio
async def test_agreed_s3_rows_removed_including_scoped_rows(
    migration_env: tuple[Path, MigrationFactory],
) -> None:
    path, factory = migration_env
    await add_row(factory, "s3_secret_key", "private-secret")
    await add_row(factory, "s3_bucket", "rag-documents", DocDomain.LEGAL.value)
    await config_migration.migrate_env_configuration(factory, path, apply=True)
    assert not any(r.key.startswith("s3_") for r in await factory.repository.get_all())


@pytest.mark.asyncio
@pytest.mark.parametrize("key,value", [("s3_secret_key", "other-secret"), ("s3_bucket", "other-bucket")])
async def test_s3_mismatch_blocks_all_mutations(
    migration_env: tuple[Path, MigrationFactory], key: str, value: str, caplog: pytest.LogCaptureFixture
) -> None:
    path, factory = migration_env
    await add_row(factory, key, value)
    before = path.read_bytes()
    with pytest.raises(ValueError, match=key.upper()):
        await config_migration.migrate_env_configuration(factory, path, apply=True)
    assert path.read_bytes() == before
    assert len(await factory.repository.get_all()) == 1
    assert value not in caplog.text


@pytest.mark.asyncio
async def test_missing_s3_credential_blocks_apply(migration_env: tuple[Path, MigrationFactory]) -> None:
    path, factory = migration_env
    path.write_text("FILE_BACKEND=s3\nS3_ACCESS_KEY=private-access\n")
    with pytest.raises(ValueError, match="S3_SECRET_KEY"):
        await config_migration.migrate_env_configuration(factory, path, apply=True)
    assert await factory.repository.get_all() == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "line",
    [
        "CHUNK_SIZE=NaN",
        "OCR_DPI=1",
        "RATE_LIMIT_CHAT_PER_MINUTE=0",
        "HTTP_POOL_MAX_CONNECTIONS=1",
        "HTTP_POOL_MAX_KEEPALIVE=99",
    ],
)
async def test_invalid_values_block_apply_without_echoing_value(
    migration_env: tuple[Path, MigrationFactory], line: str
) -> None:
    path, factory = migration_env
    path.write_text(path.read_text() + line + "\n")
    with pytest.raises(ValueError, match=line.split("=", maxsplit=1)[0]):
        await config_migration.migrate_env_configuration(factory, path, apply=True)
    assert await factory.repository.get_all() == []


@pytest.mark.asyncio
async def test_secret_file_conflict_blocks_apply(migration_env: tuple[Path, MigrationFactory]) -> None:
    path, factory = migration_env
    path.with_name(".env.secrets").write_text("S3_SECRET_KEY=other-secret\n")
    with pytest.raises(ValueError, match="S3_SECRET_KEY"):
        await config_migration.migrate_env_configuration(factory, path, apply=True)
    assert await factory.repository.get_all() == []


@pytest.mark.asyncio
async def test_commit_failure_keeps_env_and_secrets_intact(
    migration_env: tuple[Path, MigrationFactory],
) -> None:
    path, factory = migration_env
    before = path.read_bytes()
    factory.fail_commit = True
    with pytest.raises(RuntimeError, match="commit failed"):
        await config_migration.migrate_env_configuration(factory, path, apply=True)
    assert path.read_bytes() == before
    assert not path.with_name(".env.secrets").exists()
    assert await factory.repository.get_all() == []


@pytest.mark.asyncio
async def test_file_failure_after_commit_is_repeatable(
    migration_env: tuple[Path, MigrationFactory], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, factory = migration_env
    before = path.read_bytes()
    replace = config_migration.atomic_replace
    monkeypatch.setattr(
        config_migration, "atomic_replace", lambda *_args: (_ for _ in ()).throw(OSError("write failed"))
    )
    with pytest.raises(OSError, match="write failed"):
        await config_migration.migrate_env_configuration(factory, path, apply=True)
    assert path.read_bytes() == before
    assert await factory.repository.get_all()
    monkeypatch.setattr(config_migration, "atomic_replace", replace)
    await config_migration.migrate_env_configuration(factory, path, apply=True)
    assert "S3_SECRET_KEY" not in dotenv_values(path)


@pytest.mark.asyncio
async def test_existing_profiles_preserved(migration_env: tuple[Path, MigrationFactory]) -> None:
    path, factory = migration_env
    path.write_text(path.read_text() + "COMPOSE_PROFILES=monitoring\n")
    await config_migration.migrate_env_configuration(factory, path, apply=True)
    assert dotenv_values(path)["COMPOSE_PROFILES"] == "monitoring"


@pytest.mark.asyncio
async def test_exported_secret_conflict_is_not_silently_selected(
    migration_env: tuple[Path, MigrationFactory], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, factory = migration_env
    monkeypatch.setenv("S3_SECRET_KEY", "different-exported-secret")
    with pytest.raises(ValueError, match="S3_SECRET_KEY"):
        await config_migration.migrate_env_configuration(factory, path, apply=True)
    assert await factory.repository.get_all() == []


@pytest.mark.asyncio
async def test_existing_secret_file_permissions_are_hardened(
    migration_env: tuple[Path, MigrationFactory],
) -> None:
    path, factory = migration_env
    secrets = path.with_name(".env.secrets")
    secrets.write_text("S3_ACCESS_KEY=private-access\nS3_SECRET_KEY=private-secret\n")
    secrets.chmod(0o644)
    path.write_text("ML_PROVIDER=tei\n")
    await config_migration.migrate_env_configuration(factory, path, apply=True)
    assert secrets.stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_secret_installed_before_env_write_failure(
    migration_env: tuple[Path, MigrationFactory], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, factory = migration_env
    replace = config_migration.atomic_replace

    def fail_main(snapshot: EnvSnapshot, text: str) -> None:
        if snapshot.path == path:
            msg = "main env write failed"
            raise OSError(msg)

        replace(snapshot, text)

    monkeypatch.setattr(config_migration, "atomic_replace", fail_main)
    with pytest.raises(OSError, match="main env write failed"):
        await config_migration.migrate_env_configuration(factory, path, apply=True)
    assert dotenv_values(path)["S3_SECRET_KEY"] == "private-secret"  # noqa: S105 - Synthetic test data used to verify authentication/redaction.
    assert dotenv_values(path.with_name(".env.secrets"))["S3_SECRET_KEY"] == "private-secret"  # noqa: S105 - Synthetic test data used to verify authentication/redaction.
    monkeypatch.setattr(config_migration, "atomic_replace", replace)
    await config_migration.migrate_env_configuration(factory, path, apply=True)
    assert "S3_SECRET_KEY" not in dotenv_values(path)


@pytest.mark.asyncio
async def test_environment_edit_after_db_commit_is_preserved(
    migration_env: tuple[Path, MigrationFactory], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, factory = migration_env
    replace = config_migration.atomic_replace

    def edit_then_replace(snapshot: EnvSnapshot, text: str) -> None:
        if snapshot.path == path:
            path.write_text("# operator edit\n" + path.read_text())

        replace(snapshot, text)

    monkeypatch.setattr(config_migration, "atomic_replace", edit_then_replace)
    with pytest.raises(ValueError, match="changed"):
        await config_migration.migrate_env_configuration(factory, path, apply=True)
    assert path.read_text().startswith("# operator edit")


@pytest.mark.asyncio
async def test_missing_provider_credential_blocks_apply(
    migration_env: tuple[Path, MigrationFactory], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, factory = migration_env
    monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
    path.write_text("ML_PROVIDER=deepinfra\n")
    with pytest.raises(ValueError, match="DEEPINFRA_API_KEY"):
        await config_migration.migrate_env_configuration(factory, path, apply=True)
    assert await factory.repository.get_all() == []


@pytest.mark.asyncio
async def test_migration_uses_env_provider_despite_legacy_database_value(
    migration_env: tuple[Path, MigrationFactory], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, factory = migration_env
    monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
    legacy = ConfigParameter("ml_provider", "deepinfra", ConfigValueType.STR, "ml")
    await factory.repository.save(legacy)
    await config_migration.migrate_env_configuration(factory, path, apply=True, trim_env=True)
    assert dotenv_values(path)["ML_PROVIDER"] == "tei"
    assert require_present(await factory.repository.get_by_key("ml_provider")).value == "deepinfra"


@pytest.mark.parametrize("apply", [False, True])
def test_cli_defaults_to_dry_run_and_accepts_apply(
    migration_env: tuple[Path, MigrationFactory], monkeypatch: pytest.MonkeyPatch, apply: bool
) -> None:
    path, _factory = migration_env
    run = AsyncMock()
    monkeypatch.setattr(config_command, "run_env_migration", run)
    args = ["migrate-env", "--env-file", str(path)] + (["--apply"] if apply else [])
    result = CliRunner().invoke(config_command.config_app, args)
    assert result.exit_code == 0, result.output
    run.assert_awaited_once_with(path, apply=apply, trim_env=False)


def test_cli_failure_does_not_echo_database_exception(
    migration_env: tuple[Path, MigrationFactory],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path, _factory = migration_env
    monkeypatch.setattr(
        config_command, "run_env_migration", AsyncMock(side_effect=RuntimeError("private-secret"))
    )
    result = CliRunner().invoke(config_command.config_app, ["migrate-env", "--env-file", str(path)])
    assert result.exit_code == 1
    assert "private-secret" not in result.output + caplog.text


@pytest.mark.asyncio
async def test_cli_disconnects_after_migration_failure(
    migration_env: tuple[Path, MigrationFactory], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _factory = migration_env
    monkeypatch.setattr(database, "connect", AsyncMock())
    disconnect = AsyncMock()
    monkeypatch.setattr(database, "disconnect", disconnect)
    monkeypatch.setattr(
        config_command, "migrate_env_configuration", AsyncMock(side_effect=ValueError("blocked"))
    )
    with pytest.raises(ValueError, match="blocked"):
        await config_command.run_env_migration(path, apply=False)
    disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_explicit_new_pool_names_are_not_lost(migration_env: tuple[Path, MigrationFactory]) -> None:
    path, factory = migration_env
    path.write_text(
        "S3_ACCESS_KEY=private-access\nS3_SECRET_KEY=private-secret\nEMBEDDING_HTTP_MAX_CONNECTIONS=50\n"
        "EMBEDDING_HTTP_MAX_KEEPALIVE_CONNECTIONS=25\n"
    )
    await config_migration.migrate_env_configuration(factory, path, apply=True)
    rows = {row.key: row.value for row in await factory.repository.get_all()}
    assert rows["embedding_http_max_connections"] == "50"
    assert rows["embedding_http_max_keepalive_connections"] == "25"


@pytest.mark.asyncio
async def test_new_and_old_pool_conflict_blocks_migration(
    migration_env: tuple[Path, MigrationFactory],
) -> None:
    path, factory = migration_env
    path.write_text(path.read_text() + "EMBEDDING_HTTP_MAX_CONNECTIONS=50\n")
    with pytest.raises(ValueError, match="EMBEDDING_HTTP_MAX_CONNECTIONS"):
        await config_migration.migrate_env_configuration(factory, path, apply=True)
    assert await factory.repository.get_all() == []


@pytest.mark.asyncio
async def test_empty_optional_provider_key_is_preserved(
    migration_env: tuple[Path, MigrationFactory], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, factory = migration_env
    monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
    path.write_text(path.read_text() + "DEEPINFRA_API_KEY=''\n")
    await config_migration.migrate_env_configuration(factory, path, apply=True)
    assert dotenv_values(path.with_name(".env.secrets"))["DEEPINFRA_API_KEY"] == ""


@pytest.mark.asyncio
async def test_lowercase_secret_conflict_is_not_missed(migration_env: tuple[Path, MigrationFactory]) -> None:
    path, factory = migration_env
    path.write_text("s3_secret_key=lowercase-secret\n")
    path.with_name(".env.secrets").write_text("S3_SECRET_KEY=other-secret\n")
    with pytest.raises(ValueError, match="S3_SECRET_KEY"):
        await config_migration.migrate_env_configuration(factory, path, apply=True)
    assert await factory.repository.get_all() == []


@pytest.mark.asyncio
async def test_lowercase_settings_and_secrets_follow_settings_case_rules(
    migration_env: tuple[Path, MigrationFactory],
) -> None:
    path, factory = migration_env
    path.write_text(
        "S3_ACCESS_KEY=private-access\nopenrouter_model=legacy-model\ns3_secret_key=lowercase-secret\n"
    )
    await config_migration.migrate_env_configuration(factory, path, apply=True)
    assert require_present(await factory.repository.get_by_key("openrouter_model")).value == "legacy-model"
    assert "s3_secret_key" not in dotenv_values(path)
    assert dotenv_values(path.with_name(".env.secrets"))["S3_SECRET_KEY"] == "lowercase-secret"  # noqa: S105 - Synthetic test data used to verify authentication/redaction.


@pytest.mark.asyncio
@pytest.mark.parametrize("apply", [False, True])
async def test_final_trim_preserves_static_overrides_and_db_values(
    migration_env: tuple[Path, MigrationFactory], apply: bool
) -> None:
    path, factory = migration_env
    path.write_text(
        path.read_text() + "DB_POOL_SIZE=73\nREDIS_USER=custom\n"
        "FORWARDED_ALLOW_IPS=10.2.0.1\nEMBED_MODEL=obsolete\n"
        "RERANK_MODEL=obsolete\nRERANK_DEVICE=cuda\nCOMPOSE_PROFILES=\n"
    )
    original = path.read_text()
    await factory.repository.save(
        ConfigParameter(
            key="openrouter_model",
            value="existing-db-model",
            value_type=ConfigValueType.STR,
            category="llm",
        )
    )
    await config_migration.migrate_env_configuration(factory, path, apply=apply, trim_env=True)

    if not apply:
        assert path.read_text() == original
        assert not list(path.parent.glob("*.backup-*"))
        return

    env = dotenv_values(path)
    assert env["DB_POOL_SIZE"] == "73"
    assert env["REDIS_USER"] == "custom"
    assert env["FORWARDED_ALLOW_IPS"] == "10.2.0.1"
    assert env["COMPOSE_PROFILES"] == ""
    assert env["ML_PROVIDER"] == "tei"
    assert (
        not {
            "OPENROUTER_MODEL",
            "LEGAL_CHUNK_MAX_TOKENS",
            "RATE_LIMIT_ENABLED",
            "HTTP_POOL_MAX_CONNECTIONS",
            "HTTP_POOL_MAX_KEEPALIVE",
            "EMBED_MODEL",
            "RERANK_MODEL",
            "RERANK_DEVICE",
        }
        & env.keys()
    )
    assert (
        require_present(await factory.repository.get_by_key("openrouter_model")).value == "existing-db-model"
    )
    await config_migration.migrate_env_configuration(factory, path, apply=True, trim_env=True)
    assert (
        require_present(await factory.repository.get_by_key("openrouter_model")).value == "existing-db-model"
    )


@pytest.mark.asyncio
async def test_final_trim_waits_for_commit(migration_env: tuple[Path, MigrationFactory]) -> None:
    path, factory = migration_env
    original = path.read_text()
    factory.fail_commit = True
    with pytest.raises(RuntimeError, match="commit failed"):
        await config_migration.migrate_env_configuration(factory, path, apply=True, trim_env=True)
    assert path.read_text() == original
    assert not path.with_name(".env.secrets").exists()
    assert await factory.repository.get_all() == []


def test_cli_accepts_final_trim_without_applying(
    migration_env: tuple[Path, MigrationFactory], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = migration_env
    run = AsyncMock()
    monkeypatch.setattr(config_command, "run_env_migration", run)
    result = CliRunner().invoke(
        config_command.config_app, ["migrate-env", "--env-file", str(path), "--trim-env"]
    )
    assert result.exit_code == 0, result.output
    run.assert_awaited_once_with(path, apply=False, trim_env=True)
