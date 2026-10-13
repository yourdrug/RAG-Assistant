"""Migration contracts using a transactional in-memory repository and real files."""

import copy
import logging
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from dotenv import dotenv_values
from typer.testing import CliRunner

from domain.entities.config_parameter import ConfigParameter
from domain.value_objects.config_value_type import ConfigValueType
from domain.value_objects.doc_domain import DocDomain
from fakes import FakeConfigParameterRepository
from infrastructure import config_migration
from presentation.cli.commands import config as config_command


class MigrationRepository(FakeConfigParameterRepository):
    async def delete_by_key_and_domain(self, key, domain_key=None):
        self._params.pop((key, domain_key), None)


class MigrationFactory:
    def __init__(self):
        self.repository = MigrationRepository()
        self.fail_commit = False

    @asynccontextmanager
    async def create(self, master=False):
        repository = copy.deepcopy(self.repository)
        yield SimpleNamespace(config_parameters=repository)
        if self.fail_commit:
            raise RuntimeError("commit failed")
        self.repository = repository


@pytest.fixture
def migration_env(tmp_path, monkeypatch):
    for key in ("ML_PROVIDER", "S3_ACCESS_KEY", "S3_SECRET_KEY", "FILE_BACKEND", "RATE_LIMIT_ENABLED"):
        monkeypatch.delenv(key, raising=False)
    path = tmp_path / ".env"
    path.write_text(
        '# preserved comment\nLLM_MODEL="legacy model"\nLEGAL_CHUNK_SIZE=1600\n'
        "ML_PROVIDER=tei\nRATE_LIMIT_ENABLED=false\nHTTP_POOL_MAX_CONNECTIONS=42\n"
        "HTTP_POOL_MAX_KEEPALIVE=18\nS3_ACCESS_KEY=private-access\nS3_SECRET_KEY=private-secret\n"
    )
    return path, MigrationFactory()


async def add_row(factory, key, value, domain_key=None):
    await factory.repository.save(
        ConfigParameter(key, value, ConfigValueType.STR, "test", domain_key=domain_key)
    )


@pytest.mark.asyncio
async def test_dry_run_does_not_change_files_or_rows(migration_env, caplog):
    path, factory = migration_env
    before = path.read_bytes()
    caplog.set_level(logging.INFO)
    await config_migration.migrate_env_configuration(factory, path)
    assert path.read_bytes() == before
    assert await factory.repository.get_all() == []
    assert sorted(p.name for p in path.parent.iterdir()) == [".env"]
    assert "LLM_MODEL" in caplog.text
    assert "private-secret" not in caplog.text
    assert "private-access" not in caplog.text
    assert "legacy model" not in caplog.text


@pytest.mark.asyncio
async def test_apply_preserves_database_values_scopes_and_pending_env(migration_env):
    path, factory = migration_env
    await add_row(factory, "llm_model", "db model")
    await config_migration.migrate_env_configuration(factory, path, apply=True)
    rows = {(r.key, r.domain_key): r.value for r in await factory.repository.get_all()}
    assert rows[("llm_model", None)] == "db model"
    assert rows[("legal_chunk_size", DocDomain.LEGAL.value)] == "1600"
    assert ("legal_chunk_size", None) not in rows
    assert rows[("embedding_http_max_connections", None)] == "42"
    assert rows[("rerank_http_max_connections", None)] == "21"
    assert rows[("rerank_http_max_keepalive_connections", None)] == "9"
    assert rows[("rate_limit_enabled", None)] == "false"
    assert not any(key.startswith("s3_") for key, scope in rows)
    env = dotenv_values(path)
    assert env["COMPOSE_PROFILES"] == "tei"
    assert env["LLM_MODEL"] == "legacy model"
    assert env["HTTP_POOL_MAX_CONNECTIONS"] == "42"
    assert "S3_SECRET_KEY" not in env
    secret_path = path.with_name(".env.secrets")
    assert dotenv_values(secret_path)["S3_SECRET_KEY"] == "private-secret"
    assert secret_path.stat().st_mode & 0o777 == 0o600
    backups = list(path.parent.glob("*.backup-*"))
    assert backups
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in backups)
    before = copy.deepcopy(rows)
    await config_migration.migrate_env_configuration(factory, path, apply=True)
    assert {(r.key, r.domain_key): r.value for r in await factory.repository.get_all()} == before


@pytest.mark.asyncio
async def test_agreed_s3_rows_removed_including_scoped_rows(migration_env):
    path, factory = migration_env
    await add_row(factory, "s3_secret_key", "private-secret")
    await add_row(factory, "s3_bucket", "rag-documents", DocDomain.LEGAL.value)
    await config_migration.migrate_env_configuration(factory, path, apply=True)
    assert not any(r.key.startswith("s3_") for r in await factory.repository.get_all())


@pytest.mark.asyncio
@pytest.mark.parametrize("key,value", [("s3_secret_key", "other-secret"), ("s3_bucket", "other-bucket")])
async def test_s3_mismatch_blocks_all_mutations(migration_env, key, value, caplog):
    path, factory = migration_env
    await add_row(factory, key, value)
    before = path.read_bytes()
    with pytest.raises(ValueError, match=key.upper()):
        await config_migration.migrate_env_configuration(factory, path, apply=True)
    assert path.read_bytes() == before
    assert len(await factory.repository.get_all()) == 1
    assert value not in caplog.text


@pytest.mark.asyncio
async def test_missing_s3_credential_blocks_apply(migration_env):
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
async def test_invalid_values_block_apply_without_echoing_value(migration_env, line):
    path, factory = migration_env
    path.write_text(path.read_text() + line + "\n")
    with pytest.raises(ValueError, match=line.split("=")[0]):
        await config_migration.migrate_env_configuration(factory, path, apply=True)
    assert await factory.repository.get_all() == []


@pytest.mark.asyncio
async def test_secret_file_conflict_blocks_apply(migration_env):
    path, factory = migration_env
    path.with_name(".env.secrets").write_text("S3_SECRET_KEY=other-secret\n")
    with pytest.raises(ValueError, match="S3_SECRET_KEY"):
        await config_migration.migrate_env_configuration(factory, path, apply=True)
    assert await factory.repository.get_all() == []


@pytest.mark.asyncio
async def test_commit_failure_keeps_env_and_secrets_intact(migration_env):
    path, factory = migration_env
    before = path.read_bytes()
    factory.fail_commit = True
    with pytest.raises(RuntimeError, match="commit failed"):
        await config_migration.migrate_env_configuration(factory, path, apply=True)
    assert path.read_bytes() == before
    assert not path.with_name(".env.secrets").exists()
    assert await factory.repository.get_all() == []


@pytest.mark.asyncio
async def test_file_failure_after_commit_is_repeatable(migration_env, monkeypatch):
    path, factory = migration_env
    before = path.read_bytes()
    replace = config_migration.atomic_replace
    monkeypatch.setattr(
        config_migration, "atomic_replace", lambda *args: (_ for _ in ()).throw(OSError("write failed"))
    )
    with pytest.raises(OSError, match="write failed"):
        await config_migration.migrate_env_configuration(factory, path, apply=True)
    assert path.read_bytes() == before
    assert await factory.repository.get_all()
    monkeypatch.setattr(config_migration, "atomic_replace", replace)
    await config_migration.migrate_env_configuration(factory, path, apply=True)
    assert "S3_SECRET_KEY" not in dotenv_values(path)


@pytest.mark.asyncio
async def test_existing_profiles_preserved(migration_env):
    path, factory = migration_env
    path.write_text(path.read_text() + "COMPOSE_PROFILES=monitoring\n")
    await config_migration.migrate_env_configuration(factory, path, apply=True)
    assert dotenv_values(path)["COMPOSE_PROFILES"] == "monitoring,tei"


@pytest.mark.asyncio
async def test_exported_secret_conflict_is_not_silently_selected(migration_env, monkeypatch):
    path, factory = migration_env
    monkeypatch.setenv("S3_SECRET_KEY", "different-exported-secret")
    with pytest.raises(ValueError, match="S3_SECRET_KEY"):
        await config_migration.migrate_env_configuration(factory, path, apply=True)
    assert await factory.repository.get_all() == []


@pytest.mark.asyncio
async def test_existing_secret_file_permissions_are_hardened(migration_env):
    path, factory = migration_env
    secrets = path.with_name(".env.secrets")
    secrets.write_text("S3_ACCESS_KEY=private-access\nS3_SECRET_KEY=private-secret\n")
    secrets.chmod(0o644)
    path.write_text("ML_PROVIDER=tei\n")
    await config_migration.migrate_env_configuration(factory, path, apply=True)
    assert secrets.stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_secret_installed_before_env_write_failure(migration_env, monkeypatch):
    path, factory = migration_env
    replace = config_migration.atomic_replace

    def fail_main(snapshot, text):
        if snapshot.path == path:
            raise OSError("main env write failed")
        replace(snapshot, text)

    monkeypatch.setattr(config_migration, "atomic_replace", fail_main)
    with pytest.raises(OSError):
        await config_migration.migrate_env_configuration(factory, path, apply=True)
    assert dotenv_values(path)["S3_SECRET_KEY"] == "private-secret"
    assert dotenv_values(path.with_name(".env.secrets"))["S3_SECRET_KEY"] == "private-secret"
    monkeypatch.setattr(config_migration, "atomic_replace", replace)
    await config_migration.migrate_env_configuration(factory, path, apply=True)
    assert "S3_SECRET_KEY" not in dotenv_values(path)


@pytest.mark.asyncio
async def test_environment_edit_after_db_commit_is_preserved(migration_env, monkeypatch):
    path, factory = migration_env
    replace = config_migration.atomic_replace

    def edit_then_replace(snapshot, text):
        if snapshot.path == path:
            path.write_text("# operator edit\n" + path.read_text())
        replace(snapshot, text)

    monkeypatch.setattr(config_migration, "atomic_replace", edit_then_replace)
    with pytest.raises(ValueError, match="changed"):
        await config_migration.migrate_env_configuration(factory, path, apply=True)
    assert path.read_text().startswith("# operator edit")


@pytest.mark.asyncio
async def test_missing_provider_credential_blocks_apply(migration_env, monkeypatch):
    path, factory = migration_env
    monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
    path.write_text("ML_PROVIDER=deepinfra\n")
    with pytest.raises(ValueError, match="DEEPINFRA_API_KEY"):
        await config_migration.migrate_env_configuration(factory, path, apply=True)
    assert await factory.repository.get_all() == []


@pytest.mark.parametrize("apply", [False, True])
def test_cli_defaults_to_dry_run_and_accepts_apply(migration_env, monkeypatch, apply):
    path, factory = migration_env
    run = AsyncMock()
    monkeypatch.setattr(config_command, "run_env_migration", run)
    args = ["migrate-env", "--env-file", str(path)] + (["--apply"] if apply else [])
    result = CliRunner().invoke(config_command.config_app, args)
    assert result.exit_code == 0, result.output
    run.assert_awaited_once_with(path, apply)


def test_cli_failure_does_not_echo_database_exception(migration_env, monkeypatch, caplog):
    path, factory = migration_env
    monkeypatch.setattr(
        config_command, "run_env_migration", AsyncMock(side_effect=RuntimeError("private-secret"))
    )
    result = CliRunner().invoke(config_command.config_app, ["migrate-env", "--env-file", str(path)])
    assert result.exit_code == 1
    assert "private-secret" not in result.output + caplog.text


@pytest.mark.asyncio
async def test_cli_disconnects_after_migration_failure(migration_env, monkeypatch):
    path, factory = migration_env
    monkeypatch.setattr(config_command.database, "connect", AsyncMock())
    disconnect = AsyncMock()
    monkeypatch.setattr(config_command.database, "disconnect", disconnect)
    monkeypatch.setattr(
        config_command, "migrate_env_configuration", AsyncMock(side_effect=ValueError("blocked"))
    )
    with pytest.raises(ValueError, match="blocked"):
        await config_command.run_env_migration(path, False)
    disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_explicit_new_pool_names_are_not_lost(migration_env):
    path, factory = migration_env
    path.write_text("EMBEDDING_HTTP_MAX_CONNECTIONS=50\nEMBEDDING_HTTP_MAX_KEEPALIVE_CONNECTIONS=25\n")
    await config_migration.migrate_env_configuration(factory, path, apply=True)
    rows = {row.key: row.value for row in await factory.repository.get_all()}
    assert rows["embedding_http_max_connections"] == "50"
    assert rows["embedding_http_max_keepalive_connections"] == "25"


@pytest.mark.asyncio
async def test_new_and_old_pool_conflict_blocks_migration(migration_env):
    path, factory = migration_env
    path.write_text(path.read_text() + "EMBEDDING_HTTP_MAX_CONNECTIONS=50\n")
    with pytest.raises(ValueError, match="EMBEDDING_HTTP_MAX_CONNECTIONS"):
        await config_migration.migrate_env_configuration(factory, path, apply=True)
    assert await factory.repository.get_all() == []


@pytest.mark.asyncio
async def test_empty_optional_provider_key_is_preserved(migration_env, monkeypatch):
    path, factory = migration_env
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    path.write_text(path.read_text() + "OPENROUTER_API_KEY=''\n")
    await config_migration.migrate_env_configuration(factory, path, apply=True)
    assert dotenv_values(path.with_name(".env.secrets"))["OPENROUTER_API_KEY"] == ""


@pytest.mark.asyncio
async def test_lowercase_secret_conflict_is_not_missed(migration_env):
    path, factory = migration_env
    path.write_text("s3_secret_key=lowercase-secret\n")
    path.with_name(".env.secrets").write_text("S3_SECRET_KEY=other-secret\n")
    with pytest.raises(ValueError, match="S3_SECRET_KEY"):
        await config_migration.migrate_env_configuration(factory, path, apply=True)
    assert await factory.repository.get_all() == []


@pytest.mark.asyncio
async def test_lowercase_settings_and_secrets_follow_settings_case_rules(migration_env):
    path, factory = migration_env
    path.write_text("llm_model=legacy-model\ns3_secret_key=lowercase-secret\n")
    await config_migration.migrate_env_configuration(factory, path, apply=True)
    assert (await factory.repository.get_by_key("llm_model")).value == "legacy-model"
    assert "s3_secret_key" not in dotenv_values(path)
    assert dotenv_values(path.with_name(".env.secrets"))["S3_SECRET_KEY"] == "lowercase-secret"
