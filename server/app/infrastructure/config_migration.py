"""Explicit, offline env-to-database migration. No ML clients are initialized."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import math
import os
import tempfile
from domain.runtime.enums import MLProvider
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import Settings
from dotenv import dotenv_values
from dotenv.parser import parse_stream
from pydantic import TypeAdapter

from domain.runtime.config import ConfigParameter
from domain.shared.exceptions import ValidationError as DomainValidationError
from domain.runtime.config import ConfigValueType
from domain.runtime.dynamic_params import (
    DYNAMIC_PARAMS,
)
from domain.runtime.setting_keys import (
    SECRET_SETTING_KEYS,
    TARGET_DYNAMIC_SETTING_KEYS,
)
from domain.runtime.enums import LLMProvider

# Legacy names are recognized only by the offline migration parser.
LEGACY_POOL_SETTING_KEYS: frozenset[str] = frozenset({"http_pool_max_connections", "http_pool_max_keepalive"})

logger = logging.getLogger("default")
S3_CONNECTION_KEYS = frozenset({"s3_endpoint", "s3_bucket", "s3_region", "s3_access_key", "s3_secret_key"})
POOL_KEYS = {
    "embedding_http_max_connections": ("http_pool_max_connections", 1),
    "rerank_http_max_connections": ("http_pool_max_connections", 2),
    "embedding_http_max_keepalive_connections": ("http_pool_max_keepalive", 1),
    "rerank_http_max_keepalive_connections": ("http_pool_max_keepalive", 2),
}
LEGACY_POOL_DEFAULTS = {"http_pool_max_connections": 40, "http_pool_max_keepalive": 20}


@dataclass(frozen=True)
class EnvSnapshot:
    path: Path
    text: str
    values: dict[str, str | None]
    exists: bool


def read_snapshot(path: Path, optional: bool = False) -> EnvSnapshot:
    if path.is_symlink():
        raise ValueError("Migration does not accept symlink env files")

    exists = path.exists()
    text = path.read_text() if exists else ""

    if not exists and not optional:
        raise ValueError("Env file does not exist")

    if any(binding.error for binding in parse_stream(io.StringIO(text))):
        raise ValueError("Malformed env file")

    # Do not interpolate from an unrelated process environment into a protected file.
    values = {
        key.upper(): value
        for key, value in dotenv_values(stream=io.StringIO(text), interpolate=False).items()
    }
    return EnvSnapshot(path, text, values, exists)


def quoted_env_value(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def rewrite_env(
    snapshot: EnvSnapshot,
    removed: set[str],
    updates: dict[str, str],
    source: EnvSnapshot | None = None,
) -> str:
    pieces = [
        binding.original.string
        for binding in parse_stream(io.StringIO(snapshot.text))
        if binding.key is None or (binding.key.upper() not in removed and binding.key.upper() not in updates)
    ]
    text = "".join(pieces)

    if updates:
        if text and not text.endswith("\n"):
            text += "\n"

        # Preserve secret syntax: dotenv and Compose decode quoted backslashes differently.
        bindings = {
            binding.key.upper(): binding.original.string.replace(binding.key, binding.key.upper(), 1)
            for binding in parse_stream(io.StringIO(source.text if source else ""))
            if binding.key is not None
        }

        for key, value in updates.items():
            text += bindings[key] if source else f"{key}={quoted_env_value(value)}\n"

            if not text.endswith("\n"):
                text += "\n"

    return text


def protected_backup(snapshot: EnvSnapshot) -> None:
    if not snapshot.exists:
        return

    descriptor, name = tempfile.mkstemp(prefix=snapshot.path.name + ".backup-", dir=snapshot.path.parent)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(snapshot.text)
        stream.flush()
        os.fsync(stream.fileno())
    logger.info("Protected backup created for %s", snapshot.path.name)


def atomic_replace(snapshot: EnvSnapshot, text: str) -> None:
    current = read_snapshot(snapshot.path, optional=True)

    if current.text != snapshot.text or current.exists != snapshot.exists:
        raise ValueError("Env file changed during migration; rerun dry run")

    descriptor, name = tempfile.mkstemp(prefix=snapshot.path.name + ".migration-", dir=snapshot.path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, snapshot.path)
        directory = os.open(snapshot.path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def migration_parameter(key: str, raw: str) -> ConfigParameter:
    declaration = next((p for p in DYNAMIC_PARAMS if p.key == key), None)
    field = Settings.model_fields.get(key)
    value_type = declaration.type if declaration else (field.annotation if field else int)
    try:
        if "${" in raw:
            raise ValueError("Unresolved interpolation")

        if value_type is None:
            raise ValueError("Missing parameter type")

        adapter: TypeAdapter[Any] = TypeAdapter(field.rebuild_annotation() if field else value_type)
        value = adapter.validate_json(raw) if value_type is list else adapter.validate_python(raw)

        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Nonfinite number")

        encoded = json.dumps(value) if isinstance(value, (bool, list)) else str(value)
        entity = ConfigParameter(
            key=key,
            value=encoded,
            value_type=ConfigValueType(value_type.__name__) if value_type is not list else "list",
            category=declaration.category if declaration else "migration_pending",
            description=declaration.description if declaration else "Pending implementation; env retained",
            min_value=declaration.min_val if declaration else None,
            max_value=declaration.max_val if declaration else None,
            allowed_values=declaration.allowed if declaration else None,
            domain_key=declaration.domain_key if declaration else None,
        )
        entity.validate(encoded)

        if (
            key.startswith("rate_limit_")
            and isinstance(value, int)
            and not isinstance(value, bool)
            and value <= 0
        ):
            raise ValueError("Quota must be positive")

        if key in {"max_upload_size_mb", "job_cleanup_days"} and value <= 0:
            raise ValueError("Must be positive")

        if key in {"cost_hourly_limit", "cost_daily_limit"} and value <= 0:
            raise ValueError("Budget must be positive")

        return entity
    except (ValueError, TypeError, DomainValidationError):
        # Pydantic and domain exceptions can echo the input, including credentials in URLs.
        raise ValueError(f"Invalid legacy value: {key.upper()}") from None


def legacy_pool_limits(values: dict[str, str | None]) -> dict[str, int]:
    limits = {}

    for source in {key for key, divisor in POOL_KEYS.values()}:
        raw = values.get(source.upper(), str(LEGACY_POOL_DEFAULTS[source]))
        try:
            if raw is None:
                raise ValueError("Missing pool limit")

            limits[source] = int(raw)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid legacy value: {source.upper()}") from None

    connections = limits["http_pool_max_connections"]
    keepalive = limits["http_pool_max_keepalive"]

    if connections < 2:
        raise ValueError("Invalid legacy value: HTTP_POOL_MAX_CONNECTIONS")

    if keepalive < 0 or keepalive > connections:
        raise ValueError("Invalid legacy value: HTTP_POOL_MAX_KEEPALIVE")

    return limits


def pool_parameters(values: dict[str, str | None]) -> list[ConfigParameter]:
    names = set(POOL_KEYS) | {key for key, divisor in POOL_KEYS.values()}

    if not any(key.upper() in values for key in names):
        return []

    limits = legacy_pool_limits(values)
    result = {}

    for key, (source, divisor) in POOL_KEYS.items():
        # Preserve the old factories' integer division, including odd overrides.
        derived = limits[source] // divisor
        explicit = values.get(key.upper())
        entity = migration_parameter(key, explicit if explicit is not None else str(derived))
        value = int(entity.value)

        if explicit is not None and source.upper() in values and value != derived:
            raise ValueError(f"Legacy/new pool conflict: {key.upper()}")

        result[key] = entity

    for prefix in ("embedding", "rerank"):
        connections = int(result[f"{prefix}_http_max_connections"].value)
        keepalive = int(result[f"{prefix}_http_max_keepalive_connections"].value)

        if connections <= 0 or keepalive < 0 or keepalive > connections:
            raise ValueError(
                f"Invalid pool limits: {prefix.upper()}_HTTP_MAX_CONNECTIONS / KEEPALIVE_CONNECTIONS"
            )

    return list(result.values())


def secret_moves(env: EnvSnapshot, secrets: EnvSnapshot) -> dict[str, str]:
    for key in {key.upper() for key in SECRET_SETTING_KEYS}:
        sources = [source[key] for source in (env.values, secrets.values, os.environ) if key in source]

        if sources and (sources[0] is None or "${" in sources[0] or any(v != sources[0] for v in sources)):
            raise ValueError(f"Secret source conflict or missing credential: {key}")

    return {
        key.upper(): value
        for key, value in env.values.items()
        if key.lower() in SECRET_SETTING_KEYS and value is not None
    }


def compose_updates(env: EnvSnapshot, effective: dict[str, str | None]) -> dict[str, str]:
    if "COMPOSE_PROFILES" in effective:
        return {}

    profiles = [p.strip() for p in (effective.get("COMPOSE_PROFILES") or "").split(",") if p.strip()]
    provider = effective.get("ML_PROVIDER", Settings.model_fields["ml_provider"].default)

    if provider == MLProvider.TEI.value and "tei" not in profiles:
        profiles.append("tei")

    joined = ",".join(profiles)
    return {"COMPOSE_PROFILES": joined} if profiles and env.values.get("COMPOSE_PROFILES") != joined else {}


def validate_storage_and_credentials(
    rows: list[ConfigParameter], effective: dict[str, str | None]
) -> list[ConfigParameter]:
    s3_rows = [row for row in rows if row.key in S3_CONNECTION_KEYS]
    conflicts = {
        row.key.upper()
        for row in s3_rows
        if row.value != effective.get(row.key.upper(), Settings.model_fields[row.key].default)
    }

    if conflicts:
        raise ValueError("S3 env/DB mismatch: " + ", ".join(sorted(conflicts)))

    required = {"DB_PASSWORD", "REDIS_PASSWORD", "JWT_SECRET_KEY", "S3_ACCESS_KEY", "S3_SECRET_KEY"}
    providers = {
        key: effective.get(key.upper(), Settings.model_fields[key].default)
        for key in ("llm_provider", "ml_provider")
    }
    providers.update({row.key: row.value for row in rows if row.domain_key is None and row.key in providers})

    if providers["ml_provider"] == MLProvider.DEEPINFRA.value:
        required.add("DEEPINFRA_API_KEY")

    if providers["llm_provider"] == LLMProvider.OPENROUTER.value:
        required.add("OPENROUTER_API_KEY")

    missing = [key for key in sorted(required) if not effective.get(key) or effective[key] == "minioadmin"]

    if missing:
        raise ValueError("Missing credential: " + ", ".join(missing))

    return s3_rows


def report_actions(candidates, existing, s3_rows, moved, updates) -> None:
    for row in candidates:
        logger.info(
            "%s %s (scope=%s)",
            "Preserve DB" if (row.key, row.domain_key) in existing else "Insert",
            row.key.upper(),
            row.domain_key or "global",
        )

    for row in s3_rows:
        logger.info("Remove agreed legacy DB row %s (scope=%s)", row.key.upper(), row.domain_key or "global")

    for key in sorted(moved):
        logger.info("Move %s to secret env", key)

    for key in updates:
        logger.info("Prepare %s", key)


async def migrate_env_configuration(
    uow_factory, env_file: Path, apply: bool = False, trim_env: bool = False
) -> None:
    """Preserve DB values; commit before files; default to read-only action reporting.

    Run offline with old configuration writers stopped. Future keys are staged,
    never added to the active registry here. Final trimming is explicit and retains
    every static override; database commit precedes removal of migrated values.
    """
    env = await asyncio.to_thread(read_snapshot, env_file)
    secrets = await asyncio.to_thread(read_snapshot, env_file.with_name(env_file.name + ".secrets"), True)
    moved = secret_moves(env, secrets)
    effective = {**env.values, **secrets.values, **{key.upper(): value for key, value in os.environ.items()}}
    candidates = [
        migration_parameter(key, raw)
        for key in sorted(TARGET_DYNAMIC_SETTING_KEYS - POOL_KEYS.keys())
        if (raw := effective.get(key.upper())) is not None
    ]
    candidates += pool_parameters(effective)
    updates = compose_updates(env, effective)
    removed = set(moved)

    if trim_env:
        removed.update(key.upper() for key in TARGET_DYNAMIC_SETTING_KEYS | LEGACY_POOL_SETTING_KEYS)
        removed.update({"EMBED_MODEL", "RERANK_MODEL", "RERANK_DEVICE"})

    env_text = rewrite_env(env, removed, updates)
    secret_text = rewrite_env(secrets, set(), moved, source=env)
    async with uow_factory.create(master=True) as uow:
        rows = await uow.config_parameters.get_all()
        s3_rows = validate_storage_and_credentials(rows, effective)
        existing = {(row.key, row.domain_key) for row in rows}
        additions = [row for row in candidates if (row.key, row.domain_key) not in existing]
        report_actions(candidates, existing, s3_rows, moved, updates)

        for key in sorted((removed - set(moved)) & env.values.keys()):
            logger.info("Remove env key after commit: %s", key)

        if apply:
            await asyncio.to_thread(protected_backup, env)
            await asyncio.to_thread(protected_backup, secrets)

            for row in additions:
                # Plain INSERT relies on the DB uniqueness constraint: a race fails
                # the transaction instead of overwriting another writer's value.
                await uow.config_parameters.save(row)

            for row in s3_rows:
                await uow.config_parameters.delete_by_key_and_domain(row.key, row.domain_key)

    if apply:
        # Install secrets first: if the second replacement fails, the old env
        # still contains them and the next run sees identical sources.
        if secret_text != secrets.text:
            await asyncio.to_thread(atomic_replace, secrets, secret_text)
        elif secrets.exists:
            await asyncio.to_thread(secrets.path.chmod, 0o600)

        if env_text != env.text:
            await asyncio.to_thread(atomic_replace, env, env_text)

    logger.info("Configuration migration %s", "applied" if apply else "dry run; no changes")
