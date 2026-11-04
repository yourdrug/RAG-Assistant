"""Required DB configuration bootstrap; admin creation is API-only."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from application.uow.factory import UnitOfWorkFactory

from application.runtime.config import ConfigParameter
from application.runtime.dynamic_params import DYNAMIC_PARAMS, PROFILE_CONFIG_DEFAULTS
from config import Settings, settings
from domain.identity.roles import UserKind, UserRole
from domain.ingestion.profiles.registry import DomainProfileRegistry
from infrastructure.auth.password_hasher import BCryptPasswordHasher
from infrastructure.domain_profile import register_all_profiles
from infrastructure.domain_profile.settings_adapter import DomainSettingsAdapter
from infrastructure.ml.config.provider_configuration import SettingsProviderConfiguration


logger = logging.getLogger("default")


def build_defaults() -> list[ConfigParameter]:
    result = []

    for declaration in DYNAMIC_PARAMS:
        field = Settings.model_fields[declaration.key]
        raw = field.get_default(call_default_factory=True)

        if raw is None:
            raw = declaration.min_val if declaration.min_val is not None else 0.0

        scoped_default = (
            PROFILE_CONFIG_DEFAULTS.get((declaration.key, declaration.domain_key))
            if declaration.domain_key is not None
            else None
        )
        value = (
            scoped_default.value
            if scoped_default is not None
            else (json.dumps(raw) if declaration.type in (bool, list) else str(raw))
        )
        result.append(
            ConfigParameter(
                declaration.key,
                value,
                declaration.type.__name__,
                declaration.category,
                description=declaration.description,
                min_value=declaration.min_val,
                max_value=declaration.max_val,
                allowed_values=declaration.allowed,
                domain_key=declaration.domain_key,
            )
        )

    return result


async def bootstrap_admin(uow_factory: UnitOfWorkFactory) -> None:
    if not settings.admin_email or not settings.admin_password:
        logger.warning("ADMIN_EMAIL/ADMIN_PASSWORD are not set; default admin is not created")
        return

    hashed = await BCryptPasswordHasher().hash(settings.admin_password)
    async with uow_factory.create(master=True) as uow:
        await uow.users.ensure_admin(
            email=settings.admin_email,
            hashed_password=hashed,
            role=UserRole.ADMIN,
            kind=UserKind.INTERNAL,
        )


async def seed_config_defaults(uow_factory: UnitOfWorkFactory) -> None:
    async with uow_factory.create(master=True) as uow:
        await uow.config_parameters.lock_configuration()
        existing = await uow.config_parameters.get_all()
        pairs = {(row.key, row.domain_key): row for row in existing}
        defaults = build_defaults()
        domain_keys = {row.key for row in defaults if row.domain_key is not None}

        for row in existing:
            if row.domain_key is None and row.key in domain_keys:
                await uow.config_parameters.delete_by_key_and_domain(row.key, None)

        for entity in defaults:
            previous = pairs.get((entity.key, entity.domain_key))

            if previous is None:
                await uow.config_parameters.upsert(entity)
            elif previous.category != entity.category:
                await uow.config_parameters.update_category(entity.key, entity.category, entity.domain_key)


async def seed_domain_config_defaults(
    uow_factory: UnitOfWorkFactory, registry: DomainProfileRegistry
) -> None:
    async with uow_factory.create(master=True) as uow:
        await uow.config_parameters.lock_configuration()
        existing = await uow.config_parameters.get_all()
        pairs = {(row.key, row.domain_key) for row in existing}

        for profile in registry.all():
            for default in profile.config_defaults():
                if (default.key, profile.key) in pairs:
                    continue

                await uow.config_parameters.upsert(
                    ConfigParameter(
                        default.key,
                        default.value,
                        default.value_type,
                        f"domain:{profile.key}",
                        description=default.description,
                        min_value=default.min_value,
                        max_value=default.max_value,
                        domain_key=profile.key,
                    )
                )


async def load_config_from_db(
    uow_factory: UnitOfWorkFactory,
    configuration: SettingsProviderConfiguration | None = None,
    *,
    require_complete: bool = True,
) -> None:
    async with uow_factory.create(master=True) as uow:
        rows = await uow.config_parameters.get_all()

    if configuration is None:
        domain_settings = DomainSettingsAdapter()
        registry = DomainProfileRegistry()
        register_all_profiles(registry, domain_settings)
        configuration = SettingsProviderConfiguration(domain_settings=domain_settings, registry=registry)

    version = configuration.snapshot_version(rows)

    async def verify_snapshot() -> bool:
        async with uow_factory.create(master=True) as uow:
            latest = await uow.config_parameters.get_all()
        return configuration.snapshot_version(latest) == version

    await configuration.apply_async(rows, require_complete=require_complete, verify=verify_snapshot)
