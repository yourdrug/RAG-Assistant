"""Required DB configuration bootstrap; admin creation is API-only."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from application.uow.factory import UnitOfWorkFactory
    from domain.ingestion.profiles.registry import DomainProfileRegistry
    from infrastructure.ml.config.provider_configuration import SettingsProviderConfiguration

from application.runtime.config import ConfigParameter
from application.runtime.dynamic_params import DYNAMIC_PARAMS, profile_param_category
from config import Settings, settings
from domain.identity.roles import UserKind, UserRole
from infrastructure.auth.password_hasher import BCryptPasswordHasher


logger = logging.getLogger("default")


def build_defaults() -> list[ConfigParameter]:
    result = []

    for declaration in DYNAMIC_PARAMS:
        raw = Settings.model_fields[declaration.key].get_default(call_default_factory=True)

        if raw is None:
            raw = declaration.min_val if declaration.min_val is not None else 0.0

        value = json.dumps(raw) if declaration.type in (bool, list) else str(raw)
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
    if not settings.admin_email or not settings.admin_password.get_secret_value():
        logger.warning("ADMIN_EMAIL/ADMIN_PASSWORD are not set; default admin is not created")
        return

    hashed = await BCryptPasswordHasher().hash(settings.admin_password.get_secret_value())
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
                        profile_param_category(default.key, profile.key),
                        description=default.description,
                        min_value=default.min_value,
                        max_value=default.max_value,
                        domain_key=profile.key,
                    )
                )


async def load_config_from_db(
    uow_factory: UnitOfWorkFactory,
    configuration: SettingsProviderConfiguration,
    *,
    require_complete: bool = True,
) -> None:
    async with uow_factory.create(master=True) as uow:
        rows = await uow.config_parameters.get_all()

    version = configuration.snapshot_version(rows)

    async def verify_snapshot() -> bool:
        async with uow_factory.create(master=True) as uow:
            latest = await uow.config_parameters.get_all()
        return configuration.snapshot_version(latest) == version

    await configuration.apply_async(rows, require_complete=require_complete, verify=verify_snapshot)
