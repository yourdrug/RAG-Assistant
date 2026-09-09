"""Application bootstrap -- creates the default admin user and loads dynamic config from DB.

Called once during FastAPI lifespan startup.  If the admin user already
exists, the step is a no-op.  Dynamic config parameters are loaded from the
``config_parameters`` table and applied to in-memory settings.  If the table
is empty, default values are seeded from the current env-based settings.
"""

from __future__ import annotations

import json
import logging

from config import settings
from domain.entities.config_parameter import ConfigParameter
from domain.events.config_events import ConfigParameterChanged
from domain.value_objects.dynamic_config import DYNAMIC_PARAMS
from domain.value_objects.roles import UserKind, UserRole
from infrastructure.auth.password_hasher import BCryptPasswordHasher
from infrastructure.events.in_process_event_bus import event_bus

logger = logging.getLogger("default")


def _param(key, value, vtype, cat, desc, min_v=None, max_v=None, allowed=None, domain_key=None):
    """Build a ConfigParameter entity for seeding from env defaults."""
    return ConfigParameter(
        key=key,
        value=value,
        value_type=vtype,
        category=cat,
        description=desc,
        min_value=min_v,
        max_value=max_v,
        allowed_values=allowed,
        domain_key=domain_key,
    )


def _build_defaults() -> list[ConfigParameter]:
    """Build all default config parameters from current env settings.

    Derived from ``DYNAMIC_PARAMS`` — the single source of truth for
    hot-reloadable config parameters.
    """
    s = settings
    result: list[ConfigParameter] = []
    for p in DYNAMIC_PARAMS:
        # Domain-specific params with no env var: use the declared default value
        if p.domain_key and not hasattr(s, p.key):
            raw = p.min_val if p.type in (int, float) and p.min_val is not None else None
            if p.type is int:
                value = str(int(raw)) if raw is not None else "0"
            elif p.type is float:
                value = str(raw) if raw is not None else "0.0"
            else:
                value = ""
        else:
            raw = getattr(s, p.key)
            if p.type is bool:
                value = json.dumps(raw)
            elif p.type is list:
                value = json.dumps(raw)
            elif p.type is float and raw is None:
                value = str(p.min_val if p.min_val is not None else 0.0)
            elif p.type in (int, float):
                value = str(raw)
            else:
                value = str(raw)
        result.append(
            _param(
                p.key, value, p.type.__name__, p.category,
                p.description, p.min_val, p.max_val, p.allowed, p.domain_key,
            )
        )
    return result


async def initialize_app(uow_factory) -> None:
    """Run all startup initialization: bootstrap admin + seed config + load config from DB."""
    await _bootstrap_admin(uow_factory)
    await _seed_config_defaults(uow_factory)
    await _load_config_from_db(uow_factory)


async def _bootstrap_admin(uow_factory) -> None:
    """Ensure default admin user exists."""
    try:
        if not settings.admin_email or not settings.admin_password:
            logger.warning(
                "No admin exists and ADMIN_EMAIL/ADMIN_PASSWORD not set — "
                "you won't be able to log in. Set them in server/.env and restart."
            )
            return

        hasher = BCryptPasswordHasher()
        hashed = await hasher.hash(settings.admin_password)
        async with uow_factory.create(master=True) as uow:
            await uow.users.ensure_admin(
                email=settings.admin_email,
                hashed_password=hashed,
                role=UserRole.ADMIN,
                kind=UserKind.INTERNAL,
            )
            logger.info("Admin ensured: %s", settings.admin_email)
    except Exception as e:
        logger.warning("Failed to bootstrap admin: %s", e)


async def _seed_config_defaults(uow_factory) -> None:
    """Seed config_parameters table with defaults from env.

    Uses upsert: existing parameters keep their DB values,
    missing parameters are added from env defaults.
    Category is always synced to the current definition.
    """
    try:
        async with uow_factory.create(master=True) as uow:
            existing = await uow.config_parameters.get_all()
            existing_pairs = {(p.key, p.domain_key) for p in existing}
            existing_cat_pairs = {(p.key, p.domain_key): p.category for p in existing}

            # Cleanup: if a param is now domain-specific but a stale global
            # row (domain_key=NULL) still exists, delete it.
            defaults = _build_defaults()
            for stale in existing:
                if stale.domain_key is None and stale.key in {
                    p.key for p in defaults if p.domain_key
                }:
                    await uow.config_parameters.delete_by_key_and_domain(stale.key, None)
                    logger.info("Removed stale global row for domain-specific param: %s", stale.key)

            added = 0
            updated = 0
            for entity in defaults:
                pair = (entity.key, entity.domain_key)
                if pair not in existing_pairs:
                    # upsert: another process may seed the same row in parallel
                    # at boot (M-2). upsert NOTIFYs other processes atomically.
                    await uow.config_parameters.upsert(entity)
                    added += 1
                elif existing_cat_pairs.get(pair) != entity.category:
                    await uow.config_parameters.update_category(
                        entity.key, entity.category, entity.domain_key,
                    )
                    updated += 1

            if added or updated:
                logger.info(
                    "Config parameters: added %d, category fixed %d (total: %d)",
                    added,
                    updated,
                    len(defaults),
                )
            else:
                logger.info("All %d config parameters already present", len(defaults))
    except Exception as e:
        logger.warning("Failed to seed config defaults: %s", e)


async def _seed_domain_config_defaults(uow_factory, registry) -> None:
    """Seed domain-specific config defaults from DomainProfile.config_defaults().

    Each profile declares its own defaults. Seeding is generic — adding a new
    domain profile requires zero changes here.  Newly added parameters are
    published via the event bus so the in-memory DomainSettingsAdapter picks
    them up on first boot (the bulk _load_config_from_db pass ran before seeding),
    and NOTIFYed via upsert so ALREADY-RUNNING processes resync immediately.
    """
    try:
        async with uow_factory.create(master=True) as uow:
            existing = await uow.config_parameters.get_all()
            existing_pairs = {(p.key, p.domain_key) for p in existing}
            added = 0
            for profile in registry.all():
                for default in profile.config_defaults():
                    pair = (default.key, profile.key)
                    if pair in existing_pairs:
                        continue
                    entity = ConfigParameter(
                        key=default.key,
                        value=default.value,
                        value_type=default.value_type,
                        category=f"domain:{profile.key}",
                        description=default.description,
                        min_value=default.min_value,
                        max_value=default.max_value,
                        domain_key=profile.key,
                    )
                    # upsert (not save): API and worker seed in parallel at
                    # boot — plain INSERT under the race creates duplicates
                    # (M-2) and then get_by_key_and_domain raises
                    # MultipleResultsFound forever. upsert also NOTIFYs other
                    # processes within the same transaction.
                    await uow.config_parameters.upsert(entity)
                    event_bus.publish(
                        ConfigParameterChanged(
                            key=entity.key,
                            old_value=None,
                            new_value=entity.value,
                            value_type=entity.value_type,
                            domain_key=entity.domain_key,
                        )
                    )
                    added += 1
            if added:
                logger.info("Domain config parameters: added %d", added)
            else:
                logger.info("All domain config parameters already present")
    except Exception as e:
        logger.warning("Failed to seed domain config defaults: %s", e)


async def _load_config_from_db(uow_factory) -> None:
    """При старте — прогнать все сохранённые параметры через событийную шину.

    Единый путь применения конфига: и runtime-обновления, и startup идут
    via ConfigParameterChanged → EventBus → подписчики.
    """
    try:
        async with uow_factory.create(master=True) as uow:
            rows = await uow.config_parameters.get_all()
            for r in rows:
                event_bus.publish(
                    ConfigParameterChanged(
                        key=r.key,
                        old_value=None,
                        new_value=r.normalize(r.value),
                        value_type=r.value_type,
                        domain_key=r.domain_key,
                    )
                )
            logger.info("Loaded %d config parameters via event bus", len(rows))
    except Exception as e:
        logger.warning("Failed to load config from DB: %s", e)
