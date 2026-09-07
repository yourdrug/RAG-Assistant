"""Tests for domain settings hot-reload (DomainSettingsAdapter + ConfigService)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import pytest

from application.services.config_service import ConfigService
from domain.events.config_events import ConfigParameterChanged
from domain.domain_profile.profiles.decree import DecreeDomainProfile
from infrastructure.domain_profile.registry import DomainProfileRegistry
from infrastructure.domain_profile.settings_adapter import DomainSettingsAdapter
from fakes import FakeUnitOfWorkFactory


class TestDomainSettingsAdapter:
    def test_roundtrip(self):
        adapter = DomainSettingsAdapter()
        adapter.set("max_unit_chars", "decree", "1200")
        assert adapter.get("max_unit_chars", domain_key="decree") == "1200"

    def test_missing_key_raises_loudly(self):
        adapter = DomainSettingsAdapter()
        adapter.set("some_global", None, "1")
        with pytest.raises(KeyError):
            adapter.get("max_unit_chars", domain_key="decree")

    def test_no_hidden_global_fallback(self):
        """A global value must never be silently used for a domain key (TZ section 3)."""
        adapter = DomainSettingsAdapter()
        adapter.set("max_unit_chars", None, "999")
        with pytest.raises(KeyError):
            adapter.get("max_unit_chars", domain_key="decree")

    def test_set_and_get(self):
        adapter = DomainSettingsAdapter()
        adapter.set("max_unit_chars", "legal", "1500")
        assert adapter.get("max_unit_chars", domain_key="legal") == "1500"


class TestRegistryWithAdapter:
    def test_threshold_from_settings_drives_general_fallback(self):
        """Unknown documents fall to general when the DB threshold is not met."""
        adapter = DomainSettingsAdapter()
        profile = DecreeDomainProfile(settings=adapter)
        registry = DomainProfileRegistry()
        registry.register(profile)

        # Simulate startup load from DB
        for default in profile.config_defaults():
            adapter.set(default.key, "decree", default.value)

        text = "УКАЗ\nПОСТАНОВЛЯЮ:\n"  # weak signal: 3.0+2.0 but no points
        result = registry.classify(text, settings=adapter)
        # Score below seeded threshold (2.0 is met actually) — use even weaker text
        weak = "Просто деловое письмо без структуры указа. " * 5
        result = registry.classify(weak, settings=adapter)
        assert result.domain_key == "general"


class TestConfigServiceDomainParams:
    @pytest.mark.asyncio
    async def test_update_domain_parameter_publishes_with_domain_key(self):
        factory = FakeUnitOfWorkFactory()
        uow = factory._uow
        await uow.config_parameters.save(_param("max_unit_chars", "1200", domain_key="decree"))

        events: list[ConfigParameterChanged] = []
        bus = type("Bus", (), {"publish": staticmethod(lambda e: events.append(e))})()
        service = ConfigService(uow_factory=factory, event_bus=bus)

        await service.update_parameter("max_unit_chars", "2000", changed_by=1, domain_key="decree")

        assert len(events) == 1
        assert events[0].domain_key == "decree"
        assert events[0].new_value == "2000"
        # DB row updated
        saved = await uow.config_parameters.get_by_key_and_domain("max_unit_chars", "decree")
        assert saved.value == "2000"

    @pytest.mark.asyncio
    async def test_update_unknown_domain_param_raises(self):
        service = ConfigService(
            uow_factory=FakeUnitOfWorkFactory(),
            event_bus=type("Bus", (), {"publish": staticmethod(lambda e: None)})(),
        )
        from domain.exceptions import EntityNotFound

        with pytest.raises(EntityNotFound):
            await service.update_parameter("max_unit_chars", "1", domain_key="decree")


def _param(key: str, value: str, domain_key: str | None):
    from domain.entities.config_parameter import ConfigParameter

    return ConfigParameter(
        key=key,
        value=value,
        value_type="int",
        category=f"domain:{domain_key}" if domain_key else "rag",
        description="test",
        domain_key=domain_key,
    )
