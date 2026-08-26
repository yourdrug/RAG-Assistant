"""Config-change event wiring — subscribe / unsubscribe helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from composition.infrastructure import InfrastructureContainer


def _subscribe_config_events(infra: InfrastructureContainer) -> None:
    """Subscribe config-change handlers to the event bus.

    Subscribers that invalidate ML caches use closures over ml_clients,
    so they are created here rather than at module level.
    """
    from domain.events.config_events import ConfigParameterChanged
    from infrastructure.events.in_process_event_bus import event_bus
    from infrastructure.ml.config_subscribers import (
        apply_to_settings,
        audit_log_config_change,
        invalidate_paddle_ocr_cache,
        invalidate_pii_detector_cache,
        invalidate_storage_cache,
    )

    bus = event_bus
    ml = infra.ml_clients
    if ml is None:
        raise RuntimeError("InfrastructureContainer must be initialized before subscribing config events")

    bus.subscribe(ConfigParameterChanged, apply_to_settings)
    bus.subscribe(ConfigParameterChanged, invalidate_paddle_ocr_cache)
    bus.subscribe(ConfigParameterChanged, invalidate_storage_cache)
    bus.subscribe(ConfigParameterChanged, invalidate_pii_detector_cache)
    bus.subscribe(ConfigParameterChanged, audit_log_config_change)

    def _invalidate_llm(event: ConfigParameterChanged) -> None:
        llm_keys = {
            "llm_provider",
            "llm_model",
            "llm_temperature",
            "llm_top_p",
            "llm_num_ctx_narrow",
            "llm_num_predict_narrow",
            "openrouter_model",
        }
        if event.key in llm_keys:
            ml.invalidate_llm()

    def _invalidate_bm25(event: ConfigParameterChanged) -> None:
        if event.key == "hybrid_enabled":
            ml.invalidate_bm25()

    bus.subscribe(ConfigParameterChanged, _invalidate_llm)
    bus.subscribe(ConfigParameterChanged, _invalidate_bm25)


def _unsubscribe_config_events() -> None:
    """Remove all config-change handlers from the event bus.

    Called during ``Container.dispose()`` to avoid leaking handlers when the
    container is re-created (the event bus is a process-level singleton).
    """
    from infrastructure.events.in_process_event_bus import event_bus

    event_bus.unsubscribe_all()
