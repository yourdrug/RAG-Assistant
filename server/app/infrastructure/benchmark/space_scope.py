"""Request-local sweep space and stable, space-specific corpus identities."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from application.benchmark.identity import fingerprint


if TYPE_CHECKING:
    from collections.abc import Iterator

    from domain.knowledge.space import SpaceRef


active_sweep_space: ContextVar[SpaceRef | None] = ContextVar("sweep_space", default=None)


@contextmanager
def sweep_space_scope(space: SpaceRef) -> Iterator[None]:
    token = active_sweep_space.set(space)
    try:
        yield
    finally:
        active_sweep_space.reset(token)


def space_corpus_metadata(metadata: dict[str, Any], space: SpaceRef | None) -> dict[str, Any]:
    if space is None:
        return metadata

    return {
        **metadata,
        "space_id": space.id,
        "pack_key": space.pack_key,
        "corpus_namespace": fingerprint(
            {"corpus": metadata, "space_id": space.id, "pack_key": space.pack_key}
        ),
    }
