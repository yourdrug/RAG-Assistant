"""Act editions in retrieval: which edition is in force, and one edition per act in the context."""

from __future__ import annotations

from datetime import date, datetime
import logging
from typing import TYPE_CHECKING, Any

from domain.knowledge.acts.edition_metadata import (
    ACT_ID,
    ACT_VERSION_ID,
    EFFECTIVE_FROM,
    EFFECTIVE_TO,
    IS_CURRENT,
    edition_id,
)
from domain.knowledge.pack_metadata import (
    AllOf,
    AnyOf,
    DateAfter,
    DateOnOrBefore,
    Equals,
    Missing,
    Not,
    pack_metadata_of,
    parse_date,
)


if TYPE_CHECKING:
    from collections.abc import Mapping

    from langchain_core.documents import Document

    from domain.knowledge.pack_metadata import MetadataCondition


log = logging.getLogger("default")


def today() -> date:
    # The local calendar day decides which edition is in force.
    return datetime.now().astimezone().date()


def edition_selection(as_of_date: date | None, *, current_day: date | None = None) -> MetadataCondition:
    """Chunks outside editions, or of an edition in force on the requested date.

    Intervals are half-open: effective_from <= date < effective_to. The same interval
    logic serves current and historical questions, so a future edition becomes current
    on its effective date without an ingestion job. An undated edition passes only
    while it is the current one.
    """
    effective_date = as_of_date or current_day or today()
    return AnyOf(
        (
            Missing(ACT_VERSION_ID),
            AllOf(
                (
                    AnyOf((Missing(EFFECTIVE_FROM), DateOnOrBefore(EFFECTIVE_FROM, effective_date))),
                    AnyOf((Missing(EFFECTIVE_TO), DateAfter(EFFECTIVE_TO, effective_date))),
                    AnyOf(
                        (
                            Equals(IS_CURRENT, value=True),
                            Not(Missing(EFFECTIVE_FROM)),
                            Not(Missing(EFFECTIVE_TO)),
                        )
                    ),
                )
            ),
        )
    )


def resolve_edition_conflicts(docs: list[tuple[Document, float]]) -> list[tuple[Document, float]]:
    """Keep all retrieved chunks from the newest eligible edition of each act."""
    editions_by_act: dict[int, dict[int, list[tuple[Document, float]]]] = {}
    resolved: list[tuple[Document, float]] = []

    for doc, score in docs:
        metadata = pack_metadata_of(doc.metadata)
        act_id = metadata.get(ACT_ID)
        version_id = edition_id(metadata)

        if not isinstance(act_id, int) or version_id is None:
            resolved.append((doc, score))
            continue

        editions_by_act.setdefault(act_id, {}).setdefault(version_id, []).append((doc, score))

    for act_id, editions in editions_by_act.items():
        if len(editions) == 1:
            resolved.extend(next(iter(editions.values())))
            continue

        best_id = max(
            editions, key=lambda version_id: (effective_from_of(editions[version_id][0][0]), version_id)
        )
        best = editions[best_id]
        resolved.extend(best)
        dropped = sum(len(chunks) for chunks in editions.values()) - len(best)
        log.warning(
            "Temporal conflict act_id=%d: kept version with effective_from=%s, dropped %d older versions",
            act_id,
            effective_from_of(best[0][0]),
            dropped,
        )

    return resolved


def edition_of(metadata: Mapping[str, Any]) -> int | None:
    """Edition of a retrieved chunk; ``None`` outside editions."""
    return edition_id(pack_metadata_of(metadata))


def effective_from_of(doc: Document) -> date:
    return parse_date(pack_metadata_of(doc.metadata).get(EFFECTIVE_FROM)) or date.min


def edition_source_fields(metadata: Mapping[str, Any]) -> dict[str, str]:
    pack_metadata = pack_metadata_of(metadata)
    fields = {key: pack_metadata.get(key) for key in (EFFECTIVE_FROM, EFFECTIVE_TO)}
    return {key: value for key, value in fields.items() if isinstance(value, str) and value}
