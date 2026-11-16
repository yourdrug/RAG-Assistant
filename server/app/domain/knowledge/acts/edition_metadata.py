"""Pack metadata of an act edition: the keys the law pack writes on chunks and reads back."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Mapping

    from domain.knowledge.acts.models import ActVersion


ACT_VERSION_ID = "act_version_id"
ACT_ID = "act_id"
EFFECTIVE_FROM = "effective_from"
EFFECTIVE_TO = "effective_to"
IS_CURRENT = "is_current"


def edition_metadata(version: ActVersion) -> dict[str, Any]:
    """Every chunk of the edition's document carries this dictionary."""
    return {
        ACT_VERSION_ID: version.id,
        ACT_ID: version.act_id,
        IS_CURRENT: version.is_current,
        EFFECTIVE_FROM: version.effective_from.isoformat() if version.effective_from else None,
        EFFECTIVE_TO: version.effective_to.isoformat() if version.effective_to else None,
    }


def edition_id(pack_metadata: Mapping[str, Any]) -> int | None:
    value = pack_metadata.get(ACT_VERSION_ID)
    return value if isinstance(value, int) and not isinstance(value, bool) else None
