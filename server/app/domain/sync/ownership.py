"""In a knowledge space a storage file belongs to at most one channel.

Ownership is decided from channel settings alone, including channels that never ran:
``.ragignore`` changes independently of the channels and is unknown before a run. A
channel inside another channel's folder is allowed only when the outer channel's own
patterns exclude the inner folder entirely.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from domain.shared.exceptions import BusinessRuleViolationError
from domain.sync.ignore import relative_to


if TYPE_CHECKING:
    from collections.abc import Iterable

    from domain.sync.channel import SyncChannel


def nests_without_exclusion(outer: SyncChannel, inner: SyncChannel) -> bool:
    relative = relative_to(inner.settings.folder, outer.settings.folder)

    if relative is None:
        return False

    return relative == "" or not outer.settings.rules().excludes_directory(relative)


def channels_overlap(first: SyncChannel, second: SyncChannel) -> bool:
    if first.space_id != second.space_id or first.settings.location.key != second.settings.location.key:
        return False

    return nests_without_exclusion(first, second) or nests_without_exclusion(second, first)


def ensure_no_overlap(candidate: SyncChannel, channels: Iterable[SyncChannel]) -> None:
    overlapping = sorted(
        channel.id
        for channel in channels
        if channel.id is not None and channel.id != candidate.id and channels_overlap(candidate, channel)
    )

    if overlapping:
        ids = ", ".join(map(str, overlapping))
        msg = (
            f"Channel files overlap channel(s) {ids} in this space; "
            "exclude the nested folder in the outer channel or choose another folder"
        )
        raise BusinessRuleViolationError(msg)
