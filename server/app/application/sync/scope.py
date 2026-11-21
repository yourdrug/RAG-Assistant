"""Which storage keys a channel may read: never the application's own objects.

The deployment's bucket also holds uploaded documents (with their owners' access),
preview files and upload backups. A channel publishing them would give their content
the channel's audience under another document id, bypassing the original access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from application.documents.storage_areas import APPLICATION_STORAGE_PREFIXES
from domain.shared.exceptions import BusinessRuleViolationError
from domain.sync.channel import S3Location
from domain.sync.ignore import normalize_folder, relative_to


if TYPE_CHECKING:
    from domain.sync.channel import ChannelSettings


RESERVED_S3_PREFIXES = APPLICATION_STORAGE_PREFIXES


def reserved_overlap(folder: str, reserved: tuple[str, ...] = RESERVED_S3_PREFIXES) -> str | None:
    """The reserved prefix a channel folder contains or lies in, if any."""
    folder = normalize_folder(folder)

    for prefix in reserved:
        root = prefix.rstrip("/")

        if not folder or relative_to(folder, root) is not None or relative_to(root, folder) is not None:
            return prefix

    return None


def ensure_channel_scope(settings: ChannelSettings) -> None:
    if not isinstance(settings.location, S3Location):
        return

    prefix = reserved_overlap(settings.folder)

    if prefix is not None:
        msg = f"An S3 channel folder must not include the application's storage area '{prefix}'"
        raise BusinessRuleViolationError(msg)
