"""A channel's audience applies to every document of the channel."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from application.uow import UnitOfWork
    from domain.knowledge.document import Document
    from domain.sync.channel import SyncChannel


def audience_matches(channel: SyncChannel, document: Document) -> bool:
    audience = channel.settings.audience
    return (document.visibility, document.owner_id, document.group_id) == (
        audience.visibility,
        audience.owner_id,
        audience.group_id,
    )


async def apply_audience(uow: UnitOfWork, channel: SyncChannel, document: Document) -> bool:
    """Give the document and its chunks the channel's audience; ``True`` when it changed.

    Search authorizes against these rows, so narrowed access applies at once, before
    the vector payloads are rewritten by the next indexing of the document.
    """
    if audience_matches(channel, document) or document.id is None:
        return False

    audience = channel.settings.audience
    await uow.documents.set_acl(document.id, audience.visibility, audience.owner_id, audience.group_id)
    await uow.chunks.set_acl(document.id, audience.visibility.value, audience.owner_id, audience.group_id)
    return True
