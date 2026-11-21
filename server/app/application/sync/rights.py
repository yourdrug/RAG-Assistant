"""Current rights a channel depends on: its responsible user and its audience.

Checked when a channel is created or changed and again by every run: a channel whose
responsible user lost access, or whose client owner was deactivated, publishes nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from application.identity import UserContextFactory
from domain.identity.roles import UserKind
from domain.knowledge.enums import DocumentVisibility


if TYPE_CHECKING:
    from application.uow import UnitOfWork
    from domain.knowledge.space import KnowledgeSpace
    from domain.sync.channel import ChannelSettings


# Owned by a client user. A ``client_public`` document is owned, like an uploaded one,
# by the internal user who publishes it: every client reads it, whoever owns it.
CLIENT_OWNED_VISIBILITIES = frozenset({DocumentVisibility.CLIENT_PRIVATE})


async def channel_rights_problem(
    uow: UnitOfWork,
    space: KnowledgeSpace,
    settings: ChannelSettings,
    contexts: UserContextFactory | None = None,
) -> str | None:
    """Why the channel may not publish now, or ``None`` when its rights hold."""
    return await responsible_problem(
        uow, space, settings, contexts or UserContextFactory()
    ) or await owner_problem(uow, settings)


async def responsible_problem(
    uow: UnitOfWork, space: KnowledgeSpace, settings: ChannelSettings, contexts: UserContextFactory
) -> str | None:
    if settings.responsible_user_id is None:
        return "the channel has no responsible user"

    responsible = await uow.users.get_by_id(settings.responsible_user_id)

    if responsible is None or not responsible.is_active or responsible.kind != UserKind.INTERNAL:
        return "the responsible user is not an active internal user"

    ctx = await contexts.build(uow, settings.responsible_user_id, responsible.kind, responsible.role)

    if not space.can_be_filled_by(ctx):
        return "the responsible user may not add documents to this space"

    audience = settings.audience
    outside_group = (
        audience.visibility == DocumentVisibility.INTERNAL_GROUP and audience.group_id not in ctx.group_ids
    )
    return (
        "the responsible user is not a member of the channel's group"
        if outside_group and not ctx.is_admin
        else None
    )


async def owner_problem(uow: UnitOfWork, settings: ChannelSettings) -> str | None:
    audience = settings.audience

    if audience.owner_id is None:
        return None

    owner = await uow.users.get_by_id(audience.owner_id)
    expected = UserKind.CLIENT if audience.visibility in CLIENT_OWNED_VISIBILITIES else UserKind.INTERNAL

    if owner is None or not owner.is_active or owner.kind != expected:
        return f"the document owner is not an active {expected.value} user"

    return None
