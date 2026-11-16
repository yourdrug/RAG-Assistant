"""Knowledge space scenarios: listing for a user and ADMIN management.

Callers enforce the ADMIN role on management; listing is open to every authenticated
user because a space never grants access to documents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from application.identity.user_context_factory import UserContextFactory
from application.spaces.ports import DomainPackNotLoadedError
from application.spaces.resolution import SPACE_ENTITY, client_space, resolve_space
from domain.identity.roles import UserKind
from domain.knowledge.space import KnowledgeSpace
from domain.shared.exceptions import (
    BusinessRuleViolationError,
    EntityNotFoundError,
    UniqueConstraintError,
    ValidationError,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from application.spaces.ports import DomainPackCatalog
    from application.uow import UnitOfWork, UnitOfWorkFactory
    from domain.identity.user_context import UserContext
    from domain.knowledge.space import SpaceRef


@dataclass(frozen=True)
class DocumentProfileOption:
    key: str
    display_name: str


@dataclass(frozen=True)
class KnowledgeSpaceDTO:
    id: int
    name: str
    pack_key: str
    can_upload: bool
    # Only ADMIN sees which groups may fill a space.
    upload_group_ids: tuple[int, ...] | None = None

    @classmethod
    def for_user(cls, space: KnowledgeSpace, user: UserContext) -> KnowledgeSpaceDTO:
        if space.id is None:
            msg = "Space is not persisted"
            raise RuntimeError(msg)

        return cls(
            id=space.id,
            name=space.name,
            pack_key=space.pack_key,
            can_upload=space.can_be_filled_by(user),
            upload_group_ids=tuple(sorted(space.upload_group_ids)) if user.is_admin else None,
        )


@dataclass(frozen=True)
class SpaceChanges:
    name: str | None = None
    upload_group_ids: Sequence[int] | None = None


class KnowledgeSpaceService:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        catalog: DomainPackCatalog,
        user_ctx_factory: UserContextFactory | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._catalog = catalog
        self._user_ctx_factory = user_ctx_factory or UserContextFactory()

    def pack_keys(self) -> tuple[str, ...]:
        return self._catalog.pack_keys()

    async def resolve_ref(self, space_id: int | None) -> SpaceRef:
        async with self._uow_factory.create() as uow:
            return (await resolve_space(uow, space_id)).ref

    async def list_for_user(self, user_id: int, user_kind: str, user_role: str) -> list[KnowledgeSpaceDTO]:
        async with self._uow_factory.create() as uow:
            ctx = await self._user_ctx_factory.build(uow, user_id, user_kind, user_role)

            if ctx.user_kind == UserKind.CLIENT:
                assigned = await client_space(uow, user_id)
                spaces = [assigned] if assigned is not None else []
            else:
                spaces = await uow.knowledge_spaces.list_all()
        return [KnowledgeSpaceDTO.for_user(space, ctx) for space in spaces]

    async def assign_client(self, admin: UserContext, client_id: int, space_id: int | None) -> None:
        if not admin.is_admin:
            msg = "Only admin can assign a client support space"
            raise BusinessRuleViolationError(msg)

        async with self._uow_factory.create(master=True) as uow:
            user = await uow.users.get_by_id(client_id)

            if user is None:
                entity = "User"
                raise EntityNotFoundError(entity, client_id)

            if space_id is not None:
                await resolve_space(uow, space_id)

            user.assign_client_space(space_id)
            await uow.users.assign_client_space(client_id, space_id)

    async def document_profiles(self, space_id: int | None) -> list[DocumentProfileOption]:
        """Document profiles an upload into the space may request explicitly."""
        async with self._uow_factory.create() as uow:
            space = await resolve_space(uow, space_id)
        registry = self._catalog.domain_registry(space.pack_key)
        return [DocumentProfileOption(profile.key, profile.display_name) for profile in registry.all()]

    async def create(
        self, admin: UserContext, name: str, pack_key: str, upload_group_ids: Sequence[int] = ()
    ) -> KnowledgeSpaceDTO:
        if pack_key not in self._catalog.pack_keys():
            raise DomainPackNotLoadedError(pack_key)

        space = KnowledgeSpace(name=name, pack_key=pack_key, upload_group_ids=frozenset(upload_group_ids))
        async with self._uow_factory.create(master=True) as uow:
            await self.ensure_groups_exist(uow, space.upload_group_ids)
            try:
                created = await uow.knowledge_spaces.create(space)
            except UniqueConstraintError as exc:
                raise duplicate_name(space.name) from exc
        return KnowledgeSpaceDTO.for_user(created, admin)

    async def update(self, admin: UserContext, space_id: int, changes: SpaceChanges) -> KnowledgeSpaceDTO:
        """Rename a space or change its admitted groups; its pack is fixed at creation."""
        async with self._uow_factory.create(master=True) as uow:
            space = await uow.knowledge_spaces.get_by_id(space_id, for_update=True)

            if space is None:
                raise EntityNotFoundError(SPACE_ENTITY, space_id)

            if changes.name is not None:
                space.rename(changes.name)

            if changes.upload_group_ids is not None:
                space.admit_groups(changes.upload_group_ids)
                await self.ensure_groups_exist(uow, space.upload_group_ids)

            try:
                updated = await uow.knowledge_spaces.update(space)
            except UniqueConstraintError as exc:
                raise duplicate_name(space.name) from exc
        return KnowledgeSpaceDTO.for_user(updated, admin)

    async def delete(self, space_id: int) -> None:
        """Delete an empty space; its documents must be deleted first."""
        async with self._uow_factory.create(master=True) as uow:
            space = await uow.knowledge_spaces.get_by_id(space_id, for_update=True)

            if space is None:
                raise EntityNotFoundError(SPACE_ENTITY, space_id)

            if await uow.knowledge_spaces.has_documents(space_id):
                msg = "Delete the documents of the space before deleting it"
                raise BusinessRuleViolationError(msg)

            await uow.knowledge_spaces.delete(space_id)

    @staticmethod
    async def ensure_groups_exist(uow: UnitOfWork, group_ids: frozenset[int]) -> None:
        if not group_ids:
            return

        found = {group.id for group in await uow.groups.list_by_ids(sorted(group_ids))}
        missing = sorted(group_ids - found)

        if missing:
            msg = f"Unknown groups: {', '.join(map(str, missing))}"
            raise ValidationError(msg, field="upload_group_ids")


def duplicate_name(name: str) -> BusinessRuleViolationError:
    return BusinessRuleViolationError(f"A knowledge space named {name!r} already exists")
