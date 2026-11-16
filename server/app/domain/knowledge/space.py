"""Knowledge space: an area of the organization with its own documents and one domain pack.

A space selects which documents a question searches and whose rules answer it; it never
grants or restricts reading — visibility does. Filling a space with documents is limited
to the groups admitted to it. The pack of a space is fixed at creation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from domain.identity.roles import UserKind
from domain.shared.exceptions import ValidationError


if TYPE_CHECKING:
    from collections.abc import Iterable

    from domain.identity.user_context import UserContext


MAX_SPACE_NAME_LENGTH = 100
# Metadata key of the space label in chunk payloads and retrieved documents.
SPACE_ID_KEY = "space_id"


def normalize_space_name(name: str) -> str:
    normalized = name.strip()

    if not normalized:
        msg = "Space name must not be empty"
        raise ValidationError(msg, field="name")

    if len(normalized) > MAX_SPACE_NAME_LENGTH:
        msg = f"Space name must be at most {MAX_SPACE_NAME_LENGTH} characters"
        raise ValidationError(msg, field="name")

    return normalized


@dataclass(frozen=True)
class SpaceRef:
    """The space of one question or document: its id and the key of its pack."""

    id: int
    pack_key: str


@dataclass
class KnowledgeSpace:
    name: str
    pack_key: str
    id: int | None = None
    upload_group_ids: frozenset[int] = field(default_factory=frozenset)
    creation_date: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        self.name = normalize_space_name(self.name)

        if not self.pack_key.strip():
            msg = "Space pack must not be empty"
            raise ValidationError(msg, field="pack_key")

        self.upload_group_ids = frozenset(self.upload_group_ids)

    @property
    def ref(self) -> SpaceRef:
        if self.id is None:
            msg = "Space is not persisted"
            raise RuntimeError(msg)

        return SpaceRef(self.id, self.pack_key)

    def rename(self, name: str) -> None:
        self.name = normalize_space_name(name)

    def admit_groups(self, group_ids: Iterable[int]) -> None:
        self.upload_group_ids = frozenset(group_ids)

    def can_be_filled_by(self, user: UserContext) -> bool:
        """ADMIN fills any space; others only through a group admitted to this space."""
        if user.user_kind == UserKind.CLIENT:
            return False

        if user.is_admin:
            return True

        return not self.upload_group_ids.isdisjoint(user.group_ids)
