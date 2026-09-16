"""DocumentVisibility value object -- ACL visibility levels for documents."""

from __future__ import annotations

from enum import StrEnum

from domain.value_objects.base import ValidatedEnumMixin


class DocumentVisibility(ValidatedEnumMixin, StrEnum, label="visibility"):
    INTERNAL_PUBLIC = "internal_public"
    INTERNAL_GROUP = "internal_group"
    INTERNAL_PRIVATE = "internal_private"
    CLIENT_PRIVATE = "client_private"
