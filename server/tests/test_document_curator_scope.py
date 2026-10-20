"""Characterize document mutation permissions before sharing curator scope checks."""

from typing import Any

import pytest

from domain.identity.roles import UserRole
from domain.knowledge.document import Document
from domain.knowledge.enums import DocumentVisibility


@pytest.mark.parametrize("method", ["can_be_deleted_by", "can_edit_chunks"])
@pytest.mark.parametrize("role", list(UserRole))
@pytest.mark.parametrize(
    "document, scope, curator_allowed, member_delete",
    [
        ((DocumentVisibility.CLIENT_PRIVATE, 10, None), {"managed_client_ids": [10]}, True, False),
        ((DocumentVisibility.CLIENT_PRIVATE, 20, None), {"managed_client_ids": [10]}, False, False),
        ((DocumentVisibility.INTERNAL_PRIVATE, 10, None), {"managed_internal_ids": [10]}, True, False),
        ((DocumentVisibility.INTERNAL_PRIVATE, 20, None), {"managed_internal_ids": [10]}, False, False),
        ((DocumentVisibility.INTERNAL_GROUP, None, 5), {"managed_group_ids": [5]}, True, False),
        ((DocumentVisibility.INTERNAL_GROUP, None, 5), {"user_group_ids": [5]}, True, True),
        ((DocumentVisibility.INTERNAL_GROUP, None, 6), {"managed_group_ids": [5]}, False, False),
        ((DocumentVisibility.INTERNAL_GROUP, None, None), {"managed_group_ids": [5]}, False, False),
        ((DocumentVisibility.INTERNAL_PRIVATE, 20, 5), {"managed_group_ids": [5]}, False, False),
        ((DocumentVisibility.INTERNAL_GROUP, None, 5), {}, False, False),
        ((DocumentVisibility.INTERNAL_PUBLIC, None, None), {}, False, False),
        ((DocumentVisibility.INTERNAL_PRIVATE, 1, None), {}, True, False),
    ],
)
def test_document_mutation_scope(
    *,
    method: str,
    role: UserRole,
    document: tuple[DocumentVisibility, int | None, int | None],
    scope: dict[str, Any],
    curator_allowed: bool,
    member_delete: bool,
) -> None:
    visibility, owner_id, group_id = document
    doc = Document(visibility=visibility, owner_id=owner_id, group_id=group_id)
    expected = role == UserRole.ADMIN or owner_id == 1

    if role == UserRole.CURATOR:
        expected = expected or curator_allowed
    elif role == UserRole.USER and method == "can_be_deleted_by":
        expected = expected or member_delete

    assert getattr(doc, method)(1, role, **scope) is expected
