"""Capability-based permission model — static role-to-capability map.

No database tables required.  ``_ROLE_CAPABILITIES`` is the single source
of truth for which capabilities each role possesses.  Additional kind-level
overrides (e.g. CLIENT gets ``API_KEYS_MANAGE_OWN``) are checked in
``presentation/api/auth_dependencies.py``.
"""

from __future__ import annotations

from enum import StrEnum

from domain.value_objects.roles import UserRole


class Capability(StrEnum):
    CONFIG_MANAGE = "config.manage"
    SYSTEM_OBSERVE = "system.observe"
    BENCHMARK_MANAGE = "benchmark.manage"
    USERS_MANAGE = "users.manage"
    GROUPS_MANAGE = "groups.manage"
    GROUPS_VIEW = "groups.view"
    CURATOR_ASSIGNMENTS_MANAGE = "curator_assignments.manage"
    DOCUMENTS_MANAGE = "documents.manage"
    CHUNKS_MANAGE = "chunks.manage"
    DOCUMENTS_VIEW = "documents.view"
    SEARCH_USE = "search.use"
    API_KEYS_MANAGE_OWN = "api_keys.manage_own"
    API_KEYS_MANAGE_ANY = "api_keys.manage_any"


_ROLE_CAPABILITIES: dict[UserRole, frozenset[Capability]] = {
    UserRole.ADMIN: frozenset(Capability),
    UserRole.CURATOR: frozenset(
        {
            Capability.SYSTEM_OBSERVE,
            Capability.DOCUMENTS_MANAGE,
            Capability.CHUNKS_MANAGE,
            Capability.DOCUMENTS_VIEW,
            Capability.GROUPS_VIEW,
            Capability.SEARCH_USE,
        }
    ),
    UserRole.USER: frozenset(
        {
            Capability.DOCUMENTS_VIEW,
            Capability.GROUPS_VIEW,
            Capability.SEARCH_USE,
        }
    ),
}


def get_role_capabilities(role: UserRole) -> frozenset[Capability]:
    """Return the set of capabilities for the given role."""
    return _ROLE_CAPABILITIES.get(role, frozenset())
