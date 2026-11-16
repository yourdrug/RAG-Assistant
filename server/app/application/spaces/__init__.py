"""Knowledge space scenarios and their contracts with the loaded domain packs."""

from application.spaces.ports import DomainPackCatalog, DomainPackNotLoadedError
from application.spaces.resolution import SpacePackHooks, default_space, document_space, resolve_space
from application.spaces.service import (
    DocumentProfileOption,
    KnowledgeSpaceDTO,
    KnowledgeSpaceService,
    SpaceChanges,
)


__all__ = [
    "DocumentProfileOption",
    "DomainPackCatalog",
    "DomainPackNotLoadedError",
    "KnowledgeSpaceDTO",
    "KnowledgeSpaceService",
    "SpaceChanges",
    "SpacePackHooks",
    "default_space",
    "document_space",
    "resolve_space",
]
