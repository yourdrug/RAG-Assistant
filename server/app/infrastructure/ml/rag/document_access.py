"""Fail-closed authorization of indexed evidence and cache dependencies."""

from application.ports.document_access import DocumentAccessPort
from domain.value_objects.user_context import UserContext


def valid_document_id(value) -> bool:
    return type(value) is int and value > 0


async def filter_documents(docs: list, access: DocumentAccessPort, user: UserContext) -> list:
    ids = {
        doc.metadata.get("document_id") for doc in docs if valid_document_id(doc.metadata.get("document_id"))
    }
    allowed = await access.allowed_document_ids(sorted(ids), user) if ids else set()
    return [
        doc
        for doc in docs
        if valid_document_id(doc.metadata.get("document_id")) and doc.metadata.get("document_id") in allowed
    ]


async def filter_scored_documents(docs: list, access: DocumentAccessPort, user: UserContext) -> list:
    permitted = await filter_documents([doc for doc, _ in docs], access, user)
    permitted_objects = {id(doc) for doc in permitted}
    return [(doc, score) for doc, score in docs if id(doc) in permitted_objects]


async def cache_is_accessible(cached: dict, access: DocumentAccessPort, user: UserContext) -> bool:
    dependencies = cached.get("document_ids")
    # Old/incomplete entries cannot prove which documents the answer used.
    if (
        not isinstance(dependencies, list)
        or not dependencies
        or not all(map(valid_document_id, dependencies))
    ):
        return False
    required = set(dependencies)
    sources = cached.get("sources")
    if not isinstance(sources, list):
        return False
    for source in sources:
        if not isinstance(source, dict) or not valid_document_id(source.get("document_id")):
            return False
        if source["document_id"] not in required:
            return False
    return required <= await access.allowed_document_ids(sorted(required), user)
