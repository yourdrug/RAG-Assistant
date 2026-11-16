"""ChunkIndexInspectionService -- admin diagnostics of a chunk's vector-index state.

Compares the SQL chunk (and its document's ACL) with the point stored in the
vector index, so drift between the two stores is visible without database access.
"""

from __future__ import annotations

from enum import Enum
import logging
from typing import TYPE_CHECKING

from application.chunks.dto import ChunkIndexStateDTO, IndexFieldComparison
from application.identity import UserContextFactory
from domain.identity.capabilities import Capability, get_role_capabilities
from domain.identity.roles import UserKind
from domain.knowledge.access_control import check_document_access
from domain.knowledge.space import SPACE_ID_KEY
from domain.shared.exceptions import BusinessRuleViolationError, EntityNotFoundError


if TYPE_CHECKING:
    from application.chunks.dto import IndexedPointSnapshot
    from application.chunks.ports.index_inspector import ChunkIndexInspectorPort
    from application.uow import UnitOfWorkFactory
    from domain.identity.roles import UserRole

log = logging.getLogger(__name__)


def normalize(value: object) -> str | int | None:
    if isinstance(value, Enum):
        value = value.value

    if value is None or (isinstance(value, int) and not isinstance(value, bool)):
        return value

    return str(value)


def compare_fields(expected: dict[str, object], snapshot: IndexedPointSnapshot) -> list[IndexFieldComparison]:
    comparisons = []

    for name, value in expected.items():
        want = normalize(value)
        got = normalize(snapshot.metadata.get(name))
        comparisons.append(IndexFieldComparison(name=name, expected=want, actual=got, matches=want == got))

    return comparisons


class ChunkIndexInspectionService:
    """Admin-only read of a chunk's point in the vector index."""

    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        inspector: ChunkIndexInspectorPort,
        user_ctx_factory: UserContextFactory | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._inspector = inspector
        self._user_ctx_factory = user_ctx_factory or UserContextFactory()

    async def inspect(
        self,
        document_id: int,
        chunk_id: int,
        user_id: int,
        user_kind: UserKind,
        user_role: UserRole,
    ) -> ChunkIndexStateDTO:
        # The point is read by ID without a search ACL filter, so the capability is the gate.
        if user_kind != UserKind.INTERNAL or Capability.CHUNKS_INSPECT_INDEX not in get_role_capabilities(
            user_role
        ):
            msg = "No permission to inspect the vector index"
            raise BusinessRuleViolationError(msg)

        async with self._uow_factory.create() as uow:
            doc = await uow.documents.get_by_id(document_id)

            if doc is None:
                msg = "Document"
                raise EntityNotFoundError(msg, document_id)

            ctx = await self._user_ctx_factory.build(uow, user_id, user_kind, user_role)
            check_document_access(doc, ctx)
            chunk = await uow.chunks.get_by_id(chunk_id)

            if chunk is None or chunk.document_id != document_id:
                msg = "Chunk"
                raise EntityNotFoundError(msg, chunk_id)

        # The transaction is closed before the external index call.
        snapshot = await self._inspector.inspect_point(chunk.chunk_id)
        document_status = str(normalize(doc.status) or "")

        if snapshot is None:
            return ChunkIndexStateDTO(
                chunk_id=chunk.chunk_id,
                document_id=document_id,
                document_status=document_status,
                indexed=False,
                in_sync=False,
            )

        # ACL is compared with the document, the source of truth for chunk and point ACL.
        expected: dict[str, object] = {
            "document_id": document_id,
            "chunk_id": chunk.chunk_id,
            "chunk_index": chunk.chunk_index,
            "visibility": doc.visibility,
            "owner_id": doc.owner_id,
            "group_id": doc.group_id,
            SPACE_ID_KEY: doc.space_id,
            "doc_domain": doc.doc_domain,
        }
        fields = compare_fields(expected, snapshot)
        content_matches = snapshot.page_content == chunk.content
        embedding_model = snapshot.metadata.get("embedding_model")
        in_sync = (
            content_matches
            and all(f.matches for f in fields)
            and bool(snapshot.dense_dim)
            and bool(snapshot.sparse_terms)
        )

        if not in_sync:
            log.info("Index drift for chunk %d of document %d", chunk.chunk_id, document_id)

        return ChunkIndexStateDTO(
            chunk_id=chunk.chunk_id,
            document_id=document_id,
            document_status=document_status,
            indexed=True,
            in_sync=in_sync,
            dense_dim=snapshot.dense_dim,
            sparse_terms=snapshot.sparse_terms,
            embedding_model=embedding_model if isinstance(embedding_model, str) else None,
            content_matches=content_matches,
            fields=fields,
            payload_metadata=dict(snapshot.metadata),
        )
