"""SQLAlchemy implementation of DocumentRepository."""

from __future__ import annotations

from domain.entities.document import Document
from domain.value_objects.document_status import DocumentStatus
from domain.value_objects.visibility import DocumentVisibility
from sqlalchemy import text
from sqlalchemy.orm import Session


class SQLAlchemyDocumentRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def save(self, document: Document) -> Document:
        result = self._db.execute(
            text("""
                INSERT INTO documents (filename, source_path, visibility, owner_id, group_id, status)
                VALUES (:filename, :source_path, :visibility, :owner_id, :group_id, 'pending')
                RETURNING id
            """),
            {
                "filename": document.filename,
                "source_path": document.source_path,
                "visibility": document.visibility,
                "owner_id": document.owner_id,
                "group_id": document.group_id,
            },
        )
        document.id = result.scalar()
        return document

    def get_by_id(self, document_id: int) -> Document | None:
        row = self._db.execute(text("SELECT * FROM documents WHERE id = :id"), {"id": document_id}).fetchone()
        return self._to_entity(row) if row else None

    def delete(self, document_id: int) -> None:
        self._db.execute(text("DELETE FROM documents WHERE id = :id"), {"id": document_id})

    def update_status(
        self,
        document_id: int,
        status: str,
        error: str | None = None,
        chunks: int | None = None,
        chars: int | None = None,
    ) -> None:
        self._db.execute(
            text("""
                UPDATE documents
                SET status = :status,
                    error_message = :error,
                    chunks = COALESCE(:chunks, chunks),
                    chars = COALESCE(:chars, chars),
                    indexed_at = CASE WHEN :status = 'done' THEN NOW() ELSE indexed_at END
                WHERE id = :id
            """),
            {"status": status, "error": error, "chunks": chunks, "chars": chars, "id": document_id},
        )

    def set_source_path(self, document_id: int, source_path: str) -> None:
        self._db.execute(
            text("UPDATE documents SET source_path = :path WHERE id = :id"),
            {"path": source_path, "id": document_id},
        )

    def find_active_slot(self, owner_id: int | None, filename: str, group_id: int | None) -> Document | None:
        row = self._db.execute(
            text("""
                SELECT * FROM documents
                WHERE filename = :filename
                  AND owner_id IS NOT DISTINCT FROM :owner_id
                  AND group_id IS NOT DISTINCT FROM :group_id
                  AND status IN ('pending', 'processing', 'done')
                ORDER BY created_at DESC
                LIMIT 1
            """),
            {"filename": filename, "owner_id": owner_id, "group_id": group_id},
        ).fetchone()
        return self._to_entity(row) if row else None

    def list_visible(
        self,
        user_kind: str,
        user_id: int,
        group_ids: list[int],
        assigned_client_ids: list[int],
    ) -> list[Document]:
        """List documents visible to this user.

        SQL conditions are derived from domain.services.access_control.can_view_document().
        Visibility string values come from the domain enum to stay in sync.
        """
        params: dict = {"user_id": user_id}

        if user_kind == "client":
            rows = self._db.execute(
                text("""
                    SELECT * FROM documents
                    WHERE visibility = :vis
                      AND owner_id = :user_id
                    ORDER BY created_at DESC
                """),
                {"vis": DocumentVisibility.CLIENT_PRIVATE, "user_id": user_id},
            ).fetchall()
        else:
            conditions = ["visibility = :vis_public"]
            params["vis_public"] = DocumentVisibility.INTERNAL_PUBLIC

            if group_ids:
                conditions.append("(visibility = :vis_group AND group_id = ANY(:group_ids))")
                params["vis_group"] = DocumentVisibility.INTERNAL_GROUP
                params["group_ids"] = group_ids

            conditions.append("(visibility = :vis_private AND owner_id = :user_id)")
            params["vis_private"] = DocumentVisibility.INTERNAL_PRIVATE

            if assigned_client_ids:
                conditions.append("(visibility = :vis_client AND owner_id = ANY(:assigned_client_ids))")
                params["vis_client"] = DocumentVisibility.CLIENT_PRIVATE
                params["assigned_client_ids"] = assigned_client_ids

            where_clause = " OR ".join(conditions)
            rows = self._db.execute(
                text(f"SELECT * FROM documents WHERE {where_clause} ORDER BY created_at DESC"),
                params,
            ).fetchall()

        return [self._to_entity(r) for r in rows]

    @staticmethod
    def _to_entity(row) -> Document:
        return Document(
            id=row.id,
            filename=row.filename,
            source_path=row.source_path,
            visibility=DocumentVisibility(row.visibility),
            owner_id=row.owner_id,
            group_id=row.group_id,
            status=DocumentStatus(row.status),
            error_message=row.error_message,
            chunks=row.chunks,
            chars=row.chars,
            created_at=row.created_at,
            indexed_at=row.indexed_at,
        )
