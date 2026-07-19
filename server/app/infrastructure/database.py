import json

from config import settings
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)


def get_db():
    """FastAPI dependency — yields DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_conversation(db: Session, user_id: int) -> int:
    result = db.execute(
        text("INSERT INTO conversations (user_id) VALUES (:uid) RETURNING id"),
        {"uid": user_id},
    )
    db.commit()
    return result.scalar()


def get_or_create_conversation(db: Session, conversation_id: int | None, user_id: int) -> int:
    if conversation_id:
        # проверяем что такой диалог существует И принадлежит именно этому пользователю —
        # иначе можно было бы подставить чужой conversation_id и читать/дописывать чужой диалог
        row = db.execute(
            text("SELECT id FROM conversations WHERE id = :id AND user_id = :uid"),
            {"id": conversation_id, "uid": user_id},
        ).fetchone()
        if row:
            return conversation_id
    return create_conversation(db, user_id)


def get_conversation_owner(db: Session, conversation_id: int) -> int | None:
    row = db.execute(
        text("SELECT user_id FROM conversations WHERE id = :id"),
        {"id": conversation_id},
    ).fetchone()
    return row.user_id if row else None


def save_message(
    db: Session,
    conversation_id: int,
    role: str,
    content: str,
    sources: list[dict] | None = None,
):
    db.execute(
        text("""
            INSERT INTO messages (conversation_id, role, content, sources)
            VALUES (:conv_id, :role, :content, :sources)
        """),
        {
            "conv_id": conversation_id,
            "role": role,
            "content": content,
            "sources": json.dumps(sources) if sources else None,
        },
    )
    db.commit()


def get_history(db: Session, conversation_id: int, window: int = 8) -> list[dict]:
    """Возвращает последние N сообщений в формате LangChain [{role, content}]."""
    rows = db.execute(
        text("""
            SELECT role, content FROM messages
            WHERE conversation_id = :conv_id
            ORDER BY created_at DESC
            LIMIT :lim
        """),
        {"conv_id": conversation_id, "lim": window},
    ).fetchall()

    # Разворачиваем — fetchall даёт от новых к старым
    return [{"role": r.role, "content": r.content} for r in reversed(rows)]


# ---------------------------------------------------------------------------
# Пользователи
# ---------------------------------------------------------------------------


def _row_to_user(row) -> dict | None:
    if row is None:
        return None
    return {
        "id": row.id,
        "email": row.email,
        "hashed_password": row.hashed_password,
        "role": row.role,
        "kind": row.kind,
        "is_active": row.is_active,
        "created_at": row.created_at,
    }


def get_user_by_email(db: Session, email: str) -> dict | None:
    row = db.execute(
        text("SELECT * FROM users WHERE email = :email"),
        {"email": email.lower()},
    ).fetchone()
    return _row_to_user(row)


def get_user_by_id(db: Session, user_id: int) -> dict | None:
    row = db.execute(
        text("SELECT * FROM users WHERE id = :id"),
        {"id": user_id},
    ).fetchone()
    return _row_to_user(row)


def any_admin_exists(db: Session) -> bool:
    row = db.execute(text("SELECT 1 FROM users WHERE role = 'admin' LIMIT 1")).fetchone()
    return row is not None


def create_user(
    db: Session,
    email: str,
    hashed_password: str,
    role: str = "user",
    kind: str = "internal",
) -> dict:
    if kind == "client" and role == "admin":
        raise ValueError("Клиент не может быть admin")

    result = db.execute(
        text("""
            INSERT INTO users (email, hashed_password, role, kind)
            VALUES (:email, :hashed_password, :role, :kind)
            RETURNING id, email, hashed_password, role, kind, is_active, created_at
        """),
        {"email": email.lower(), "hashed_password": hashed_password, "role": role, "kind": kind},
    )
    db.commit()
    return _row_to_user(result.fetchone())


def list_users(db: Session) -> list[dict]:
    rows = db.execute(
        text("SELECT id, email, role, kind, is_active, created_at FROM users ORDER BY created_at")
    ).fetchall()
    return [
        {
            "id": r.id,
            "email": r.email,
            "role": r.role,
            "kind": r.kind,
            "is_active": r.is_active,
            "created_at": r.created_at,
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Группы (плоские отделы, без иерархии)
# ---------------------------------------------------------------------------


def create_group(db: Session, name: str) -> int:
    result = db.execute(text("INSERT INTO groups (name) VALUES (:name) RETURNING id"), {"name": name})
    db.commit()
    return result.scalar()


def list_groups(db: Session, only_ids: list[int] | None = None) -> list[dict]:
    if only_ids is not None:
        if not only_ids:
            return []
        rows = db.execute(
            text("SELECT id, name FROM groups WHERE id = ANY(:ids) ORDER BY name"), {"ids": only_ids}
        ).fetchall()
    else:
        rows = db.execute(text("SELECT id, name FROM groups ORDER BY name")).fetchall()
    return [{"id": r.id, "name": r.name} for r in rows]


def add_user_to_group(db: Session, user_id: int, group_id: int) -> None:
    db.execute(
        text("INSERT INTO user_groups (user_id, group_id) VALUES (:uid, :gid) ON CONFLICT DO NOTHING"),
        {"uid": user_id, "gid": group_id},
    )
    db.commit()


def remove_user_from_group(db: Session, user_id: int, group_id: int) -> None:
    db.execute(
        text("DELETE FROM user_groups WHERE user_id = :uid AND group_id = :gid"),
        {"uid": user_id, "gid": group_id},
    )
    db.commit()


def get_user_group_ids(db: Session, user_id: int) -> list[int]:
    rows = db.execute(
        text("SELECT group_id FROM user_groups WHERE user_id = :uid"), {"uid": user_id}
    ).fetchall()
    return [r.group_id for r in rows]


def list_group_members(db: Session, group_id: int) -> list[dict]:
    rows = db.execute(
        text("""
            SELECT u.id, u.email FROM users u
            JOIN user_groups ug ON ug.user_id = u.id
            WHERE ug.group_id = :gid
            ORDER BY u.email
        """),
        {"gid": group_id},
    ).fetchall()
    return [{"id": r.id, "email": r.email} for r in rows]


# ---------------------------------------------------------------------------
# Назначения: internal-сотрудник <-> client
# ---------------------------------------------------------------------------


def assign_client(db: Session, internal_user_id: int, client_user_id: int, assigned_by: int) -> None:
    db.execute(
        text("""
            INSERT INTO client_assignments (internal_user_id, client_user_id, assigned_by)
            VALUES (:iu, :cu, :by)
            ON CONFLICT DO NOTHING
        """),
        {"iu": internal_user_id, "cu": client_user_id, "by": assigned_by},
    )
    db.commit()


def unassign_client(db: Session, internal_user_id: int, client_user_id: int) -> None:
    db.execute(
        text("DELETE FROM client_assignments WHERE internal_user_id = :iu AND client_user_id = :cu"),
        {"iu": internal_user_id, "cu": client_user_id},
    )
    db.commit()


def get_assigned_client_ids(db: Session, internal_user_id: int) -> list[int]:
    rows = db.execute(
        text("SELECT client_user_id FROM client_assignments WHERE internal_user_id = :iu"),
        {"iu": internal_user_id},
    ).fetchall()
    return [r.client_user_id for r in rows]


def list_assignments_for_client(db: Session, client_user_id: int) -> list[dict]:
    rows = db.execute(
        text("""
            SELECT u.id AS internal_user_id, u.email, ca.assigned_at
            FROM client_assignments ca
            JOIN users u ON u.id = ca.internal_user_id
            WHERE ca.client_user_id = :cu
            ORDER BY ca.assigned_at
        """),
        {"cu": client_user_id},
    ).fetchall()
    return [
        {"internal_user_id": r.internal_user_id, "email": r.email, "assigned_at": r.assigned_at} for r in rows
    ]


# ---------------------------------------------------------------------------
# Документы (пользовательские загрузки через POST /documents)
# ---------------------------------------------------------------------------


def _row_to_document(row) -> dict | None:
    if row is None:
        return None
    return {
        "id": row.id,
        "filename": row.filename,
        "source_path": row.source_path,
        "visibility": row.visibility,
        "owner_id": row.owner_id,
        "group_id": row.group_id,
        "status": row.status,
        "error_message": row.error_message,
        "chunks": row.chunks,
        "chars": row.chars,
        "created_at": row.created_at,
        "indexed_at": row.indexed_at,
    }


def create_document_row(
    db: Session,
    filename: str,
    visibility: str,
    owner_id: int | None,
    group_id: int | None,
    source_path: str = "",
) -> int:
    result = db.execute(
        text("""
            INSERT INTO documents (filename, source_path, visibility, owner_id, group_id, status)
            VALUES (:filename, :source_path, :visibility, :owner_id, :group_id, 'pending')
            RETURNING id
        """),
        {
            "filename": filename,
            "source_path": source_path,
            "visibility": visibility,
            "owner_id": owner_id,
            "group_id": group_id,
        },
    )
    db.commit()
    return result.scalar()


def set_document_source_path(db: Session, document_id: int, source_path: str) -> None:
    db.execute(
        text("UPDATE documents SET source_path = :path WHERE id = :id"),
        {"path": source_path, "id": document_id},
    )
    db.commit()


def find_active_slot(db: Session, owner_id: int | None, filename: str, group_id: int | None) -> dict | None:
    row = db.execute(
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
    return _row_to_document(row)


def get_document(db: Session, document_id: int) -> dict | None:
    row = db.execute(text("SELECT * FROM documents WHERE id = :id"), {"id": document_id}).fetchone()
    return _row_to_document(row)


def update_document_status(
    db: Session,
    document_id: int,
    status: str,
    error: str | None = None,
    chunks: int | None = None,
    chars: int | None = None,
) -> None:
    db.execute(
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
    db.commit()


def delete_document_row(db: Session, document_id: int) -> None:
    db.execute(text("DELETE FROM documents WHERE id = :id"), {"id": document_id})
    db.commit()


def list_documents_visible(db: Session, user: dict) -> list[dict]:
    if user["kind"] == "client":
        rows = db.execute(
            text("""
                SELECT * FROM documents
                WHERE visibility = 'client_private' AND owner_id = :uid
                ORDER BY created_at DESC
            """),
            {"uid": user["id"]},
        ).fetchall()
    else:
        group_ids = get_user_group_ids(db, user["id"]) or [-1]
        assigned_ids = get_assigned_client_ids(db, user["id"]) or [-1]
        rows = db.execute(
            text("""
                SELECT * FROM documents
                WHERE visibility = 'internal_public'
                   OR (visibility = 'internal_group' AND group_id = ANY(:group_ids))
                   OR (visibility = 'internal_private' AND owner_id = :uid)
                   OR (visibility = 'client_private' AND owner_id = ANY(:assigned_ids))
                ORDER BY created_at DESC
            """),
            {"group_ids": group_ids, "assigned_ids": assigned_ids, "uid": user["id"]},
        ).fetchall()
    return [_row_to_document(r) for r in rows]


def set_user_active(db: Session, user_id: int, is_active: bool) -> bool:
    result = db.execute(
        text("UPDATE users SET is_active = :active WHERE id = :id"),
        {"active": is_active, "id": user_id},
    )
    db.commit()
    return result.rowcount > 0
