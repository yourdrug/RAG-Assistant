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


def create_user(db: Session, email: str, hashed_password: str, role: str = "user") -> dict:
    result = db.execute(
        text("""
            INSERT INTO users (email, hashed_password, role)
            VALUES (:email, :hashed_password, :role)
            RETURNING id, email, hashed_password, role, is_active, created_at
        """),
        {"email": email.lower(), "hashed_password": hashed_password, "role": role},
    )
    db.commit()
    return _row_to_user(result.fetchone())


def list_users(db: Session) -> list[dict]:
    rows = db.execute(
        text("SELECT id, email, role, is_active, created_at FROM users ORDER BY created_at")
    ).fetchall()
    return [
        {
            "id": r.id,
            "email": r.email,
            "role": r.role,
            "is_active": r.is_active,
            "created_at": r.created_at,
        }
        for r in rows
    ]


def set_user_active(db: Session, user_id: int, is_active: bool) -> bool:
    result = db.execute(
        text("UPDATE users SET is_active = :active WHERE id = :id"),
        {"active": is_active, "id": user_id},
    )
    db.commit()
    return result.rowcount > 0
