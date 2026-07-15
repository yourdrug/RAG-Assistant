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


def create_conversation(db: Session, user_id: str = "default") -> int:
    result = db.execute(
        text("INSERT INTO conversations (user_id) VALUES (:uid) RETURNING id"),
        {"uid": user_id},
    )
    db.commit()
    return result.scalar()


def get_or_create_conversation(db: Session, conversation_id: int | None, user_id: str = "default") -> int:
    if conversation_id:
        # проверяем что такой диалог существует
        row = db.execute(
            text("SELECT id FROM conversations WHERE id = :id"),
            {"id": conversation_id},
        ).fetchone()
        if row:
            return conversation_id
    return create_conversation(db, user_id)


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
