"""
main.py — FastAPI приложение с SSE-стримингом, историей диалогов и авторизацией.

Роли:
  admin — POST /ingest*, POST /auth/users (заводит новых пользователей), видит /auth/users
  user  — только /chat, /chat/sync, /conversations (свои)

Эндпоинты:
  POST /auth/login          — логин, выдаёт JWT
  GET  /auth/me              — текущий пользователь
  POST /auth/users            — [admin] завести нового пользователя
  GET  /auth/users             — [admin] список пользователей
  PATCH /auth/users/{id}        — [admin] активировать/деактивировать пользователя
  POST /chat                — стриминговый ответ (SSE) [user]
  POST /chat/sync           — синхронный ответ (для тестов) [user]
  POST /conversations       — создать новый диалог [user]
  GET  /conversations/{id}  — получить историю диалога (только свой) [user]
  POST /ingest               — запустить индексацию документов [admin]
  GET  /health               — проверка сервисов (без авторизации, для healthcheck)
"""

import json
from contextlib import asynccontextmanager

from auth import create_access_token, get_current_user, hash_password, require_admin, verify_password
from config import settings
from database import (
    SessionLocal,
    any_admin_exists,
    create_conversation,
    create_user,
    get_conversation_owner,
    get_db,
    get_history,
    get_or_create_conversation,
    get_user_by_email,
    list_users,
    save_message,
    set_user_active,
)
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from rag_chain import rag_invoke, rag_stream
from sqlalchemy.orm import Session


def bootstrap_admin():
    """
    Если в БД ещё нет ни одного admin, а в окружении заданы ADMIN_EMAIL/ADMIN_PASSWORD —
    создаёт первого admin-пользователя. Так self-host инсталляция получает рабочий
    аккаунт сразу после первого запуска, без ручных SQL-вставок.
    """
    db = SessionLocal()
    try:
        if any_admin_exists(db):
            return
        if not settings.admin_email or not settings.admin_password:
            print(
                "[bootstrap_admin] Нет ни одного admin, а ADMIN_EMAIL/ADMIN_PASSWORD не заданы — "
                "залогиниться будет некому. Задай их в server/.env и перезапусти."
            )
            return
        create_user(
            db,
            email=settings.admin_email,
            hashed_password=hash_password(settings.admin_password),
            role="admin",
        )
        print(f"[bootstrap_admin] Создан admin: {settings.admin_email}")
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    bootstrap_admin()
    yield


app = FastAPI(
    title="RAG API",
    description="Корпоративный ассистент на основе RAG",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic схемы
# ---------------------------------------------------------------------------


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str


class UserResponse(BaseModel):
    id: int
    email: str
    role: str
    is_active: bool


class CreateUserRequest(BaseModel):
    email: str
    password: str
    role: str = "user"  # "user" | "admin"


class ChatRequest(BaseModel):
    question: str
    conversation_id: int | None = None


class ChatResponse(BaseModel):
    answer: str
    conversation_id: int
    sources: list


class NewConversationResponse(BaseModel):
    conversation_id: int


# ---------------------------------------------------------------------------
# Роуты: авторизация
# ---------------------------------------------------------------------------


@app.post("/auth/login", response_model=TokenResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    """Логин по email/паролю, выдаёт JWT на settings.jwt_expire_minutes."""
    user = get_user_by_email(db, req.email)
    if user is None or not user["is_active"] or not verify_password(req.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Неверный email или пароль")

    token = create_access_token(user_id=user["id"], role=user["role"])
    return TokenResponse(access_token=token, role=user["role"])


@app.get("/auth/me", response_model=UserResponse)
def get_me(current_user: dict = Depends(get_current_user)):
    return current_user


@app.post("/auth/users", response_model=UserResponse)
def add_user(req: CreateUserRequest, admin: dict = Depends(require_admin), db: Session = Depends(get_db)):
    """[admin] Завести нового пользователя (org-фича: приглашение сотрудников)."""
    if req.role not in ("admin", "user"):
        raise HTTPException(status_code=400, detail="role должен быть 'admin' или 'user'")
    if get_user_by_email(db, req.email) is not None:
        raise HTTPException(status_code=409, detail="Пользователь с таким email уже существует")

    user = create_user(db, email=req.email, hashed_password=hash_password(req.password), role=req.role)
    return user


@app.get("/auth/users", response_model=list[UserResponse])
def list_all_users(admin: dict = Depends(require_admin), db: Session = Depends(get_db)):
    """[admin] Список всех пользователей организации."""
    return list_users(db)


@app.patch("/auth/users/{user_id}")
def toggle_user_active(
    user_id: int,
    is_active: bool,
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """[admin] Активировать/деактивировать пользователя (не удаляя историю его диалогов)."""
    if user_id == admin["id"] and not is_active:
        raise HTTPException(status_code=400, detail="Нельзя деактивировать самого себя")
    if not set_user_active(db, user_id, is_active):
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    return {"id": user_id, "is_active": is_active}


# ---------------------------------------------------------------------------
# Роуты
# ---------------------------------------------------------------------------


@app.post("/conversations", response_model=NewConversationResponse)
def new_conversation(current_user: dict = Depends(get_current_user), db: Session = Depends(get_db)):
    """Создать новый диалог и получить его ID."""
    conv_id = create_conversation(db, current_user["id"])
    return {"conversation_id": conv_id}


@app.get("/conversations/{conversation_id}")
def get_conversation_history(
    conversation_id: int,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Получить историю сообщений диалога — только если он принадлежит текущему пользователю."""
    owner_id = get_conversation_owner(db, conversation_id)
    if owner_id is None:
        raise HTTPException(status_code=404, detail="Диалог не найден")
    if owner_id != current_user["id"] and current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Это не твой диалог")

    history = get_history(db, conversation_id, window=100)
    return {"conversation_id": conversation_id, "messages": history}


@app.post("/chat")
async def chat_stream(
    req: ChatRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Стриминговый чат (SSE).
    Клиент читает event-stream, каждый chunk — кусок ответа.
    Последнее событие типа 'done' содержит conversation_id и sources.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Вопрос не может быть пустым")

    # Получаем или создаём диалог (get_or_create_conversation сама проверяет владение)
    conv_id = get_or_create_conversation(db, req.conversation_id, current_user["id"])

    # Сохраняем вопрос пользователя
    save_message(db, conv_id, "user", req.question)

    # Получаем историю (без только что добавленного вопроса)
    history = get_history(db, conv_id, window=settings.history_window)
    # Убираем последний вопрос из истории (он уже в question)
    if history and history[-1]["role"] == "user":
        history = history[:-1]

    async def event_generator():
        full_answer = ""
        sources = []

        try:
            async for chunk in rag_stream(req.question, history):
                if chunk.startswith("\n__sources__:"):
                    sources = json.loads(chunk.replace("\n__sources__:", ""))
                else:
                    full_answer += chunk
                    # SSE формат
                    yield f"data: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"

            # Сохраняем ответ ассистента в БД
            save_message(db, conv_id, "assistant", full_answer, sources=sources)

            # Финальное событие с метаданными
            yield f"event: done\ndata: {json.dumps({'conversation_id': conv_id, 'sources': sources}, ensure_ascii=False)}\n\n"

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/chat/sync", response_model=ChatResponse)
async def chat_sync(
    req: ChatRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Синхронный чат — ждёт полного ответа.
    Удобно для тестирования через curl / Swagger UI.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Вопрос не может быть пустым")

    conv_id = get_or_create_conversation(db, req.conversation_id, current_user["id"])
    save_message(db, conv_id, "user", req.question)

    history = get_history(db, conv_id, window=settings.history_window)
    if history and history[-1]["role"] == "user":
        history = history[:-1]

    answer, sources = await rag_invoke(req.question, history)
    save_message(db, conv_id, "assistant", answer, sources=sources)

    return ChatResponse(answer=answer, conversation_id=conv_id, sources=sources)


@app.post("/ingest")
async def ingest_documents(
    background_tasks: BackgroundTasks,
    docs_dir: str = f"{settings.data_dir}/docs_sample",
    reset: bool = False,
    admin: dict = Depends(require_admin),
):
    """
    [admin] Индексирует папку с документами в фоне.

    - **reset=false** (по умолчанию) — добавляет только новые/изменённые файлы,
      уже загруженные пропускаются.
    - **reset=true** — удаляет коллекцию и реестр, переиндексирует всё с нуля.

    Прогресс: `docker compose logs -f server`
    """
    from ingestion import run_ingestion

    background_tasks.add_task(run_ingestion, docs_dir, reset)
    mode = "RESET + full reindex" if reset else "APPEND (new files only)"
    return {"status": "started", "mode": mode, "docs_dir": docs_dir}


@app.post("/ingest/file")
async def ingest_single_file(
    background_tasks: BackgroundTasks,
    file_path: str,
    force: bool = False,
    admin: dict = Depends(require_admin),
):
    """
    [admin] Добавляет один файл не затрагивая остальные документы.

    - **file_path** — путь внутри контейнера: `/code/project/data/docs_sample/report.pdf`
    - **force=true** — переиндексирует даже если файл уже в реестре
    """
    from pathlib import Path as P

    from ingestion import load_registry, run_ingest_file, save_registry

    if force:
        reg = load_registry()
        reg.pop(P(file_path).name, None)
        save_registry(reg)
    background_tasks.add_task(run_ingest_file, file_path)
    return {"status": "started", "file": file_path, "force": force}


@app.get("/ingest/registry")
async def get_ingest_registry(admin: dict = Depends(require_admin)):
    """[admin] Список всех проиндексированных файлов с метаданными."""
    from ingestion import load_registry

    registry = load_registry()
    items = [
        {
            "filename": name,
            "chunks": meta.get("chunks", 0),
            "chars": meta.get("chars", 0),
            "indexed_at": meta.get("indexed_at", ""),
            "source": meta.get("source", ""),
        }
        for name, meta in sorted(registry.items())
    ]
    return {
        "total_files": len(items),
        "total_chunks": sum(i["chunks"] for i in items),
        "files": items,
    }


@app.get("/health")
async def health():
    """Проверяет доступность всех сервисов."""
    import httpx
    from qdrant_client import QdrantClient

    status = {"api": "ok", "qdrant": "unknown", "ollama": "unknown"}

    # Qdrant
    try:
        client = QdrantClient(url=settings.qdrant_url, timeout=3)
        client.get_collections()
        status["qdrant"] = "ok"
    except Exception as e:
        status["qdrant"] = f"error: {e}"

    # Ollama
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{settings.ollama_base_url}/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
            status["ollama"] = "ok"
            status["ollama_models"] = models
    except Exception as e:
        status["ollama"] = f"error: {e}"

    return status
