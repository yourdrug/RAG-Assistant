"""
api/routes.py — FastAPI route handlers.
"""

import json
import logging

from config import settings
from fastapi import BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from infrastructure.auth import (
    create_access_token,
    get_current_user,
    hash_password,
    require_admin,
    verify_password,
)
from infrastructure.database import (
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
from infrastructure.vector_store import (
    load_registry,
    rag_invoke,
    rag_stream,
    run_ingest_file,
    run_ingestion,
    save_registry,
)
from sqlalchemy.orm import Session

from api.schemas import (
    ChatRequest,
    ChatResponse,
    CreateUserRequest,
    LoginRequest,
    NewConversationResponse,
    TokenResponse,
    UploadResponse,
    UserResponse,
)

logger = logging.getLogger("default")


class Application:
    def __init__(self) -> None:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware

        self.app: FastAPI = FastAPI(
            title="RAG API",
            description="Корпоративный ассистент на основе RAG",
            version="0.1.0",
        )

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.allowed_origins_list,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self.add_routers()

    def add_routers(self) -> None:
        self.app.add_api_route("/auth/login", self.login, methods=["POST"], response_model=TokenResponse)
        self.app.add_api_route("/auth/me", self.get_me, methods=["GET"], response_model=UserResponse)
        self.app.add_api_route("/auth/users", self.add_user, methods=["POST"], response_model=UserResponse)
        self.app.add_api_route(
            "/auth/users", self.list_all_users, methods=["GET"], response_model=list[UserResponse]
        )
        self.app.add_api_route("/auth/users/{user_id}", self.toggle_user_active, methods=["PATCH"])

        self.app.add_api_route(
            "/conversations", self.new_conversation, methods=["POST"], response_model=NewConversationResponse
        )
        self.app.add_api_route(
            "/conversations/{conversation_id}", self.get_conversation_history, methods=["GET"]
        )

        self.app.add_api_route("/chat", self.chat_stream, methods=["POST"])
        self.app.add_api_route("/chat/sync", self.chat_sync, methods=["POST"], response_model=ChatResponse)

        self.app.add_api_route("/ingest", self.ingest_documents, methods=["POST"])
        self.app.add_api_route("/ingest/file", self.ingest_single_file, methods=["POST"])
        self.app.add_api_route("/ingest/registry", self.get_ingest_registry, methods=["GET"])

        self.app.add_api_route("/upload", self.upload_files, methods=["POST"], response_model=UploadResponse)

        self.app.add_api_route("/health", self.health, methods=["GET"])

    # -----------------------------------------------------------------------
    # Auth
    # -----------------------------------------------------------------------

    async def login(self, req: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
        user = get_user_by_email(db, req.email)
        if (
            user is None
            or not user["is_active"]
            or not verify_password(req.password, user["hashed_password"])
        ):
            raise HTTPException(status_code=401, detail="Неверный email или пароль")

        token = create_access_token(user_id=user["id"], role=user["role"])
        return TokenResponse(access_token=token, role=user["role"])

    async def get_me(self, current_user: dict = Depends(get_current_user)) -> UserResponse:
        return current_user

    async def add_user(
        self,
        req: CreateUserRequest,
        admin: dict = Depends(require_admin),
        db: Session = Depends(get_db),
    ) -> UserResponse:
        if req.role not in ("admin", "user"):
            raise HTTPException(status_code=400, detail="role должен быть 'admin' или 'user'")
        if get_user_by_email(db, req.email) is not None:
            raise HTTPException(status_code=409, detail="Пользователь с таким email уже существует")

        user = create_user(db, email=req.email, hashed_password=hash_password(req.password), role=req.role)
        return user

    async def list_all_users(
        self,
        admin: dict = Depends(require_admin),
        db: Session = Depends(get_db),
    ) -> list[UserResponse]:
        return list_users(db)

    async def toggle_user_active(
        self,
        user_id: int,
        is_active: bool,
        admin: dict = Depends(require_admin),
        db: Session = Depends(get_db),
    ) -> dict:
        if user_id == admin["id"] and not is_active:
            raise HTTPException(status_code=400, detail="Нельзя деактивировать самого себя")
        if not set_user_active(db, user_id, is_active):
            raise HTTPException(status_code=404, detail="Пользователь не найден")
        return {"id": user_id, "is_active": is_active}

    # -----------------------------------------------------------------------
    # Conversations
    # -----------------------------------------------------------------------

    async def new_conversation(
        self,
        current_user: dict = Depends(get_current_user),
        db: Session = Depends(get_db),
    ) -> NewConversationResponse:
        conv_id = create_conversation(db, current_user["id"])
        return {"conversation_id": conv_id}

    async def get_conversation_history(
        self,
        conversation_id: int,
        current_user: dict = Depends(get_current_user),
        db: Session = Depends(get_db),
    ) -> dict:
        owner_id = get_conversation_owner(db, conversation_id)
        if owner_id is None:
            raise HTTPException(status_code=404, detail="Диалог не найден")
        if owner_id != current_user["id"] and current_user["role"] != "admin":
            raise HTTPException(status_code=403, detail="Это не твой диалог")

        history = get_history(db, conversation_id, window=100)
        return {"conversation_id": conversation_id, "messages": history}

    # -----------------------------------------------------------------------
    # Chat
    # -----------------------------------------------------------------------

    async def chat_stream(
        self,
        req: ChatRequest,
        current_user: dict = Depends(get_current_user),
        db: Session = Depends(get_db),
    ):
        if not req.question.strip():
            raise HTTPException(status_code=400, detail="Вопрос не может быть пустым")

        conv_id = get_or_create_conversation(db, req.conversation_id, current_user["id"])
        save_message(db, conv_id, "user", req.question)

        history = get_history(db, conv_id, window=settings.history_window)
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
                        yield f"data: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"

                save_message(db, conv_id, "assistant", full_answer, sources=sources)
                yield f"event: done\ndata: {json.dumps({'conversation_id': conv_id, 'sources': sources}, ensure_ascii=False)}\n\n"

            except Exception as e:
                yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def chat_sync(
        self,
        req: ChatRequest,
        current_user: dict = Depends(get_current_user),
        db: Session = Depends(get_db),
    ) -> ChatResponse:
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

    # -----------------------------------------------------------------------
    # Ingestion
    # -----------------------------------------------------------------------

    async def ingest_documents(
        self,
        background_tasks: BackgroundTasks,
        docs_dir: str = f"{settings.data_dir}/docs_sample",
        reset: bool = False,
        admin: dict = Depends(require_admin),
    ):
        background_tasks.add_task(run_ingestion, docs_dir, reset)
        mode = "RESET + full reindex" if reset else "APPEND (new files only)"
        return {"status": "started", "mode": mode, "docs_dir": docs_dir}

    async def ingest_single_file(
        self,
        background_tasks: BackgroundTasks,
        file_path: str,
        force: bool = False,
        admin: dict = Depends(require_admin),
    ):
        from pathlib import Path as P

        if force:
            reg = load_registry()
            reg.pop(P(file_path).name, None)
            save_registry(reg)
        background_tasks.add_task(run_ingest_file, file_path)
        return {"status": "started", "file": file_path, "force": force}

    async def get_ingest_registry(self, admin: dict = Depends(require_admin)):
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

    async def upload_files(
        self,
        files: list[UploadFile] = File(...),
        admin: dict = Depends(require_admin),
    ) -> UploadResponse:
        from infrastructure.storage import get_storage

        storage = get_storage()
        prefix = "docs/"
        uploaded: list[str] = []
        for f in files:
            key = prefix + f.filename
            data = await f.read()
            storage.upload_file(key, data)
            uploaded.append(key)
            logger.info("Uploaded: %s (%d bytes)", key, len(data))
        return UploadResponse(files=uploaded)

    # -----------------------------------------------------------------------
    # Health
    # -----------------------------------------------------------------------

    async def health(self) -> dict:
        import httpx
        from qdrant_client import QdrantClient

        status = {"api": "ok", "qdrant": "unknown", "ollama": "unknown"}

        try:
            client = QdrantClient(url=settings.qdrant_url, timeout=3)
            client.get_collections()
            status["qdrant"] = "ok"
        except Exception as e:
            status["qdrant"] = f"error: {e}"

        try:
            async with httpx.AsyncClient(timeout=3) as client:
                r = await client.get(f"{settings.ollama_base_url}/api/tags")
                models = [m["name"] for m in r.json().get("models", [])]
                status["ollama"] = "ok"
                status["ollama_models"] = models
        except Exception as e:
            status["ollama"] = f"error: {e}"

        return status
