"""
main.py — FastAPI приложение с SSE-стримингом и историей диалогов.

Эндпоинты:
  POST /chat                — стриминговый ответ (SSE)
  POST /chat/sync           — синхронный ответ (для тестов)
  POST /conversations       — создать новый диалог
  GET  /conversations/{id}  — получить историю диалога
  POST /ingest              — запустить индексацию документов
  GET  /health              — проверка сервисов
"""

import json

from config import settings
from database import (
    create_conversation,
    get_db,
    get_history,
    get_or_create_conversation,
    save_message,
)
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from rag_chain import rag_invoke, rag_stream
from sqlalchemy.orm import Session

app = FastAPI(
    title="RAG API",
    description="Корпоративный ассистент на основе RAG",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ограничь в продакшне
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic схемы
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str
    conversation_id: int | None = None
    user_id: str = "default"


class ChatResponse(BaseModel):
    answer: str
    conversation_id: int
    sources: list


class NewConversationResponse(BaseModel):
    conversation_id: int


# ---------------------------------------------------------------------------
# Роуты
# ---------------------------------------------------------------------------

@app.post("/conversations", response_model=NewConversationResponse)
def new_conversation(user_id: str = "default", db: Session = Depends(get_db)):
    """Создать новый диалог и получить его ID."""
    conv_id = create_conversation(db, user_id)
    return {"conversation_id": conv_id}


@app.get("/conversations/{conversation_id}")
def get_conversation_history(conversation_id: int, db: Session = Depends(get_db)):
    """Получить историю сообщений диалога."""
    history = get_history(db, conversation_id, window=100)
    return {"conversation_id": conversation_id, "messages": history}


@app.post("/chat")
async def chat_stream(req: ChatRequest, db: Session = Depends(get_db)):
    """
    Стриминговый чат (SSE).
    Клиент читает event-stream, каждый chunk — кусок ответа.
    Последнее событие типа 'done' содержит conversation_id и sources.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Вопрос не может быть пустым")

    # Получаем или создаём диалог
    conv_id = get_or_create_conversation(db, req.conversation_id, req.user_id)

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
async def chat_sync(req: ChatRequest, db: Session = Depends(get_db)):
    """
    Синхронный чат — ждёт полного ответа.
    Удобно для тестирования через curl / Swagger UI.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Вопрос не может быть пустым")

    conv_id = get_or_create_conversation(db, req.conversation_id, req.user_id)
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
    docs_dir: str = "/code/project/docs",
    reset: bool = False,
):
    """
    Индексирует папку с документами в фоне.

    - **reset=false** (по умолчанию) — добавляет только новые/изменённые файлы,
      уже загруженные пропускаются.
    - **reset=true** — удаляет коллекцию и реестр, переиндексирует всё с нуля.

    Прогресс: `docker logs -f rag_api`
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
):
    """
    Добавляет один файл не затрагивая остальные документы.

    - **file_path** — путь внутри контейнера: `/app/docs/report.pdf`
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
async def get_ingest_registry():
    """Список всех проиндексированных файлов с метаданными."""
    from ingestion import load_registry
    registry = load_registry()
    items = [
        {
            "filename":   name,
            "chunks":     meta.get("chunks", 0),
            "chars":      meta.get("chars", 0),
            "indexed_at": meta.get("indexed_at", ""),
            "source":     meta.get("source", ""),
        }
        for name, meta in sorted(registry.items())
    ]
    return {
        "total_files":  len(items),
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
