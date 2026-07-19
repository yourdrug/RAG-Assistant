"""
infrastructure/document_service.py — приём документов через POST /documents
(любой пользователь, не только admin), их сохранение через FileStorage
(local или S3) и фоновая индексация в Qdrant с ACL-payload.

Отличия от admin bulk-эндпоинтов /ingest*:
  - /ingest* работает с папкой/файлом на сервере, только admin, всегда visibility=internal_public
  - /documents принимает multipart upload от ЛЮБОГО пользователя, visibility зависит
    от kind пользователя, ключи лежат под отдельным префиксом settings.uploads_prefix
"""

from pathlib import Path

from config import settings
from domain.ingestion import PARSERS, parse_pdf, split_documents
from fastapi import HTTPException, UploadFile
from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from infrastructure.acl import ensure_acl_payload_indexes
from infrastructure.database import SessionLocal, delete_document_row, update_document_status
from infrastructure.storage import get_storage
from infrastructure.vector_store import ensure_collection, get_embeddings, upload_to_qdrant


def _storage_key_for(owner_id: int | None, group_id: int | None, document_id: int, filename: str) -> str:
    safe_name = Path(filename).name
    prefix = settings.uploads_prefix.rstrip("/")
    if owner_id is not None:
        return f"{prefix}/users/{owner_id}/{document_id}_{safe_name}"
    if group_id is not None:
        return f"{prefix}/groups/{group_id}/{document_id}_{safe_name}"
    return f"{prefix}/public/{document_id}_{safe_name}"


async def save_upload(file: UploadFile, document_id: int, owner_id: int | None, group_id: int | None) -> str:
    """Читает файл целиком и сохраняет через FileStorage. Возвращает storage key."""
    ext = Path(file.filename).suffix.lower()
    if ext not in settings.supported_extensions:
        raise HTTPException(400, f"Неподдерживаемый формат файла: {ext}")

    key = _storage_key_for(owner_id, group_id, document_id, file.filename)
    data = await file.read()
    get_storage().upload_file(key, data)
    return key


def _parse_uploaded_file(local_path: Path, original_filename: str) -> list[Document]:
    ext = local_path.suffix.lower()

    if ext == ".pdf":
        docs = parse_pdf(local_path)
    else:
        parser = PARSERS.get(ext)
        if parser is None:
            raise RuntimeError(f"Неподдерживаемый формат: {ext}")
        text = parser(local_path)
        if not text or len(text.strip()) < 20:
            raise RuntimeError("Слишком мало текста в документе")
        docs = [Document(page_content=text, metadata={})]

    for doc in docs:
        doc.metadata["source"] = original_filename
    return docs


def process_upload(
    document_id: int,
    storage_key: str,
    original_filename: str,
    visibility: str,
    owner_id: int | None,
    group_id: int | None,
    replace_id: int | None,
) -> None:
    """Вызывается через BackgroundTasks из POST /documents."""
    db = SessionLocal()
    storage = get_storage()
    temp_path: Path | None = None
    try:
        update_document_status(db, document_id, "processing")

        temp_path = storage.download_to_temp(storage_key)
        docs = _parse_uploaded_file(temp_path, original_filename)
        if not docs:
            update_document_status(db, document_id, "failed", error="Не удалось извлечь текст из файла")
            return

        chunks = split_documents(docs)
        for chunk in chunks:
            chunk.metadata.update(
                {
                    "document_id": document_id,
                    "visibility": visibility,
                    "owner_id": owner_id,
                    "group_id": group_id,
                }
            )

        embeddings = get_embeddings()
        vector_size = len(embeddings.embed_query("тест"))
        client = QdrantClient(url=settings.qdrant_url)
        ensure_collection(client, vector_size, reset=False)
        ensure_acl_payload_indexes(client)

        upload_to_qdrant(chunks, embeddings)

        if replace_id is not None:
            client.delete(
                collection_name=settings.collection_name,
                points_selector=Filter(
                    must=[FieldCondition(key="document_id", match=MatchValue(value=replace_id))]
                ),
            )
            old = None
            try:
                from infrastructure.database import get_document

                old = get_document(db, replace_id)
            except Exception:
                pass
            if old and old.get("source_path"):
                storage.delete_file(old["source_path"])
            delete_document_row(db, replace_id)

        total_chars = sum(len(d.page_content) for d in docs)
        update_document_status(db, document_id, "done", chunks=len(chunks), chars=total_chars)

    except Exception as e:
        update_document_status(db, document_id, "failed", error=str(e))
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        db.close()
