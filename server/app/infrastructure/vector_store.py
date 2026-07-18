"""
infrastructure/vector_store.py — Qdrant, Ollama, embeddings singletons + ingestion orchestration.
"""

import json
import logging
import time
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path

from config import settings
from domain.ingestion import PARSERS, parse_pdf, split_documents
from domain.rag import (
    build_prompt,
    extract_sources,
    format_docs,
    history_to_messages,
    rerank_documents,
)
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import CrossEncoder

from infrastructure.storage import FileItem, get_storage

log = logging.getLogger("default")

Path(settings.data_dir).mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY_PATH = Path(settings.data_dir) / "ingestion_registry.json"


def load_registry() -> dict:
    if REGISTRY_PATH.exists():
        try:
            return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_registry(registry: dict):
    REGISTRY_PATH.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def file_hash(source) -> str:
    if isinstance(source, FileItem):
        return f"{source.size_bytes}_{source.last_modified}"
    stat = source.stat()
    return f"{stat.st_size}_{int(stat.st_mtime)}"


def is_already_indexed(source, registry: dict) -> bool:
    key = source.name if isinstance(source, FileItem) else source.name
    if key not in registry:
        return False
    return registry[key].get("hash") == file_hash(source)


def _base_metadata(source, temp_path: Path | None = None) -> dict:
    if isinstance(source, FileItem):
        return {
            "source": f"s3://{settings.s3_bucket}/{source.key}",
            "filename": source.filename,
            "extension": source.extension,
            "size_bytes": source.size_bytes,
        }
    return {
        "source": str(source),
        "filename": source.name,
        "extension": source.suffix.lower(),
        "size_bytes": source.stat().st_size,
    }


# ---------------------------------------------------------------------------
# Client singletons
# ---------------------------------------------------------------------------

_embeddings: HuggingFaceEmbeddings | None = None
_vector_store: QdrantVectorStore | None = None
_llm: ChatOllama | None = None
_reranker: CrossEncoder | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        log.info("Загружаю эмбеддинг-модель %s ...", settings.embed_model)
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.embed_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_vector_store() -> QdrantVectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = QdrantVectorStore.from_existing_collection(
            embedding=get_embeddings(),
            url=settings.qdrant_url,
            collection_name=settings.collection_name,
        )
    return _vector_store


def get_llm() -> ChatOllama:
    global _llm
    if _llm is None:
        _llm = ChatOllama(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            temperature=0.1,
        )
    return _llm


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        log.info("Загружаю реранкер %s ...", settings.rerank_model)
        _reranker = CrossEncoder(
            settings.rerank_model,
            max_length=1024,
            device=settings.rerank_device,
        )
        log.info("Реранкер загружен")
    return _reranker


# ---------------------------------------------------------------------------
# Qdrant helpers
# ---------------------------------------------------------------------------


def ensure_collection(client: QdrantClient, vector_size: int, reset: bool = False):
    existing = [c.name for c in client.get_collections().collections]
    if settings.collection_name in existing:
        if reset:
            log.info("Удаляю коллекцию '%s' ...", settings.collection_name)
            client.delete_collection(settings.collection_name)
        else:
            info = client.get_collection(settings.collection_name)
            count = info.points_count or 0
            log.info(
                "Коллекция '%s' существует — %d точек. Добавляю новые документы.",
                settings.collection_name,
                count,
            )
            return
    log.info("Создаю коллекцию '%s' (dim=%d) ...", settings.collection_name, vector_size)
    client.create_collection(
        collection_name=settings.collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def upload_to_qdrant(chunks: list[Document], embeddings: HuggingFaceEmbeddings):
    batch_size = 100
    total = len(chunks)
    log.info("Загружаю %d чанков в Qdrant батчами по %d ...", total, batch_size)
    t0 = time.monotonic()

    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        QdrantVectorStore.from_documents(
            documents=batch,
            embedding=embeddings,
            url=settings.qdrant_url,
            collection_name=settings.collection_name,
            force_recreate=False,
        )
        done = min(i + batch_size, total)
        elapsed = time.monotonic() - t0
        speed = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / speed if speed > 0 else 0
        log.info(
            "  Загружено %d/%d чанков  (%.1f ч/с, ETA ~%.0fs)",
            done,
            total,
            speed,
            eta,
        )

    log.info("Загрузка в Qdrant завершена за %.1fs", time.monotonic() - t0)


# ---------------------------------------------------------------------------
# Parse file
# ---------------------------------------------------------------------------


def parse_file(source, temp_path: Path | None = None) -> list[Document] | None:
    if isinstance(source, FileItem):
        path = temp_path or Path(source.filename)
        ext = source.extension
    else:
        path = source
        ext = path.suffix.lower()

    if ext == ".pdf":
        try:
            pages = parse_pdf(path)
            if not pages:
                return None
            base = _base_metadata(source, path)
            for doc in pages:
                doc.metadata.update(base)
            return pages
        except Exception as e:
            log.error("  ERROR %s: %s", source.filename if isinstance(source, FileItem) else path.name, e)
            return None

    parser = PARSERS.get(ext)
    fname = source.filename if isinstance(source, FileItem) else path.name
    if parser is None:
        log.debug("  SKIP  неподдерживаемый формат: %s", fname)
        return None
    try:
        text = parser(path)
        if not text or len(text.strip()) < 20:
            log.warning("  SKIP  слишком мало текста: %s", fname)
            return None
        return [
            Document(
                page_content=text,
                metadata=_base_metadata(source, path),
            )
        ]
    except Exception as e:
        log.error("  ERROR %s: %s", fname, e)
        return None


# ---------------------------------------------------------------------------
# Load documents
# ---------------------------------------------------------------------------


def load_documents(
    docs_dir: str,
    registry: dict,
    force: bool = False,
    prefix: str | None = None,
) -> tuple[list[Document], int]:
    storage = get_storage()

    if settings.file_backend == "s3":
        s3_prefix = prefix or "docs/"
        items = storage.list_files(s3_prefix)
        log.info("Найдено файлов: %d в s3://%s/%s", len(items), settings.s3_bucket, s3_prefix)
    else:
        docs_path = Path(docs_dir)
        if not docs_path.exists():
            log.error("Папка не найдена: %s", docs_dir)
            return [], 0
        items = None
        local_files = sorted(f for f in docs_path.rglob("*") if f.is_file() and f.suffix.lower() in PARSERS)
        log.info("Найдено файлов: %d в %s", len(local_files), docs_dir)

    documents, skipped_cached, ok, errors = [], 0, 0, 0

    if settings.file_backend == "s3":
        for i, file_item in enumerate(items, 1):
            tag = f"[{i:>3}/{len(items)}]"
            if not force and is_already_indexed(file_item, registry):
                log.info("%s CACHED  %s", tag, file_item.filename)
                skipped_cached += 1
                continue

            size_kb = file_item.size_bytes / 1024
            log.info("%s PARSE   %s  (%.1f KB)", tag, file_item.filename, size_kb)
            t0 = time.monotonic()

            temp_path = storage.download_to_temp(file_item.key)
            try:
                docs = parse_file(file_item, temp_path)
            finally:
                temp_path.unlink(missing_ok=True)

            elapsed = time.monotonic() - t0
            if docs:
                documents.extend(docs)
                total_chars = sum(len(d.page_content) for d in docs)
                log.info(
                    "%s OK      %s — %s символов, %d стр., %.2fs",
                    tag,
                    file_item.filename,
                    f"{total_chars:,}",
                    len(docs),
                    elapsed,
                )
                ok += 1
            else:
                errors += 1
    else:
        for i, file_path in enumerate(local_files, 1):
            tag = f"[{i:>3}/{len(local_files)}]"
            if not force and is_already_indexed(file_path, registry):
                log.info("%s CACHED  %s", tag, file_path.name)
                skipped_cached += 1
                continue

            size_kb = file_path.stat().st_size / 1024
            log.info("%s PARSE   %s  (%.1f KB)", tag, file_path.name, size_kb)
            t0 = time.monotonic()

            docs = parse_file(file_path)
            elapsed = time.monotonic() - t0

            if docs:
                documents.extend(docs)
                total_chars = sum(len(d.page_content) for d in docs)
                log.info(
                    "%s OK      %s — %s символов, %d стр., %.2fs",
                    tag,
                    file_path.name,
                    f"{total_chars:,}",
                    len(docs),
                    elapsed,
                )
                ok += 1
            else:
                errors += 1

    log.info(
        "Парсинг завершён: ✓ %d загружено, ✗ %d ошибок, ○ %d уже в реестре",
        ok,
        errors,
        skipped_cached,
    )
    return documents, skipped_cached


# ---------------------------------------------------------------------------
# Public ingestion functions
# ---------------------------------------------------------------------------


def run_ingestion(docs_dir: str, reset: bool = False, prefix: str | None = None):
    t_start = time.monotonic()
    log.info("=" * 55)
    log.info("RAG Ingestion  |  режим: %s", "RESET" if reset else "APPEND")
    log.info("backend  : %s", settings.file_backend)
    if settings.file_backend == "s3":
        log.info("prefix   : %s (bucket: %s)", prefix or "docs/", settings.s3_bucket)
    else:
        log.info("docs_dir : %s", docs_dir)
    log.info("model    : %s", settings.embed_model)
    log.info("qdrant   : %s  /  collection: %s", settings.qdrant_url, settings.collection_name)
    log.info("=" * 55)

    registry = load_registry()
    if reset:
        registry = {}

    embeddings = get_embeddings()
    vector_size = len(embeddings.embed_query("тест"))
    qdrant_client = QdrantClient(url=settings.qdrant_url)
    ensure_collection(qdrant_client, vector_size, reset=reset)

    docs, cached = load_documents(docs_dir, registry, force=reset, prefix=prefix)
    if not docs:
        if cached > 0:
            log.info(
                "Все файлы уже в реестре — нечего индексировать. Передай --reset чтобы переиндексировать."
            )
        else:
            log.error("Ни одного документа не загружено. Проверь папку и форматы.")
        return

    chunks = split_documents(docs)
    upload_to_qdrant(chunks, embeddings)

    storage = get_storage()
    source_chars: dict[str, int] = {}
    for doc in docs:
        src = doc.metadata["source"]
        source_chars[src] = source_chars.get(src, 0) + len(doc.page_content)
    for src, chars in source_chars.items():
        if src.startswith("s3://"):
            fname = Path(src).name
            file_info = None
            if settings.file_backend == "s3":
                key = "/".join(src.split("/")[3:])
                file_info = storage.get_file_info(key)
            if file_info:
                h = f"{file_info.size_bytes}_{file_info.last_modified}"
            else:
                h = "unknown"
        else:
            path = Path(src)
            fname = path.name
            h = file_hash(path)
        registry[fname] = {
            "hash": h,
            "source": src,
            "chunks": sum(1 for c in chunks if c.metadata.get("source") == src),
            "chars": chars,
            "indexed_at": datetime.now().isoformat(timespec="seconds"),
        }
    save_registry(registry)

    total_elapsed = time.monotonic() - t_start
    log.info("=" * 55)
    log.info("✓ ГОТОВО  |  %d чанков  |  %.1fs всего", len(chunks), total_elapsed)
    log.info("Реестр: %d файлов в %s", len(registry), REGISTRY_PATH)
    log.info("=" * 55)


def run_ingest_file(file_path: str):
    t_start = time.monotonic()
    storage = get_storage()

    log.info("=" * 55)
    log.info("RAG Ingestion  |  режим: SINGLE FILE")
    log.info("file     : %s", file_path)
    log.info("backend  : %s", settings.file_backend)
    log.info("=" * 55)

    registry = load_registry()

    if settings.file_backend == "s3":
        key = file_path
        file_info = storage.get_file_info(key)
        if file_info is None:
            log.error("Файл не найден в S3: %s", key)
            return
        if file_info.extension.lower() not in PARSERS:
            log.error("Неподдерживаемый формат: %s", file_info.extension)
            return

        if not is_already_indexed(file_info, registry):
            temp_path = storage.download_to_temp(key)
            try:
                docs = parse_file(file_info, temp_path)
            finally:
                temp_path.unlink(missing_ok=True)
            if not docs:
                log.error("Не удалось распарсить файл.")
                return

            total_chars = sum(len(d.page_content) for d in docs)
            log.info("OK  %s  —  %s символов, %d стр.", file_info.filename, f"{total_chars:,}", len(docs))

            embeddings = get_embeddings()
            vector_size = len(embeddings.embed_query("тест"))
            qdrant_client = QdrantClient(url=settings.qdrant_url)
            ensure_collection(qdrant_client, vector_size, reset=False)

            chunks = split_documents(docs)
            upload_to_qdrant(chunks, embeddings)

            registry[file_info.filename] = {
                "hash": f"{file_info.size_bytes}_{file_info.last_modified}",
                "source": f"s3://{settings.s3_bucket}/{key}",
                "chunks": len(chunks),
                "chars": total_chars,
                "indexed_at": datetime.now().isoformat(timespec="seconds"),
            }
            save_registry(registry)
        else:
            log.warning("Файл '%s' уже в реестре.", file_info.filename)
    else:
        path = Path(file_path)
        if not path.exists():
            log.error("Файл не найден: %s", file_path)
            return
        if path.suffix.lower() not in PARSERS:
            log.error("Неподдерживаемый формат: %s", path.suffix)
            return

        if is_already_indexed(path, registry):
            log.warning(
                "Файл '%s' уже в реестре с тем же хэшем. " "Передай --reset-file чтобы переиндексировать.",
                path.name,
            )
            return

        log.info("PARSE   %s  (%.1f KB)", path.name, path.stat().st_size / 1024)
        docs = parse_file(path)
        if not docs:
            log.error("Не удалось распарсить файл.")
            return

        total_chars = sum(len(d.page_content) for d in docs)
        log.info("OK  %s  —  %s символов, %d стр.", path.name, f"{total_chars:,}", len(docs))

        embeddings = get_embeddings()
        vector_size = len(embeddings.embed_query("тест"))
        qdrant_client = QdrantClient(url=settings.qdrant_url)
        ensure_collection(qdrant_client, vector_size, reset=False)

        chunks = split_documents(docs)
        upload_to_qdrant(chunks, embeddings)

        registry[path.name] = {
            "hash": file_hash(path),
            "source": str(path),
            "chunks": len(chunks),
            "chars": total_chars,
            "indexed_at": datetime.now().isoformat(timespec="seconds"),
        }
        save_registry(registry)

    log.info("=" * 55)
    log.info("✓ ГОТОВО  |  %d чанков  |  %.1fs", len(chunks), time.monotonic() - t_start)
    log.info("=" * 55)


def list_registry():
    registry = load_registry()
    if not registry:
        log.info("Реестр пуст — ни одного файла не проиндексировано.")
        return
    log.info("Проиндексированных файлов: %d", len(registry))
    log.info("%-50s  %6s  %8s  %s", "Файл", "Чанков", "Символов", "Дата")
    log.info("-" * 85)
    for name, meta in sorted(registry.items()):
        log.info(
            "%-50s  %6s  %8s  %s",
            name[:50],
            meta.get("chunks", "?"),
            f"{meta.get('chars', 0):,}",
            meta.get("indexed_at", "?")[:19],
        )


# ---------------------------------------------------------------------------
# RAG orchestration (uses domain/rag functions + client singletons)
# ---------------------------------------------------------------------------


async def rag_stream(
    question: str,
    history: list[dict],
) -> AsyncIterator[str]:
    import json

    retriever = get_vector_store().as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.retriever_fetch_k},
    )
    llm = get_llm()
    prompt = build_prompt()

    candidates = retriever.invoke(question)
    docs = rerank_documents(question, candidates, top_n=settings.retriever_top_k, reranker=get_reranker())

    context = format_docs(docs)
    sources = extract_sources(docs)
    history_messages = history_to_messages(history)

    messages = prompt.format_messages(
        context=context,
        history=history_messages,
        question=question,
    )

    full_answer = ""
    async for chunk in llm.astream(messages):
        text = chunk.content
        if text:
            full_answer += text
            yield text

    yield f"\n__sources__:{json.dumps(sources, ensure_ascii=False)}"


async def rag_invoke(question: str, history: list[dict]) -> tuple[str, list[dict]]:
    import json

    answer_parts = []
    sources = []

    async for chunk in rag_stream(question, history):
        if chunk.startswith("\n__sources__:"):
            sources = json.loads(chunk.replace("\n__sources__:", ""))
        else:
            answer_parts.append(chunk)

    return "".join(answer_parts), sources
