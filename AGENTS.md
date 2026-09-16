# AGENTS.md — RAG Project

## Quick Commands

```bash
# Development (no Docker)
task install          # Install deps locally via uv
task test             # pytest (DATA_DIR auto-set in tests/conftest.py)
task lint             # ruff check
task fmt              # ruff format + auto-fix

# Docker stack
task init             # First setup: .env files + build image
task up               # Start all services (qdrant, ollama, postgres, server, client)
task up -- gpu        # Start with GPU support
task down             # Stop stack
task build            # Rebuild images (supports -- gpu)
task restart -- server  # Restart single service

# CLI commands (via Docker)
docker compose exec server python main.py runserver --host 0.0.0.0 --port 8001
docker compose exec server python main.py ingest run --docs-dir /code/project/data/docs_sample
docker compose exec server python main.py ingest file /code/project/data/docs_sample/report.pdf
docker compose exec server python main.py ingest list
docker compose exec server python main.py benchmark run --questions /code/project/data/test_questions.json

# Load testing
task loadtest:setup-users  # Create 50 test users via API
task loadtest:smoke        # Smoke test (3 VU, 1 min)
task loadtest:run          # Load test (0→500 VU, 17 min)
task loadtest:spike        # Spike test (50→500 in 30s)
task loadtest:soak         # Soak test (200 VU, 1 hour)
task loadtest:breakpoint   # Breakpoint test (find failure point)
task loadtest:sse          # Locust SSE test (streaming /chat)
```

## Architecture

- **Entry point**: `server/app/main.py` (FastAPI app)
- **Clean architecture layers**: `server/app/domain/` → `server/app/application/` → `server/app/infrastructure/` →
  `server/app/presentation/`
- **CLI**: `server/app/presentation/cli/` — typer-based CLI, invoked via `python main.py <command>`
- **Entrypoint**: `server/entrypoint.sh` does `alembic upgrade head`, then `cd app` then `exec "$@"`
- **API port**: 8001 (not 8000)
- **DATA_DIR**: defaults to `/code/project/data` inside container; tests use `tests/conftest.py` to set it automatically

## Key Paths

```
server/app/              ← Application code (clean architecture layers)
server/app/domain/       ← Business logic (entities, value objects, exceptions, repos interfaces)
server/app/application/  ← Services, ports (protocols), DTOs
server/app/infrastructure/ ← SQLAlchemy, Qdrant, Ollama, S3 implementations
server/app/presentation/ ← FastAPI routes, middleware, exception handlers
server/app/presentation/cli/  ← CLI commands (runserver, ingest, benchmark)
server/app/config.py     ← Pydantic settings (reads server/.env)
server/app/infrastructure/logging/logging_config.py  ← Logging config dict
server/tests/            ← pytest tests
server/pyproject.toml    ← Dependencies (uv)
server/uv.lock           ← Locked versions (committed to git)
data/docs_sample/        ← Documents for indexing
loadtest/                ← Load testing (k6 + Locust)
```

## CLI Commands

| Command                                                 | Description                   |
|---------------------------------------------------------|-------------------------------|
| `runserver --host --port --reload`                      | Run uvicorn server            |
| `ingest run --docs-dir DIR --reset`                     | Full document ingestion       |
| `ingest file PATH --force`                              | Ingest single file            |
| `ingest list`                                           | Show indexed files            |
| `benchmark run --questions --out --top-k --judge-model` | RAG quality benchmark         |

## Environment Variables

Two `.env` files:

```
server/.env       → Qdrant, Ollama, JWT, CORS, OCR, DATA_DIR (application config)
client/.env       → VITE_API_URL (Vite build arg)
```

Both `.env.example` files exist as templates. Taskfile reads both via `dotenv:`.
Root-level `QDRANT_API_KEY` and `VITE_API_URL` for docker-compose interpolation are resolved from these files via Task's
`dotenv:` loading into shell environment.

## Logging

- **`infrastructure/logging/logging_config.py`**: dict-based config with `default`, `detailed`, `uvicorn` loggers
- Custom filters: `ExceptionFilter`, `LevelThresholdFilter`, `LevelMinFilter`, `RequestIDFilter`
- Applied in `lifespan` via `logging.config.dictConfig()`
- All modules use `logging.getLogger("default")` or `logging.getLogger("detailed")`
- No `print()` calls — everything goes through logger

## Model Preloading

- Models (`bge-m3`, `bge-reranker-v2-m3`) are preloaded during startup in `_preload_models()`
- Checks HuggingFace cache first — skips download if already cached
- First request is not blocked by model loading

## Testing

- `task test` runs pytest from `server/` directory
- Tests mock external services (Qdrant, Ollama) — no real services needed
- `tests/conftest.py` sets `DATA_DIR` to temp directory automatically
- Run single test: `cd server && uv run pytest tests/test_rag_chain.py -v`

## Code Style

- **Formatter/Linter**: ruff (line-length=110, target py311)
- `task fmt` runs both `ruff format` and `ruff check --fix`
- `task lint` runs `ruff check` without modifications
- Pre-commit hooks: ruff check, ruff format, pytest

## Gotchas

- `uv.lock` is committed — run `task lock` after changing `pyproject.toml`
- Server runs from `server/` dir via `task install` or Docker
- Local dev without Docker needs `DATA_DIR` env var or use `task test`
- JWT tokens expire after `JWT_EXPIRE_MINUTES` (default 24h) — re-run `task login`
- `task clean` deletes all data (postgres, qdrant, ollama models) — destructive
- First request after restart loads models (~2.5 min) — preloading mitigates this

## Deferred decisions

- **API versioning (FINDING-010 / API-001)**: Намеренно не реализовано внутри FastAPI-роутеров.
  Версионирование будет делаться через nginx (`api.example.com/v1/`) на уровне reverse proxy.
  Это позволяет версионировать API без изменения кода приложений и даёт гибкость при
  миграции клиентов между версиями.

## Docker

- **Compose files**: `docker-compose.yml` (base) + `docker-compose.override.yml` (dev: build + bind-mounts) +
  `docker-compose.gpu.yml` (GPU)
- Base file is pull-only (no `build:` blocks) — production deploys use `SERVER_IMAGE`/`CLIENT_IMAGE` env vars to point
  at GHCR images
- Dev override adds `build:` blocks and live-reload bind-mounts; auto-loaded by `docker compose up`
- Multi-stage build: python-base → builder-base → uv-base → development/production
- venv lives at `/code/.venv` (separate from code, survives bind-mount)
- Server command: `python main.py runserver` (not direct uvicorn)
- Services: qdrant, ollama, postgres, redis, server, worker, minio (S3), client (web UI), tei-embed, tei-rerank (
  optional profile)
- Client runs nginx that proxies `/api/*` to `server:8001` — external nginx can proxy to `client:3001` as a single
  upstream
- No TLS termination inside the stack — use external nginx with certbot/letsencrypt

# Project Engineering Rules

## General

This is a production-grade Python backend.

The project must be treated as a potentially hostile and failure-prone system.

When analyzing or modifying code:

* understand the existing architecture before changing it
* preserve domain invariants
* preserve transaction boundaries
* avoid leaking infrastructure concerns into the domain
* prefer explicit dependencies
* do not introduce abstractions without a concrete reason
* do not silently change business behavior
* do not hide errors
* do not weaken security to make implementation easier

## Production code hygiene

### No `assert` in production code

`assert` statements are stripped when Python runs with `-O` (optimized mode). They provide zero safety in production and create a false sense of correctness.

**Forbidden** in `server/app/**` (excluding `tests/`):

```python
assert user is not None          # WRONG — silently disappears
assert status == 200             # WRONG
assert len(items) > 0            # WRONG
```

**Required replacement** — use explicit checks with proper exceptions:

```python
if user is None:
    raise ValueError("user must not be None")
if status != 200:
    raise RuntimeError(f"unexpected status {status}")
if not items:
    raise ValueError("items must not be empty")
```

Enforced by ruff rule **S101** (enabled in `[tool.ruff.lint] select`). Tests are exempt via `per-file-ignores`.

### No lightweight nested dependencies

Do not pull in transitive dependencies that are not directly used by the project. Every dependency in `pyproject.toml` must be imported explicitly in code.

**Forbidden**:

* Adding a library solely because it re-exports something from another library (use the source library directly).
* Depending on a heavy umbrella package when only a small sub-module is needed.
* Keeping a dependency "just in case" — if no code imports it, remove it.

**Before adding a new dependency**, verify:

1. No existing dependency already provides the needed functionality.
2. The package is actively maintained and has a stable release.
3. Its own dependency tree is reasonable (check with `uv tree`).

After any change to `pyproject.toml`, run `task lock` to update `uv.lock`.

## Architecture

The intended architectural direction is:

Presentation
↓
Application
↓
Domain

Infrastructure implements abstractions required by Application/Domain.

The Domain layer must not depend on:

* FastAPI
* SQLAlchemy
* HTTP
* Redis
* Kafka
* PostgreSQL
* Pydantic
* infrastructure implementations

Application code should not contain infrastructure implementation details.

Routers/controllers should remain thin.

Business rules belong in the domain/application layer according to their nature.

## Database

Database constraints are part of correctness.

Do not rely exclusively on application-level checks for:

* uniqueness
* referential integrity
* state consistency
* required fields

Always consider concurrency between:

READ → CHECK → WRITE

## Transactions

Every important use case must have an explicit transaction boundary.

Be suspicious of:

DB transaction
→ external HTTP call
→ message publishing
→ DB commit

Consider:

* idempotency
* retries
* outbox pattern
* optimistic locking
* pessimistic locking
* transaction isolation

## Async

Async code must not perform blocking I/O in the event loop.

Be suspicious of:

* synchronous HTTP clients
* synchronous DB access
* blocking filesystem operations
* CPU-heavy operations
* subprocesses
* long-running loops

## Security

Every endpoint must be considered attacker-controlled.

Never assume:

* user IDs are trustworthy
* object IDs belong to the authenticated user
* frontend validation is sufficient
* authentication implies authorization
* internal APIs are automatically trusted

Always consider:

* BOLA/IDOR
* privilege escalation
* authentication bypass
* mass assignment
* excessive data exposure
* injection
* SSRF
* resource exhaustion
* race conditions
* sensitive data leakage

## Error handling

Do not use broad exception handling to hide failures.

Avoid patterns such as:

```python
try:
    ...
except Exception:
    pass
```

or:

```python
try:
    ...
except Exception:
    return None
```

unless there is a very explicit and documented reason.

## Testing

Every important business invariant should have a test.

Security-sensitive behavior requires tests for both:

* authorized behavior
* unauthorized behavior

Concurrency-sensitive behavior requires concurrency tests where practical.

## Audit mode

When an audit agent is active:

* do not modify application source code
* do not automatically fix findings
* inspect the entire relevant execution path
* distinguish confirmed issues from hypotheses
* provide file and line references
* explain reproducible failure/security scenarios
* prioritize real impact over code-style preferences

Prefer:

10 proven findings

over:

50 speculative findings.

<!--
  Добавить этот блок в конец существующего server/../AGENTS.md
  (или в корневой AGENTS.md проекта — opencode подтягивает его автоматически).
  Ничего из существующего файла не удаляем, только дополняем.
-->

## Layering rules (enforced, not aspirational)

Текущее состояние ЧИСТОЕ — сохраняем инвариант:

- `server/app/domain/**` — НИКОГДА не импортирует `application.*`, `infrastructure.*`,
  `presentation.*`, `config` (см. `domain/services/rag_policy.py` как эталон:
  "framework-agnostic (no LangChain, no infrastructure imports)").
- `server/app/application/**` — импортирует только `domain.*` и собственные `application.ports.*`
  (Protocol-классы, см. `application/ports/chat_rag_port.py`). НИКОГДА не импортирует
  `infrastructure.*` напрямую — только через порт, инжектированный в конструктор/DI.
- `server/app/infrastructure/**` — реализует порты из `application/ports/`. Классы называются
  `<Tech><PortName>`, например `infrastructure/repositories/sqlalchemy_document_repository.py`
  реализует `domain/repositories/document_repository.py`.
- `server/app/composition/**` — единственное место, где происходит связывание конкретных
  infrastructure-реализаций с application-сервисами (`composition/application.py`,
  `composition/infrastructure.py`, `composition/service_providers.py`).

Перед любым PR, трогающим `domain/` или `application/`, agent обязан прогнать:

```bash
grep -rn "^from \(application\|infrastructure\|presentation\)\|^import \(application\|infrastructure\|presentation\)" server/app/domain/
grep -rln "^from \(infrastructure\|presentation\)\|^import \(infrastructure\|presentation\)" server/app/application/
```

Оба должны вернуть пусто. Если grep что-то нашёл — это регрессия, а не рефакторинг.

## Known god-files (refactor backlog, приоритет сверху вниз)

| # | Файл                                                  | Строк | Проблема                                                                                                                          | Тесты сейчас                                                          |
|---|-------------------------------------------------------|-------|-----------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| 1 | `server/app/infrastructure/ml/rag_service.py`         | 283   | ~~`stream()` 270 строк~~ → рефакторинг: `stream()` вынесен в `rag/rag_steps.py`, retrieval — в `rag/rag_retrieval.py`             | `test_curator_role.py`, `test_rag_chain.py`, `tests/fakes.py`         |
| 2 | `server/app/infrastructure/ml/benchmark/`             | ~1200 | Разбит на `benchmark.py` (35) + `runner.py`, `retrieval.py`, `sweep_scoring.py`, `report.py` —但仍无 unit-тестов на judge/metrics | Косвенно через `test_benchmark*.py`                                   |
| 3 | `server/app/infrastructure/ml/ingestion/`             | ~1900 | Разбит по форматам: `pdf.py`, `docx.py`, `markdown.py`, `rtf.py`, `splitting.py` — Strategy-паттерн применён                      | `test_ingestion_parsers.py`, `test_ingestion_text.py`                 |
| 4 | `server/app/presentation/api/schemas.py`              | 813   | Все Pydantic-схемы (chat/admin/document/auth) в одном файле                                                                       | —                                                                     |
| 5 | `server/app/application/services/document_service.py` | 518   | Смешаны оркестрация pipeline и CRUD                                                                                               | —                                                                     |
| 6 | `server/app/application/services/chunk_service.py`    | 532   | Добавлена передача ACL в BM25 при add/replace                                                                                     | `tests/test_chunk_service.py` есть, но частично                       |

Правило: **не трогать файл из этого списка без предварительного `/add-characterization-tests`**
на затрагиваемый публичный метод. Baseline тестов должен быть зелёным ДО рефакторинга.

## RAG-pipeline specific conventions

- Оркестрация ответа: `presentation/api/routes/chat.py` → `application/services/chat_service.py`
  (`ChatService`) → `infrastructure/ml/rag_service.py` (`RagService`, реализует
  `application/ports/chat_rag_port.py::ChatRAGPort`).
- Чистая доменная логика классификации/промптов живёт в `domain/services/rag_policy.py` —
  новую бизнес-логику (не зависящую от LangChain/Qdrant/Ollama) добавлять туда, а не в
  `infrastructure/ml/rag.py` или `rag_service.py`.
- Ретеривал: dense — `infrastructure/ml/rag/rag_retrieval.py::qdrant_dense_search`,
  гибридный — `infrastructure/ml/rag/rag_retrieval.py::run_hybrid_search`. При рефакторинге
  эти функции должны остаться чистыми (без побочных эффектов логирования вперемешку с бизнес-логикой).
- **ACL-инвариант** (нарушение = data leak): `is_in_search_scope(doc, ctx)` ⇔ `build_qdrant_filter`
  ⇔ `BM25Index._doc_matches_acl`. Тройной треугольник покрыт property-тестами:
  - `tests/test_acl.py::TestACLInvariant` — domain ⇔ Qdrant filter
  - `tests/test_bm25_acl.py::TestBM25PredicateInvariant` — domain ⇔ BM25 predicate
- **CuratorScope**: `domain/value_objects/curator_scope.py` — frozen VO, snapshot полномочий куратора
  на момент запроса. Пробрасывается через `ChatContext.curator_scope` в `RagService._init_state`.
  Максимальный размер managed-списков: `CURATOR_SCOPE_MAX_IDS` (config, по умолчанию 1000),
  проверяется в `ChatService._prepare_chat`.
- **BM25 ACL pre-filter**: `infrastructure/bm25/bm25_index.py` хранит per-doc ACL-метаданные
  (`doc_visibility/owner_id/group_id`), pre-filter до скоринга через `_doc_matches_acl`.
  Второй рубеж (Qdrant resolve с `access_filter`) **никогда не убирается** — defense in depth.
  `last_survival_ratio` на индексе → метрика `rag_sparse_survival_ratio` с лейблом `role`
  в `rag_steps.py::step_retrieve`.
- **Answer cache**: `CACHE_PREFIX = "rag:cache:v3:"`. Hash включает `user_kind:user_role:user_id:sorted(groups):curator_scope`.
  Смена assignments или роли → новый хэш → cache miss (fail-closed).
- `domain/services/access_control.py` — single source of truth для ACL. Все пути (CRUD listing,
  Qdrant filter, BM25 predicate) берут условия из `get_visibility_conditions()`.

### Invariant tests (никогда не удалять)

| Тест                                                  | Что фиксирует                                                         |
|-------------------------------------------------------|-----------------------------------------------------------------------|
| `test_acl.py::TestACLInvariant`                       | `is_in_search_scope` ⇔ `build_qdrant_filter` (USER/ADMIN/CURATOR)    |
| `test_bm25_acl.py::TestBM25PredicateInvariant`        | `is_in_search_scope` ⇔ `BM25Index._doc_matches_acl` (USER/ADMIN/CURATOR) |
| `test_answer_cache.py::TestCacheHashIncludesRoleAndScope` | Cache hash includes role + scope; downgrade → different hash       |
| `test_curator_role.py::TestRagServiceCuratorScope`    | `RagService._init_state` passes managed-ids from CuratorScope         |
| `test_bm25_acl.py::TestBM25SearchWithACL`             | BM25 pre-filter correctly excludes docs by visibility/owner/group     |

## ACL data flow

```
UserContext.build()          → managed_*_ids from DB (CURATOR only)
  → ChatService._prepare_chat → CuratorScope VO (CURATOR_SCOPE_MAX_IDS checked)
    → ChatContext.curator_scope
      → RagService._init_state
        → get_visibility_conditions(for_list=False) → VisibilityCondition[]
          → build_qdrant_filter()            → Qdrant Filter (dense + resolve)
          → BM25Index.search_with_hashes()   → pre-filter (first barrier)
            → resolve_hashes_batch()         → second barrier (Qdrant + access_filter)
        → answer_cache.compute_visibility_scope_hash() → cache key (v3)
```

## Refactor status

Полный план: `docs/refactor/REFACTOR_PLAN.md`

| Фаза | Статус | Описание |
|---|---|---|
| Фаза 1: Security | ✅ DONE | CuratorScope, BM25 pre-filter, neighbors ACL, answer cache v3, dead code removal |
| Фаза 2: Resilience | ❌ NOT STARTED | Circuit breaker, async client, separate semaphores, Qdrant timeout/retry |
| Фаза 3: Decoupling | ❌ NOT STARTED | 15+ infrastructure imports в application, uow._session, LangChain |
| Фаза 4: API | 🟡 PARTIAL | extra="forbid" ✅, cache invalidation ✅, email validation ✅; schemas split ❌, rate limiting ❌ |
| Фаза 5: God-Files | ✅ DONE | rag_service stream → rag_steps, ingestion → parsers, benchmark → modules |
| Фаза 6: Performance | ✅ DONE | BM25 thread safety, reverse index, gzip, batch rebuild, metrics |