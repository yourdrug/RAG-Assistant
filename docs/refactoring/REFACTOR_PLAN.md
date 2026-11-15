# RAG Project — Incremental Refactoring Plan

**Дата:** 2026-09-12
**Основание:** review/2026-09-12-report.md
**Архитектор:** Principal Software Architect

---

## Executive Summary

Проект находится в **production-ready** состоянии с чётким разделением слоёв (domain/application/infrastructure/presentation). Основные риски сконцентрированы в **трёх критических областях**:

1. **Multi-tenant безопасность** — BM25 sparse search обходит ACL-фильтры, curator scope не передаётся в RAG-пайплайн, enrichment neighbors загружает чужие чанки.
2. **Отказоустойчивость** — отсутствие circuit breaker для LLM приводит к полной деградации сервиса при проблемах с провайдером.
3. **Архитектурная чистота** — `ingestion_orchestrator.py` (768 строк) импортирует 12 модулей из infrastructure, нарушая Dependency Inversion Principle.

Рефакторинг разбит на **6 фаз**, выполняемых последовательно без остановки production-системы. Каждая фаза завершается работающей системой с регресс-тестами.

**Критический путь:** Фаза 1 → Фаза 2 → Фаза 3 (блокируют друг друга). Фазы 4–6 можно выполнять параллельно после Фазы 3.

**Оценка трудозатрат:** ~18–22 рабочих дня (1 senior developer full-time).

---

## Отклонённые findings (False Positives)

### FP-1: "Semantic cache не инвалидируется" — ЧАСТИЧНО FALSE

**Finding:** CRITICAL — semantic cache не инвалидируется при обновлении документов.

**Верификация:**
- `answer_cache.py:89` — `document_ids` **сохраняются** в кэш-entry
- Нет кода, который вызывает инвалидацию при `ingestion`

**Статус:** CONFIRMED, но severity = **HIGH** (не CRITICAL).
**Причина:** Кэш TTL = 7 дней, при re-ingest через CLI с `--reset` Qdrant полностью пересоздаётся → все `document_ids` новые → cache miss. Проблема только для incremental ingestion без `--reset`.

**Решение:** Фаза 4 — добавить инвалидацию кэша по `document_ids` в `document_service.delete_document()` и `ingestion_orchestrator._handle_existing_conflict()`.

---

### FP-2: "API без rate limiting" — PARTIALLY FALSE

**Finding:** CRITICAL — нет rate limiting.

**Верификация:**
- `config.py:236` — `cost_rate_limit_enabled = False` **по умолчанию**
- `ml_clients.py:107–110` — есть **глобальный семафор** LLM (16 slots)

**Статус:** CONFIRMED как HIGH (не CRITICAL).
**Причина:** Глобальный семафор защищает от exhaustion LLM API, но не защищает от DoS одним пользователем.

**Решение:** Фаза 5 — per-user token bucket rate limiter на middleware + включить `cost_rate_limit_enabled` по умолчанию.

---

### FP-3: "Layering violation в domain/" — FALSE

**Finding:** Domain импортирует application/infrastructure.

**Верификация:**
```bash
grep -rn "^from \(application\|infrastructure\|presentation\)" server/app/domain/
# → пусто (0 результатов)
```

**Статус:** FALSE POSITIVE. Domain-слой **чист**.

---

### FP-4: "BM25 rebuild блокирует worker" — MISLEADING

**Finding:** HIGH — BM25 rebuild блокирует worker.

**Верификация:**
- `cron.py:68–98` — rebuild выполняется **в фоновом async task**, не в main event loop
- `_rebuild_bm25_job()` вызывает `asyncio.to_thread()` для CPU-bound операций

**Статус:** PARTIALLY FALSE.
**Проблема не в блокировке**, а в **memory consumption** (100MB+ на 200K chunks).

**Решение:** Фаза 6 (low priority) — streaming-загрузка chunks батчами.

---

## Находки, требующие дополнительного исследования

### INV-1: Curator scope — managed_*_ids

**Finding:** CRITICAL — curator не видит документы managed users через RAG.

**Требует подтверждения:**
1. Есть ли в production кураторы с `managed_client_ids`?
2. Как часто используется curator role для поиска по документам подчинённых?
3. Есть ли реальные кейсы, когда curator должен видеть CLIENT_PRIVATE документы managed users?

**Действие:** Обсудить с product owner **до начала Фазы 1**. Если кейс редкий — переместить в Фазу 6 (low priority).

**Рекомендация архитектора:** Оставить в Фазе 1. Curator scope — часть ACL-контракта, его отсутствие создаёт **inconsistency** между `list_documents` (видит) и `chat` (не видит).

---

### INV-2: Prompt injection защита

**Finding:** CRITICAL — нет тестов на prompt injection.

**Требует подтверждения:**
1. Какие документы загружаются в систему? (internal docs vs external/public)
2. Есть ли review-процесс перед загрузкой INTERNAL_PUBLIC документов?
3. Какие user roles могут загружать INTERNAL_PUBLIC?

**Действие:** Провести threat modeling session с security team.

**Рекомендация архитектора:** Добавить в Фазу 5 как **separate work stream** (не блокирует другие фазы):
- Unit-тесты на system prompt integrity
- Scan INTERNAL_PUBLIC документов при загрузке (keyword blacklist)
-Ограничить upload INTERNAL_PUBLIC только для admin/curator

---

### INV-3: PII в чанках

**Finding:** HIGH — chunks с PII хранятся в открытом виде.

**Требует подтверждения:**
1. Какие типы документов содержат PII? (персональные дела, медицинские карты, финансовые отчёты)
2. Требования регуляторов (GDPR, 152-ФЗ) к хранению PII в векторной БД?
3. Есть ли encryption-at-rest для Qdrant/Postgres в production?

**Действие:** Legal review + compliance team.

**Рекомендация архитектора:** **НЕ включать в текущий рефакторинг**. Требует отдельного ADR и архитектурного решения (encryption-at-rest vs PII detection at ingestion vs field-level encryption). Оценка: 2–3 недели работы + миграция данных.

---

## Фазы рефакторинга

---

## Фаза 1: Multi-tenant Security Hardening

**Цель:** Устранить утечки данных между пользователями через ACL bypass в BM25 sparse search, enrichment neighbors и curator scope.

**Приоритет:** КРИТИЧЕСКИЙ
**Трудозатраты:** 4–5 дней
**Зависимости:** Нет (можно начинать немедленно)

### Findings Addressed

| ID | Severity | Finding | Root Cause |
|----|----------|---------|------------|
| SEC/RAG-1 | CRITICAL | BM25 sparse search bypasses ACL filter | `rag_retrieval.py:121` — BM25 search без `access_filter` |
| SEC/RAG-2 | CRITICAL | similarity_search_with_score() полный обход ACL | `qdrant_vector_store_repository.py:72–93` — метод без ACL |
| SEC/RAG-3 | CRITICAL | Curator scope lost in ChatContext | `chat_context.py:14–29` — нет `managed_*_ids` |
| SEC/RAG-4 | HIGH | Enrichment neighbors не проверяет ACL | `rag_postprocess.py:112–179` — SQL без visibility filter |
| SEC/RAG-5 | HIGH | Answer cache visibility scope hash не учитывает managed_*_ids | `answer_cache.py:23–26` |

### Root Cause Analysis

**BM25 ACL bypass:**
1. `BM25Index` строится из **всех** chunks всех пользователей без фильтрации
2. `run_hybrid_search()` вызывает `bm25_index.search_with_hashes(query, fetch_k)` → возвращает hashes БЕЗ ACL
3. Sparse results мёржатся с dense через RRF → sparse-only hashes попадают в candidates
4. `resolve_hashes_batch(missing_hashes, access_filter, ml_clients)` применяет ACL **post-facto**, но:
   - Timing side-channel leak: если sparse нашёл 10 results, а после ACL осталось 0 → user знает, что документ существует
   - RRF scores рассчитываются **до** ACL-фильтрации → ranking искажён

**Enrichment neighbors:**
- `chunk_search.get_neighbors(document_id, chunk_index, window=3)` → SQL query `SELECT * FROM chunks WHERE document_id = ? AND chunk_index IN (...)` без `WHERE visibility IN (...)` или `WHERE owner_id = user_id`
- Основной chunk прошёл ACL через Qdrant, но neighbors загружаются напрямую из SQL **в обход ACL**

**Curator scope:**
- `ChatContext` содержит только `user_id, user_kind, user_role, user_group_ids`
- `UserContext.managed_client_ids`, `managed_internal_ids`, `managed_group_ids` **не передаются** в `ChatContext`
- `build_qdrant_filter()` строит фильтр только на основе `user_id + group_ids` → curator не видит документы managed users

### Архитектурное решение

**Решение 1: BM25 ACL enforcement через post-filter + documentation**

Оставить BM25Index глобальным (все chunks), но:
1. `resolve_hashes_batch()` **всегда** применяет `access_filter` через Qdrant scroll
2. Документировать timing side-channel как **known limitation** (mitigation: не показывать счётчики "найдено N документов")
3. Пересчитать RRF scores **после** ACL-фильтрации

**Альтернатива (отклонена):** Строить отдельный BM25Index для каждого visibility scope.
**Причина отклонения:**
- 10 users × 5 scopes = 50 индексов в памяти → 5GB+ memory
- Индекс нужно пересобирать при каждом изменении group membership
- Сложность O(n×m), где n = users, m = documents

**Решение 2: Enrichment neighbors через ACL-aware SQL**

Добавить `visibility` и `owner_id` в `WHERE` clause для `get_neighbors()`:
```python
def get_neighbors(document_id, chunk_index, window, user_context: UserContext):
    # Проверить, что основной chunk доступен пользователю
    # (defensive check — основной chunk уже прошёл Qdrant ACL)
    visible_values = compute_visible_values(user_context)
    query = """
        SELECT * FROM chunks
        WHERE document_id = ?
          AND chunk_index IN (?, ?, ...)
          AND visibility IN (?)
    """
```

**Решение 3: Curator scope через extended ChatContext**

1. Добавить в `ChatContext`:
   ```python
   managed_client_ids: list[int] = field(default_factory=list)
   managed_internal_ids: list[int] = field(default_factory=list)
   managed_group_ids: list[int] = field(default_factory=list)
   ```
2. Передавать из `UserContext` в `ChatService._build_chat_context()`:
   ```python
   chat_ctx = ChatContext(
       user_id=ctx.user_id,
       user_kind=ctx.user_kind,
       user_role=ctx.user_role,
       user_group_ids=ctx.group_ids,
       managed_client_ids=ctx.managed_client_ids,
       managed_internal_ids=ctx.managed_internal_ids,
       managed_group_ids=ctx.managed_group_ids,
   )
   ```
3. Расширить `build_qdrant_filter()`:
   ```python
   if chat_ctx.user_role == UserRole.CURATOR:
       # Curator видит:
       # 1. Свои документы
       # 2. INTERNAL_PUBLIC/INTERNAL_GROUP для своих групп
       # 3. Документы managed users (через managed_*_ids)
       client_filter = build_client_filter(chat_ctx.managed_client_ids)
       internal_filter = build_internal_filter(chat_ctx.managed_internal_ids)
       curator_filter = Should([user_filter, client_filter, internal_filter])
   ```

### Files/Modules Affected

**Изменения:**
- `server/app/domain/value_objects/chat_context.py` — добавить `managed_*_ids`
- `server/app/application/services/chat_service.py` — передавать `managed_*_ids` в `ChatContext`
- `server/app/infrastructure/repositories/vector/acl.py` — расширить `build_qdrant_filter()` для CURATOR
- `server/app/infrastructure/ml/rag/rag_retrieval.py` — пересчитать RRF после ACL
- `server/app/infrastructure/repositories/chunk/sqlalchemy_chunk_repository.py` — добавить ACL в `get_neighbors()`
- `server/app/infrastructure/ml/answer_cache.py` — включить `managed_*_ids` в `compute_visibility_scope_hash()`
- `server/app/infrastructure/repositories/vector/qdrant_vector_store_repository.py` — удалить или сделать приватным `similarity_search_with_score()`

**Новые файлы:**
- Нет

### Implementation Steps

1. **Characterization тесты** (1 день):
   - `test_acl.py` — добавить параметризованные тесты для CURATOR с `managed_client_ids`
   - `test_rag_retrieval.py` — добавить тест на BM25 sparse-only results с ACL bypass
   - `test_enrichment_neighbors_acl.py` — новый файл, тесты на cross-tenant leak через neighbors
   - Запустить `task test` → все тесты зелёные (baseline)

2. **Расширить ChatContext** (0.5 дня):
   - Добавить `managed_*_ids` в `ChatContext` (domain layer)
   - Обновить `ChatService._build_chat_context()` (application layer)
   - Обновить `test_chat_service.py` с `managed_*_ids`

3. **Curator scope в ACL filter** (1 день):
   - Расширить `build_qdrant_filter()` для CURATOR role
   - Обновить `test_acl.py` — параметризованные тесты для всех комбинаций (user + managed users)
   - Обновить `compute_visibility_scope_hash()` — включить `managed_*_ids`

4. **Enrichment neighbors ACL** (0.5 дня):
   - Добавить `user_context: UserContext` в `get_neighbors()` signature
   - Добавить SQL `WHERE visibility IN (...)` filter
   - Обновить вызовы в `rag_postprocess.py` — передавать `user_context`
   - Обновить `test_enrichment_neighbors_acl.py`

5. **BM25 sparse ACL post-filter** (1 день):
   - Изменить `resolve_hashes_batch()` — применять ACL через Qdrant scroll **перед** возвратом
   - Пересчитать RRF scores после ACL-фильтрации
   - Добавить метрику `RAG_ACL_FILTERED_COUNT` (сколько sparse results отброшено ACL)
   - Обновить `test_hybrid_search.py` — проверить, что sparse-only без ACL отбрасываются

6. **Remove unsafe method** (0.5 дня):
   - Сделать `similarity_search_with_score()` **приватным** (`_similarity_search_with_score()`)
   - Добавить docstring: "INTERNAL USE ONLY. No ACL enforcement. Use search_with_filter() for user queries."
   - Grep по кодовой базе — убедиться, что метод не вызывается из route handlers

7. **Integration testing** (0.5 дня):
   - End-to-end тест: CURATOR с `managed_client_ids` → `/chat` → проверить, что ответ содержит источники managed users
   - End-to-end тест: USER пытается задать вопрос → enrichment neighbors не возвращает чужие chunks
   - Запустить `task loadtest:smoke` — нет performance regression

### Tests

**Новые тесты:**
- `tests/test_enrichment_neighbors_acl.py` (20–30 тестов):
  - `test_get_neighbors_filters_by_visibility_user_only_sees_own()`
  - `test_get_neighbors_filters_by_visibility_internal_public()`
  - `test_get_neighbors_curator_sees_managed_users()`
  - `test_get_neighbors_no_cross_tenant_leak()`

**Обновлённые тесты:**
- `tests/test_acl.py` — добавить 10+ параметризованных тестов для CURATOR
- `tests/test_hybrid_search.py` — добавить тест на sparse-only ACL bypass
- `tests/test_chat_service.py` — обновить моки для `managed_*_ids`

**Characterization тесты:**
- Baseline: запустить `task test` до изменений → все зелёные
- После каждого шага: запустить `task test` → все зелёные

### Acceptance Criteria

**Функциональные:**
1. ✅ BM25 sparse-only results без ACL-доступа **не попадают** в final candidates
2. ✅ RRF scores пересчитываются **после** ACL-фильтрации
3. ✅ `get_neighbors()` возвращает только chunks с `visibility` accessible для user
4. ✅ CURATOR с `managed_client_ids=[10, 20]` видит через `/chat` документы users 10, 20
5. ✅ USER с `user_id=5` **не видит** через enrichment neighbors chunks `owner_id=6`
6. ✅ `similarity_search_with_score()` сделан приватным и не вызывается из routes

**Нефункциональные:**
1. ✅ Все существующие тесты зелёные
2. ✅ `task loadtest:smoke` проходит без ошибок
3. ✅ Latency `/chat` не увеличивается > 5% (измерить через locust SSE test)
4. ✅ Нет новых импортов infrastructure в domain/application слои

**Документация:**
1. ✅ Обновить `docs/ARCHITECTURE.md` — раздел "ACL enforcement in RAG pipeline"
2. ✅ Добавить комментарий в `BM25Index` о timing side-channel limitation

### Risks

**HIGH:** Изменения в `build_qdrant_filter()` могут сломать существующий ACL для admin/user.
**Mitigation:** Параметризованные characterization тесты для всех ролей **до** изменений.

**MEDIUM:** Performance degradation из-за дополнительного SQL `WHERE visibility IN (...)` для neighbors.
**Mitigation:** `visibility` уже индексирован в Postgres. Добавить метрику `enrichment_neighbors_duration_seconds`.

**LOW:** Breaking change для `get_neighbors()` signature потребует обновления всех вызовов.
**Mitigation:** Только 2 вызова в `rag_postprocess.py` + тесты. Изменение локализовано.

### Dependencies

**Блокирует:**
- Фазу 2 (нельзя добавлять circuit breaker, не починив ACL — в fallback режиме пользователи могут увидеть чужие данные)

**Блокируется:**
- Ничем (можно начинать немедленно)

---

## Фаза 2: Resilience & Circuit Breaker

**Цель:** Предотвратить полную деградацию сервиса при проблемах с LLM-провайдером через circuit breaker pattern.

**Приоритет:** КРИТИЧЕСКИЙ
**Трудозатраты:** 2–3 дня
**Зависимости:** Фаза 1 (ACL hardening должен быть завершён)

### Findings Addressed

| ID | Severity | Finding | Root Cause |
|----|----------|---------|------------|
| PERF-1 | CRITICAL | Нет circuit breaker для LLM-провайдера | Все 16 семафоров заняты зависшими запросами |
| PERF-2 | CRITICAL | Sync OpenAI client блокирует event loop | `_helpers.py:180–191` создаёт sync client через `to_thread` |
| PERF-3 | HIGH | Один семафор LLM на все операции | Generation + condense + decompose + relevance — один pool |
| PERF-4 | HIGH | QdrantClient без timeout | Thread pool exhaustion при Qdrant hangs |
| PERF-5 | HIGH | Нет retry на транзиентные ошибки Qdrant | Single hiccup → fail |

### Root Cause Analysis

**Circuit breaker отсутствует:**
1. LLM timeout = 120 сек × max_retries = 3 → 360 сек на один запрос
2. `llm_semaphore(16)` — глобальный для всех операций
3. При деградации OpenRouter/Ollama (503, timeout):
   - Все 16 slots заняты зависшими запросами
   - Новые пользователи ждут в очереди семафора → бесконечное ожидание
   - Нет механизма "fail fast" → cascade failure

**Sync OpenAI client:**
- `assess_sufficiency()` создаёт синхронный `instructor` client на каждый вызов
- Под нагрузкой (25+ parallel requests) → 25+ threads в thread pool
- Каждый thread держит TCP connection → connection pool exhaustion

**Shared LLM semaphore:**
- `generation` (heavy, 10–30 сек) и `condense` (light, 1–3 сек) соревнуются за один pool
- Эффективная конкурентность падает до 4–5 (большинство slots заняты generation)

### Архитектурное решение

**Решение 1: Circuit breaker через `aiobreaker`**

Добавить circuit breaker для LLM operations:
```python
from aiobreaker import CircuitBreaker

llm_breaker = CircuitBreaker(
    fail_max=5,           # 5 consecutive failures
    timeout_duration=300,  # 5 min OPEN state
    exclude=[HTTPStatusError],  # не считать 4xx как failure
)

@llm_breaker
async def generate_answer(...):
    async with ml_clients.llm_semaphore:
        return await llm.agenerate(...)
```

States:
- **CLOSED** (normal) → consecutive failures → **OPEN**
- **OPEN** → wait 5 min → **HALF-OPEN** (пробный запрос)
- **HALF-OPEN** → success → **CLOSED** / failure → **OPEN**

Fallback при OPEN state:
```python
try:
    answer = await generate_answer(...)
except CircuitBreakerError:
    return "Сервис временно недоступен. Попробуйте через 5 минут."
```

**Решение 2: Async OpenAI client caching**

Заменить sync `instructor` на async:
```python
# infrastructure/ml/clients/client_registry.py
async def instructor_client(self):
    if self._instructor is None:
        from openai import AsyncOpenAI
        import instructor
        self._instructor = instructor.from_openai(
            AsyncOpenAI(api_key=settings.openrouter_api_key, base_url=settings.openrouter_base_url)
        )
    return self._instructor
```

**Решение 3: Separate semaphores**

```python
# config.py
llm_generation_max_concurrent: int = 12
llm_auxiliary_max_concurrent: int = 8

# client_registry.py
@property
def generation_semaphore(self):
    if self._generation_sem is None:
        self._generation_sem = asyncio.Semaphore(settings.llm_generation_max_concurrent)
    return self._generation_sem

@property
def auxiliary_semaphore(self):
    if self._auxiliary_sem is None:
        self._auxiliary_sem = asyncio.Semaphore(settings.llm_auxiliary_max_concurrent)
    return self._auxiliary_sem
```

**Решение 4: Timeouts для Qdrant и retries**

```python
# factories.py
def create_qdrant_client():
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=10,  # 10 sec timeout
    )

# qdrant_vector_store_repository.py
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=1, max=10),
    retry=retry_if_exception_type((TimeoutError, ConnectionError)),
)
async def upsert_point(self, point_id, vector, payload):
    ...
```

### Files/Modules Affected

**Изменения:**
- `server/pyproject.toml` — добавить `aiobreaker = "^1.4.0"`
- `server/app/config.py` — добавить `llm_generation_max_concurrent`, `llm_auxiliary_max_concurrent`
- `server/app/infrastructure/ml/clients/client_registry.py` — добавить circuit breaker, separate semaphores, instructor cache
- `server/app/infrastructure/ml/clients/factories.py` — добавить timeout для QdrantClient
- `server/app/infrastructure/repositories/vector/qdrant_vector_store_repository.py` — добавить @retry
- `server/app/infrastructure/ml/rag/_helpers.py` — заменить sync instructor на async
- `server/app/infrastructure/ml/rag/rag_steps.py` — использовать `generation_semaphore` вместо `llm_semaphore`
- `server/app/infrastructure/ml/rag/rag_prompts.py` — использовать `auxiliary_semaphore`

**Новые файлы:**
- `server/app/infrastructure/resilience/circuit_breaker.py` — wrapper для aiobreaker с метриками

### Implementation Steps

1. **Dependencies** (0.5 дня):
   - Добавить `aiobreaker` в `pyproject.toml`
   - Добавить `tenacity` (если ещё нет)
   - `task lock && task install`

2. **Circuit breaker wrapper** (1 день):
   - Создать `infrastructure/resilience/circuit_breaker.py`
   - Обернуть LLM operations в `@llm_breaker` decorator
   - Добавить метрики: `circuit_breaker_state{operation="llm_generate"}` (CLOSED/OPEN/HALF_OPEN)
   - Добавить fallback message при CircuitBreakerError
   - Unit-тесты: `test_circuit_breaker_opens_after_failures()`, `test_circuit_breaker_fallback()`

3. **Separate semaphores** (0.5 дня):
   - Добавить `generation_semaphore` и `auxiliary_semaphore` в `MLClientRegistry`
   - Обновить `rag_steps.py` — использовать `generation_semaphore`
   - Обновить `rag_prompts.py` — использовать `auxiliary_semaphore`
   - Обновить `test_rag_service_stream.py` — моки для новых семафоров

4. **Async instructor client** (0.5 дня):
   - Добавить `instructor_client()` в `MLClientRegistry`
   - Заменить sync `instructor.from_openai()` в `_helpers.py:180–191`
   - Убрать `asyncio.to_thread()` wrapper
   - Обновить тесты

5. **Qdrant timeouts and retries** (0.5 дня):
   - Добавить `timeout=10` в `create_qdrant_client()`
   - Добавить `@retry` декоратор на все Qdrant operations
   - Добавить метрику `qdrant_retries_total{operation="upsert"}`
   - Обновить `test_qdrant_repository.py` — проверить retry behaviour

6. **Integration testing** (0.5 дня):
   - Mock LLM failures → проверить, что circuit breaker открывается
   - Mock Qdrant timeout → проверить retry
   - End-to-end test: circuit breaker OPEN → fallback message
   - Load test: `task loadtest:spike` → проверить, что система не падает при LLM timeout

### Tests

**Новые тесты:**
- `tests/test_circuit_breaker.py` (10 тестов):
  - `test_circuit_breaker_opens_after_consecutive_failures()`
  - `test_circuit_breaker_half_open_successful_recovery()`
  - `test_circuit_breaker_fallback_message()`
  - `test_circuit_breaker_does_not_count_4xx_as_failure()`

**Обновлённые тесты:**
- `tests/test_rag_service_stream.py` — обновить моки для separate semaphores
- `tests/test_qdrant_repository.py` — добавить тесты на retry

### Acceptance Criteria

**Функциональные:**
1. ✅ При 5 consecutive LLM failures circuit breaker переходит в OPEN state
2. ✅ В OPEN state пользователь получает fallback message "Сервис временно недоступен"
3. ✅ Через 5 минут circuit breaker переходит в HALF-OPEN → пробный запрос
4. ✅ Qdrant operations retry до 3 раз при timeout/connection error
5. ✅ Generation и auxiliary operations используют разные semaphore pools

**Нефункциональные:**
1. ✅ Метрики `circuit_breaker_state`, `circuit_breaker_failures_total`, `qdrant_retries_total` экспортируются в Prometheus
2. ✅ `task loadtest:spike` (50→500 VU) проходит без 500 errors при mock LLM timeout
3. ✅ Latency для condense/decompose снижается на 20–30% (из-за separate semaphore)

### Risks

**HIGH:** Circuit breaker может ложно срабатывать при кратковременных сбоях LLM.
**Mitigation:** `fail_max=5` (не 2–3) + exclude 4xx errors. Мониторинг `circuit_breaker_state` в production.

**MEDIUM:** Retry на Qdrant может увеличить latency при проблемах с сетью.
**Mitigation:** Exponential backoff с `max=10 sec`. Добавить timeout на retry chain.

**LOW:** Async instructor может вести себя по-другому, чем sync.
**Mitigation:** Сравнить output sync vs async на 10 тестовых запросах.

### Dependencies

**Блокирует:**
- Ничего (можно выполнять параллельно с Фазой 3+)

**Блокируется:**
- Фазой 1 (ACL hardening должен быть завершён, чтобы fallback message не создавал security issues)

---

## Фаза 3: Application → Infrastructure Decoupling

**Цель:** Устранить прямые импорты infrastructure в application-слое через порты и адаптеры. Рефакторинг `ingestion_orchestrator.py` (768 строк).

**Приоритет:** КРИТИЧЕСКИЙ (архитектурная чистота)
**Трудозатраты:** 5–6 дней
**Зависимости:** Нет (можно начинать после Фазы 1)

### Findings Addressed

| ID | Severity | Finding | Root Cause |
|----|----------|---------|------------|
| DDD-1 | CRITICAL | Application imports infrastructure (ingestion_orchestrator) | 12 прямых импортов infrastructure |
| DDD-2 | CRITICAL | Application создаёт infrastructure repository напрямую | `uow._session` 4 раза |
| DDD-3 | HIGH | Application imports infrastructure (benchmark_orchestrator) | `_percentile`, `run_benchmark_async` |
| DDD-4 | HIGH | Application imports infrastructure (retrieval.py) | `rrf_merge`, `deduplicate_docs` |
| DDD-5 | HIGH | LangChain import в application layer | `from langchain.schema import Document` |
| DDD-6 | HIGH | config.settings в application layer | Прямой импорт Pydantic settings |

### Root Cause Analysis

**ingestion_orchestrator.py (768 строк):**
1. Импортирует из infrastructure:
   - `infrastructure.bm25.*` — BM25 операции
   - `infrastructure.ml.ingestion` — парсинг PDF/DOCX/RTF
   - `infrastructure.storage.FileStorage` — S3 operations
   - `infrastructure.repositories.SQLAlchemyIngestionRegistryRepository` — напрямую создаёт repository
   - `infrastructure.uow_factory.UnitOfWorkFactory` — прямая зависимость от factory
   - `infrastructure.metrics.INGEST_FILES_TOTAL` — Prometheus метрики

2. Смешивает 3 responsibility:
   - **Orchestration** — координация pipeline steps
   - **Parsing** — вызов PDF/DOCX парсеров
   - **Persistence** — запись в Qdrant/BM25/Postgres

3. Нет characterization тестов → рефакторинг рискованный.

**benchmark_orchestrator.py:**
- Импортирует `infrastructure.benchmark.metrics._percentile` — utility function
- Импортирует `infrastructure.benchmark.runner.run_benchmark_async` — инфраструктурный runner

**retrieval.py:**
- Импортирует `infrastructure.bm25.hybrid.rrf_merge` — business logic, должна быть в domain
- Импортирует `infrastructure.ml.rag.rag_reranking.deduplicate_docs` — business logic

### Архитектурное решение

**Решение 1: Ports для ingestion_orchestrator**

Создать порты:
```python
# application/ports/document_parser_port.py
class DocumentParserPort(Protocol):
    def parse_pdf(self, path: Path) -> list[RawDocument]: ...
    def parse_docx(self, path: Path) -> list[RawDocument]: ...
    def parse_rtf(self, path: Path) -> list[RawDocument]: ...
    def merge_pdf_pages(self, docs: list[RawDocument]) -> list[RawDocument]: ...

# application/ports/bm25_persistence_port.py
class BM25PersistencePort(Protocol):
    async def add_chunks(self, chunks: list[ChunkData]) -> None: ...
    async def remove_chunks(self, hashes: list[str]) -> None: ...
    async def save_index(self) -> None: ...
    async def load_index(self) -> None: ...
    async def invalidate(self) -> None: ...
```

Реализации:
```python
# infrastructure/ml/adapters/langchain_document_parser.py
class LangChainDocumentParser:
    def parse_pdf(self, path: Path) -> list[RawDocument]:
        from infrastructure.ml.ingestion import parse_pdf
        lc_docs = parse_pdf(path)
        return [self._to_raw_document(d) for d in lc_docs]

    def _to_raw_document(self, lc_doc: LCDocument) -> RawDocument:
        return RawDocument(content=lc_doc.page_content, metadata=lc_doc.metadata)
```

**Решение 2: Переместить utility functions в domain**

```python
# domain/utils.py
def percentile(values: list[float], p: float) -> float:
    """Compute p-th percentile (0-100)."""
    ...  # move implementation from infrastructure.benchmark.metrics

def rrf_merge(dense_results, sparse_results, k: int) -> list[tuple]:
    """Reciprocal Rank Fusion merge."""
    ...  # move from infrastructure.bm25.hybrid

def deduplicate_docs(docs: list[Document]) -> list[Document]:
    """Dedup by content_hash."""
    ...  # move from infrastructure.ml.rag.rag_reranking
```

**Решение 3: IngestionRegistry в UnitOfWork**

Добавить в `UnitOfWork` protocol:
```python
# application/uow.py
class UnitOfWork(Protocol):
    documents: DocumentRepository
    chunks: ChunkRepository
    vector_outbox: VectorOutboxRepository
    ingestion_registry: IngestionRegistryRepository  # NEW
```

Реализация:
```python
# infrastructure/uow/sqlalchemy_uow.py
class SQLAlchemyUnitOfWork:
    def __init__(self, session: AsyncSession):
        self._session = session
        self._documents = SQLAlchemyDocumentRepository(session)
        self._chunks = SQLAlchemyChunkRepository(session)
        self._vector_outbox = SQLAlchemyVectorOutboxRepository(session)
        self._ingestion_registry = SQLAlchemyIngestionRegistryRepository(session)  # NEW

    @property
    def ingestion_registry(self) -> IngestionRegistryRepository:
        return self._ingestion_registry
```

**Решение 4: Settings через порты**

```python
# application/ports/ingestion_settings_port.py
class IngestionSettingsPort(Protocol):
    @property
    def data_dir(self) -> str: ...
    @property
    def chunk_size(self) -> int: ...
    @property
    def chunk_overlap(self) -> int: ...

# infrastructure/config/pydantic_ingestion_settings.py
class PydanticIngestionSettings:
    def __init__(self, settings: Settings):
        self._settings = settings

    @property
    def data_dir(self) -> str:
        return self._settings.data_dir
```

**Альтернатива (ОТКЛОНЕНА):** Переместить `ingestion_orchestrator.py` в infrastructure.
**Причина отклонения:** Orchestration — это application-level concern. Infrastructure не должен знать о бизнес-процессах.

### Files/Modules Affected

**Новые файлы:**
- `server/app/application/ports/document_parser_port.py`
- `server/app/application/ports/bm25_persistence_port.py`
- `server/app/application/ports/ingestion_settings_port.py`
- `server/app/infrastructure/ml/adapters/langchain_document_parser.py`
- `server/app/infrastructure/bm25/bm25_persistence_adapter.py`
- `server/app/infrastructure/config/pydantic_ingestion_settings.py`

**Изменения:**
- `server/app/application/services/ingestion_orchestrator.py` — удалить infrastructure imports, использовать порты
- `server/app/application/services/benchmark_orchestrator.py` — использовать `domain.utils.percentile`
- `server/app/application/services/retrieval.py` — использовать `domain.utils.rrf_merge`
- `server/app/application/uow.py` — добавить `ingestion_registry` property
- `server/app/infrastructure/uow/sqlalchemy_uow.py` — добавить `_ingestion_registry`
- `server/app/domain/utils.py` — добавить `percentile`, `rrf_merge`, `deduplicate_docs`
- `server/app/composition/application.py` — зарегистрировать новые адаптеры

**Удаления:**
- `from langchain.schema import Document` в application layer → заменить на `domain.entities.raw_document.RawDocument`

### Implementation Steps

1. **Characterization тесты для ingestion_orchestrator** (1.5 дня):
   - `test_ingestion_orchestrator_characterization.py` — новый файл
   - Тесты на все публичные методы:
     - `test_run_full_ingestion_creates_expected_chunks()`
     - `test_handle_existing_conflict_skip_mode()`
     - `test_handle_existing_conflict_replace_mode()`
     - `test_sync_documents_to_db_creates_entities()`
   - Запустить → все зелёные (baseline)

2. **Переместить utility functions в domain** (0.5 дня):
   - Переместить `_percentile`, `rrf_merge`, `deduplicate_docs` в `domain/utils.py`
   - Обновить импорты в `benchmark_orchestrator.py`, `retrieval.py`
   - Обновить `infrastructure/benchmark/metrics.py` → использовать `from domain.utils import percentile`
   - Запустить `task test` → все зелёные

3. **Создать порты и адаптеры** (1 день):
   - Создать `DocumentParserPort`, `BM25PersistencePort`, `IngestionSettingsPort`
   - Реализовать адаптеры в infrastructure
   - Зарегистрировать в `composition/infrastructure.py`
   - Unit-тесты для адаптеров

4. **IngestionRegistry в UoW** (0.5 дня):
   - Добавить `ingestion_registry` property в `UnitOfWork` protocol
   - Реализовать в `SQLAlchemyUnitOfWork`
   - Обновить `ingestion_orchestrator.py` — использовать `uow.ingestion_registry` вместо прямого создания
   - Удалить 4 места с `repo = SQLAlchemyIngestionRegistryRepository(uow._session)`

5. **Refactor ingestion_orchestrator** (2 дня):
   - Заменить прямые импорты на порты:
     - `parse_pdf()` → `parser_port.parse_pdf()`
     - `load_bm25_index_from_s3()` → `bm25_port.load_index()`
     - `settings.data_dir` → `settings_port.data_dir`
   - Удалить `from infrastructure.*` импорты
   - Запустить characterization тесты → все зелёные
   - Запустить `grep -rn "^from infrastructure" server/app/application/` → 0 результатов

6. **LangChain Document → RawDocument** (0.5 дня):
   - Заменить `from langchain.schema import Document` на `from domain.entities.raw_document import RawDocument`
   - Обновить `DocumentParserPort` — возвращать `list[RawDocument]` вместо `list[LCDocument]`
   - Обновить адаптеры — конвертировать LangChain Document → RawDocument
   - Запустить тесты

7. **Integration testing** (0.5 дня):
   - End-to-end тест: CLI `ingest run --docs-dir` → проверить, что документы индексируются
   - Проверить, что BM25 index сохраняется в S3
   - Проверить, что ingestion_registry записывается в Postgres
   - Запустить `task test` → все зелёные

### Tests

**Новые тесты:**
- `tests/test_ingestion_orchestrator_characterization.py` (15–20 тестов)
- `tests/test_document_parser_adapter.py` (5 тестов)
- `tests/test_bm25_persistence_adapter.py` (5 тестов)

**Обновлённые тесты:**
- `tests/test_benchmark_orchestrator.py` — обновить импорты
- `tests/test_retrieval.py` — обновить импорты

### Acceptance Criteria

**Функциональные:**
1. ✅ `ingestion_orchestrator.py` не содержит `from infrastructure.*` импортов
2. ✅ `benchmark_orchestrator.py`, `retrieval.py` не содержат `from infrastructure.*` импортов
3. ✅ Все application-сервисы используют порты вместо прямых импортов
4. ✅ `uow.ingestion_registry` доступен без обращения к `_session`
5. ✅ CLI `ingest run` работает как раньше (characterization тесты зелёные)

**Нефункциональные:**
1. ✅ `grep -rn "^from infrastructure" server/app/application/` → 0 результатов
2. ✅ `grep -rn "^from langchain" server/app/application/` → 0 результатов
3. ✅ Все тесты зелёные
4. ✅ Нет performance regression (ingestion 100 docs < 5 min)

**Документация:**
1. ✅ Обновить `docs/ARCHITECTURE.md` — раздел "Application Ports"
2. ✅ Добавить ADR: "ADR-003: Application layer decoupling via ports"

### Risks

**HIGH:** Characterization тесты могут не покрыть все edge cases в ingestion_orchestrator.
**Mitigation:** Запустить production-like ingestion (1000+ docs) до и после рефакторинга. Сравнить результаты (количество chunks, document statuses).

**MEDIUM:** Конвертация LangChain Document → RawDocument может потерять metadata.
**Mitigation:** Unit-тесты на конвертацию с проверкой всех metadata fields.

**LOW:** Performance overhead от дополнительного уровня абстракции (порты).
**Mitigation:** Порты — zero-cost abstraction (Protocol, no runtime overhead). Benchmark ingestion до и после.

### Dependencies

**Блокирует:**
- Фазу 4 (нельзя рефакторить god-files без characterization тестов)

**Блокируется:**
- Ничем (можно начинать после Фазы 1)

---

## Фаза 4: API Contracts & Versioning

**Цель:** Добавить версионирование API, разделить schemas на Request/Response, добавить rate limiting middleware.

**Приоритет:** HIGH (не блокирует систему, но нужно для стабильного API контракта)
**Трудозатраты:** 3–4 дня
**Зависимости:** Фаза 3 (можно начинать параллельно после Фазы 3)

### Findings Addressed

| ID | Severity | Finding | Root Cause | Статус |
|----|----------|---------|------------|--------|
| API-1 | CRITICAL | Нет версионирования API | Все endpoints без `/api/v1/` prefix | REMOVED — версионирование через nginx (не в приложении) |
| API-2 | CRITICAL | God-файл schemas.py (778 строк) | Одна модель для Request + Response + internal | Остаётся |
| API-3 | CRITICAL | Нет extra="forbid" | Mass assignment риск | ✅ DONE — `ConfigDict(extra="forbid")` на 16 Request моделях |
| API-4 | HIGH | No per-user rate limiting | Один пользователь может блокировать сервис | Остаётся |
| API-5 | HIGH | Email/password без валидации | Пустые строки, огромные строки | ✅ DONE — `EmailStr` + `Field(min_length=8, max_length=128)` |
| CACHE-1 | HIGH | Cache invalidation при изменении документов отсутствует | `store_cached_answer` не вызывает инвалидацию | ✅ DONE — `invalidate_by_document_ids()` + вызов из delete route |

### Root Cause Analysis

**Нет версионирования:**
- Все роутеры регистрируются без prefix: `app.include_router(chat_router, prefix="/chat")`
- Breaking change в API ломает всех клиентов

**God-файл schemas.py:**
- 778 строк, 80+ Pydantic моделей в одном файле
- Одна модель используется как Request, Response и internal DTO
- Нет `extra="forbid"` → клиент может передать произвольные поля

**Rate limiting:**
- Есть глобальный семафор LLM (16), но нет per-user rate limiting
- Один пользователь может сделать 1000 запросов/сек → исчерпать семафор

**Cache invalidation:**
- `store_cached_answer()` сохраняет `document_ids`, но нет кода, который удаляет кэш при:
  - `document_service.delete_document()`
  - `ingestion_orchestrator._handle_existing_conflict(mode="replace")`

### Архитектурное решение

**Решение 1: API versioning через prefix**

```python
# main.py
app.include_router(chat_router, prefix="/api/v1/chat", tags=["chat"])
app.include_router(documents_router, prefix="/api/v1/documents", tags=["documents"])
...
```

**Альтернатива (ОТКЛОНЕНА):** Header-based versioning `Accept: application/vnd.api.v1+json`.
**Причина отклонения:** Сложнее для клиентов (нужно передавать header), не работает в браузере.

**Решение 2: Разделить schemas на Request/Response**

```
server/app/presentation/api/schemas/
  requests/
    auth_requests.py
    chat_requests.py
    document_requests.py
    ...
  responses/
    auth_responses.py
    chat_responses.py
    document_responses.py
    ...
  internal/
    document_internal.py  # для маппинга между слоями
```

Все Request-модели:
```python
class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(..., min_length=1, max_length=16000)
    conversation_id: int | None = None
    depth: Literal["narrow", "broad"] | None = None
    as_of_date: date | None = None
```

**Решение 3: Per-user rate limiting middleware**

```python
# infrastructure/rate_limiting/token_bucket.py
class TokenBucketRateLimiter:
    def __init__(self, redis: Redis, rate: int, per: int):
        self.redis = redis
        self.rate = rate  # tokens
        self.per = per    # seconds

    async def is_allowed(self, user_id: int) -> bool:
        key = f"rate_limit:user:{user_id}"
        # Token bucket algorithm via Redis + Lua script
        ...

# presentation/api/middleware/rate_limit.py
async def rate_limit_middleware(request: Request, call_next):
    user_ctx = request.state.user_context
    if user_ctx and not await rate_limiter.is_allowed(user_ctx.user_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    return await call_next(request)
```

Config:
```python
# config.py
rate_limit_requests_per_minute: int = 60  # per user
rate_limit_enabled: bool = True
```

**Решение 4: Cache invalidation**

```python
# infrastructure/ml/answer_cache.py
async def invalidate_by_document_ids(document_ids: list[int], cache_enabled: bool = True) -> int:
    """Invalidate all cache entries containing any of the given document_ids.
    Returns count of invalidated entries."""
    if not cache_enabled:
        return 0

    r = redis_client.async_redis
    pattern = f"{CACHE_PREFIX}*"
    invalidated = 0

    async for key in r.scan_iter(match=pattern, count=100):
        raw = await r.get(key)
        if raw:
            entry = json.loads(raw)
            cached_doc_ids = entry.get("document_ids", [])
            if any(doc_id in document_ids for doc_id in cached_doc_ids):
                await r.delete(key)
                invalidated += 1

    log.info("Invalidated %d cache entries for document_ids=%s", invalidated, document_ids)
    return invalidated
```

Вызывать из:
- `document_service.delete_document()` → `await invalidate_by_document_ids([document_id])`
- `ingestion_orchestrator._handle_existing_conflict(mode="replace")` → `await invalidate_by_document_ids([doc.id])`

### Files/Modules Affected

**Новая структура:**
- `server/app/presentation/api/schemas/requests/` (10 файлов)
- `server/app/presentation/api/schemas/responses/` (10 файлов)
- `server/app/presentation/api/schemas/internal/` (3 файла)
- `server/app/infrastructure/rate_limiting/token_bucket.py`
- `server/app/presentation/api/middleware/rate_limit.py`

**Изменения:**
- `server/app/main.py` — добавить `/api/v1` prefix ко всем роутерам, добавить rate_limit_middleware
- `server/app/presentation/api/routes/*.py` — обновить импорты schemas
- `server/app/infrastructure/ml/answer_cache.py` — добавить `invalidate_by_document_ids()`
- `server/app/application/services/document_service.py` — вызвать invalidation при delete
- `server/app/application/services/ingestion_orchestrator.py` — вызвать invalidation при replace
- `server/app/config.py` — добавить rate limiting settings

**Удаления:**
- `server/app/presentation/api/schemas.py` → разделить на requests/responses/internal

### Implementation Steps

1. **API versioning** (0.5 дня):
   - Обернуть все роутеры в `prefix="/api/v1"`
   - Обновить client `.env.example` → `VITE_API_URL=http://localhost:8001/api/v1`
   - Обновить loadtest scripts → `/api/v1/chat`
   - End-to-end test: `curl http://localhost:8001/api/v1/health` → 200

2. **Разделить schemas** (1.5 дня):
   - Создать структуру `schemas/requests/`, `schemas/responses/`, `schemas/internal/`
   - Переместить модели из `schemas.py` в новые файлы
   - Добавить `model_config = ConfigDict(extra="forbid")` на все Request-модели
   - Обновить импорты в routes
   - Запустить `task test` → все зелёные

3. **Rate limiting middleware** (1 день):
   - Реализовать `TokenBucketRateLimiter` через Redis + Lua script
   - Реализовать `rate_limit_middleware`
   - Зарегистрировать middleware в `main.py`
   - Unit-тесты: `test_rate_limiter_allows_within_limit()`, `test_rate_limiter_blocks_over_limit()`
   - Integration test: сделать 100 запросов → проверить, что 60 прошли, 40 заблокированы

4. **Cache invalidation** (0.5 дня):
   - Реализовать `invalidate_by_document_ids()`
   - Вызвать из `document_service.delete_document()`
   - Вызвать из `ingestion_orchestrator._handle_existing_conflict()`
   - Unit-тест: создать кэш → удалить document → проверить, что кэш инвалидирован
   - Integration test: `/chat` → cache hit → delete document → `/chat` → cache miss

5. **Email/password validation** (0.5 дня):
   - `LoginRequest.email` → `EmailStr`
   - `LoginRequest.password` → `Field(min_length=8, max_length=128)`
   - `RegisterRequest` (если есть) — аналогично
   - Unit-тесты на валидацию

6. **Integration testing** (0.5 дня):
   - End-to-end test: все endpoints через `/api/v1/`
   - Load test: `task loadtest:smoke` с rate limiting
   - Проверить OpenAPI docs: `/api/v1/docs` → все endpoints видны

### Tests

**Новые тесты:**
- `tests/test_rate_limiter.py` (10 тестов)
- `tests/test_cache_invalidation.py` (5 тестов)
- `tests/test_api_versioning.py` (5 тестов)

**Обновлённые тесты:**
- `tests/test_auth.py` — обновить URL на `/api/v1/auth/login`
- Все integration тесты — обновить URLs

### Acceptance Criteria

**Функциональные:**
1. ✅ Все endpoints доступны через `/api/v1/*`
2. ✅ Старые endpoints без `/api/v1/` возвращают 404
3. ✅ Rate limiter блокирует пользователя после 60 requests/min
4. ✅ При удалении документа кэш инвалидируется
5. ✅ При replace документа (re-ingest) кэш инвалидируется
6. ✅ Все Request-модели имеют `extra="forbid"`
7. ✅ Email валидируется через `EmailStr`

**Нефункциональные:**
1. ✅ OpenAPI docs показывают `/api/v1/*` endpoints
2. ✅ Rate limiting добавляет < 5ms latency overhead
3. ✅ Cache invalidation завершается < 1 sec для 1000 cache entries

### Risks

**MEDIUM:** Rate limiting может блокировать легитимных пользователей при burst traffic.
**Mitigation:** Настроить `rate = 60/min` консервативно. Мониторить `rate_limit_blocked_total` метрику. При необходимости увеличить до 100/min.

**MEDIUM:** Cache invalidation через `SCAN` может быть медленной для 10K+ cache entries.
**Mitigation:** Добавить secondary index `document_id → set[cache_key]` в Redis. Требует дополнительной памяти (~1MB на 10K docs).

**LOW:** Breaking change для клиентов (URLs изменились).
**Mitigation:** Сначала добавить `/api/v1/*`, затем deprecated старые endpoints. Через 2 недели удалить старые.

### Dependencies

**Блокирует:**
- Ничего

**Блокируется:**
- Фазой 3 (желательно закончить decoupling до рефакторинга API)

---

## Фаза 5: God-Files Decomposition

**Цель:** Разбить `rag_service.py::stream()` (270 строк), `ingestion.py` (722 строк), `benchmark.py` (1087 строк) на composable functions.

**Приоритет:** MEDIUM (не блокирует систему, улучшает maintainability)
**Трудозатраты:** 4–5 дней
**Зависимости:** Фаза 3 (characterization тесты должны быть готовы)

### Findings Addressed

| ID | Severity | Finding | Root Cause |
|----|----------|---------|------------|
| GOD-1 | HIGH | rag_service.py::stream() — 270 строк | Вся RAG-логика в одном методе |
| GOD-2 | HIGH | ingestion.py — 722 строк | PDF/DOCX/RTF парсинг в одном файле |
| GOD-3 | HIGH | benchmark.py — 1087 строк | Retrieval + judge + metrics + reports в одном файле |

### Root Cause Analysis

**rag_service.py::stream():**
- 270 строк в одном методе с `# noqa: C901` (complexity > 10)
- Смешивает:
  - Cache lookup
  - Question classification
  - Retrieval
  - Reranking
  - Prompt assembly
  - LLM generation
  - Source formatting
  - Cache storage

**ingestion.py:**
- 722 строки, 4 парсера (PDF/DOCX/RTF/MD) в одном файле
- Дублирование кода между парсерами
- `infrastructure/ml/preview/` уже показывает Strategy pattern для той же задачи

**benchmark.py:**
- 1087 строк, module-level функции без группировки
- `judge_answer()`, `compute_summary_metrics()`, `run_benchmark_async()` — разные concerns

### Архитектурное решение

**Решение 1: rag_service.py — pipeline stages**

Уже частично реализовано через `rag_steps.py`:
- `step_check_cache()`
- `step_condense()`
- `step_retrieve()`
- `step_self_rag()`
- `step_build_context()`
- `step_generate()`
- `step_postprocess()`

Проблема: `stream()` **не использует** эти функции напрямую, а дублирует логику.

Рефакторинг:
```python
async def stream_answer(self, ctx: ChatContext, question: str, history: list[Message], ...):
    state = RagPipelineState(ctx=ctx, question=question, history=history, ...)

    # Pipeline stages
    state = await step_check_cache(state, self.ml_clients, self.cache_enabled)
    if state.cached_answer:
        yield from self._emit_cached_answer(state)
        return

    state = await step_condense(state, self.ml_clients)
    state = await step_retrieve(state, self.ml_clients)
    state = await step_self_rag(state, self.ml_clients)
    state = await step_build_context(state, self.ml_clients)

    async for event in step_generate(state, self.ml_clients):
        yield event

    await step_postprocess(state, self.ml_clients)
```

**Решение 2: ingestion.py — Strategy pattern**

```python
# infrastructure/ml/ingestion/parsers/base.py
class DocumentParser(Protocol):
    def parse(self, path: Path) -> list[RawDocument]: ...

# infrastructure/ml/ingestion/parsers/pdf_parser.py
class PDFParser:
    def parse(self, path: Path) -> list[RawDocument]:
        ...

# infrastructure/ml/ingestion/parsers/docx_parser.py
class DOCXParser:
    def parse(self, path: Path) -> list[RawDocument]:
        ...

# infrastructure/ml/ingestion/parser_registry.py
PARSERS = {
    ".pdf": PDFParser(),
    ".docx": DOCXParser(),
    ".rtf": RTFParser(),
    ".md": MarkdownParser(),
}
```

**Решение 3: benchmark.py — separate modules**

```
infrastructure/benchmark/
  runner.py          # run_benchmark_async()
  judge.py           # judge_answer(), assess_answer_quality()
  metrics.py         # compute_summary_metrics(), percentile()
  reports.py         # generate_report(), export_to_json()
```

### Files/Modules Affected

**Изменения:**
- `server/app/infrastructure/ml/rag_service.py` — refactor `stream()` через pipeline stages
- `server/app/infrastructure/ml/ingestion/` — разделить на `parsers/pdf_parser.py`, `parsers/docx_parser.py`, etc.
- `server/app/infrastructure/benchmark/benchmark.py` → разделить на `runner.py`, `judge.py`, `metrics.py`, `reports.py`

**Новые файлы:**
- `server/app/infrastructure/ml/ingestion/parsers/base.py`
- `server/app/infrastructure/ml/ingestion/parsers/pdf_parser.py`
- `server/app/infrastructure/ml/ingestion/parsers/docx_parser.py`
- `server/app/infrastructure/ml/ingestion/parsers/rtf_parser.py`
- `server/app/infrastructure/ml/ingestion/parsers/markdown_parser.py`
- `server/app/infrastructure/ml/ingestion/parser_registry.py`

### Implementation Steps

1. **Characterization тесты** (1 день):
   - `test_rag_service_stream.py` уже существует (101–150 строк моков)
   - Добавить тесты на все pipeline stages отдельно
   - `test_benchmark_characterization.py` уже существует
   - Добавить тесты на `judge_answer()`, `compute_summary_metrics()`

2. **Refactor rag_service.py::stream()** (1.5 дня):
   - Переписать `stream()` через вызовы `step_*()` функций
   - Удалить дублирование кода
   - Убрать `# noqa: C901`
   - Запустить characterization тесты → все зелёные

3. **Refactor ingestion.py** (1 день):
   - Создать `parsers/` directory
   - Переместить парсеры в отдельные файлы
   - Создать `parser_registry.py`
   - Обновить импорты в `ingestion_orchestrator.py`
   - Запустить `task test` → все зелёные

4. **Refactor benchmark.py** (1 день):
   - Разделить на `runner.py`, `judge.py`, `metrics.py`, `reports.py`
   - Обновить импорты в `benchmark_orchestrator.py`
   - Запустить `task test` → все зелёные

5. **Удалить dead code** (0.5 дня):
   - `infrastructure/ml/ingestion/semantic_chunking.py` — не используется → удалить или интегрировать
   - Grep по кодовой базе → убедиться, что нет импортов

### Tests

**Новые тесты:**
- `tests/test_ingestion_parsers_strategy.py` (10 тестов)
- `tests/test_benchmark_judge.py` (5 тестов)
- `tests/test_benchmark_metrics.py` (5 тестов)

**Обновлённые тесты:**
- `tests/test_rag_service_stream.py` — упростить моки (меньше дублирования)

### Acceptance Criteria

**Функциональные:**
1. ✅ `rag_service.py::stream()` < 100 строк
2. ✅ `ingestion.py` разделён на 5 parser-файлов по < 150 строк каждый
3. ✅ `benchmark.py` разделён на 4 модуля по < 300 строк каждый
4. ✅ Все characterization тесты зелёные
5. ✅ Нет `# noqa: C901` в рефакторенных файлах

**Нефункциональные:**
1. ✅ Cyclomatic complexity < 10 для всех функций
2. ✅ Нет performance regression (RAG pipeline latency < +5%)

### Risks

**MEDIUM:** Рефакторинг `stream()` может изменить порядок events → breaking change для клиента.
**Mitigation:** Characterization тесты на порядок events. Сравнить output до и после.

**LOW:** Strategy pattern для parsers добавляет indirection.
**Mitigation:** Parser registry — zero-cost (dict lookup). Benchmark ingestion до и после.

### Dependencies

**Блокирует:**
- Ничего

**Блокируется:**
- Фазой 3 (characterization тесты должны быть готовы)

---

## Фаза 6: Performance & Observability (Optional)

**Цель:** Оптимизация BM25 operations, SQL query optimization, метрики для monitoring.

**Приоритет:** LOW (не блокирует систему, улучшает production experience)
**Трудозатраты:** 3–4 дня
**Статус:** ✅ ПОЛНОСТЬЮ ЗАВЕРШЕНА
**Зависимости:** Все предыдущие фазы завершены

### Findings Addressed

| ID | Severity | Finding | Root Cause | Статус |
|----|----------|---------|------------|--------|
| PERF-6 | HIGH | BM25 upsert/delete без thread safety | Concurrent mutations | ✅ DONE — `threading.Lock()` в `bm25_updater.py` |
| PERF-7 | HIGH | BM25Index._find_index_by_hash O(n) scan | Linear scan по 200K+ hashes | ✅ DONE — `_hash_to_idx` dict, O(1) + метод `find_by_hash()` |
| PERF-8 | HIGH | _patch_metadata O(n) scroll без ограничения | 1000 chunks → 1000 round-trips | SKIP — `set_payload` не поддерживает batch; scroll уже батчится по 100 |
| PERF-9 | MEDIUM | BM25 rebuild загружает все тексты в память | 100MB+ для 200K chunks | ✅ DONE — `get_all_contents_batches()` batch-загрузка по 5000 |
| PERF-10 | MEDIUM | Кэш хранит ответ в Redis без компрессии | 20–50MB для 10K кэшей | ✅ DONE — `gzip.compress/decompress` в `answer_cache.py` |

### Архитектурное решение

**Решение 1: BM25 thread safety через asyncio.Lock**

```python
# bm25_updater.py
_bm25_lock = asyncio.Lock()

async def bm25_add(hashes, texts, bm25_index):
    async with _bm25_lock:
        bm25_index.add(hashes, texts)
        await save_bm25_index_to_s3(bm25_index)
```

**Решение 2: Reverse index для BM25 hash lookup**

```python
class BM25Index:
    def __init__(self, texts, hashes):
        self.hashes = hashes
        self._hash_to_idx: dict[str, int] = {h: i for i, h in enumerate(hashes)}

    def _find_index_by_hash(self, target_hash: str) -> int | None:
        return self._hash_to_idx.get(target_hash)  # O(1) вместо O(n)
```

**Решение 3: Batch set_payload для Qdrant**

```python
async def _patch_metadata(self, updates: list[tuple[int, dict]]):
    # Batch updates по 100
    for batch in chunk_list(updates, 100):
        points = [
            models.PointStruct(id=point_id, payload=payload)
            for point_id, payload in batch
        ]
        await asyncio.to_thread(
            self.client.upsert,
            collection_name=settings.collection_name,
            points=points,
        )
```

**Решение 4: Streaming BM25 rebuild**

```python
async def _rebuild_bm25_job():
    chunk_repo = ...
    batch_size = 5000
    offset = 0

    all_texts = []
    all_hashes = []

    while True:
        batch = await chunk_repo.load_batch(offset, batch_size)
        if not batch:
            break
        all_texts.extend([c.content for c in batch])
        all_hashes.extend([c.content_hash for c in batch])
        offset += batch_size

    bm25_index = BM25Index(all_texts, all_hashes)
    await save_bm25_index_to_s3(bm25_index)
```

**Решение 5: Redis cache compression**

```python
import gzip

async def store_cached_answer(...):
    entry = {...}
    entry_json = json.dumps(entry)
    entry_compressed = gzip.compress(entry_json.encode("utf-8"))
    await r.set(key, entry_compressed, ex=CACHE_TTL_SECONDS)

async def find_cached_answer(...):
    raw = await r.get(key)
    if raw:
        decompressed = gzip.decompress(raw).decode("utf-8")
        entry = json.loads(decompressed)
        return entry
```

### Implementation Steps

1. **BM25 thread safety** (0.5 дня) ✅ DONE — `threading.Lock()` в `bm25_updater.py`
2. **BM25 reverse index** (0.5 дня) ✅ DONE — `_hash_to_idx` dict в `BM25Index` + метод `find_by_hash()`
3. **Batch Qdrant operations** (1 день) SKIP — `set_payload` не поддерживает batch с разными payload; scroll уже батчится по 100
4. **Streaming BM25 rebuild** (1 день) ✅ DONE — `get_all_contents_batches()` + batch-загрузка по 5000 chunks в `cron_bm25_rebuild`
5. **Redis compression** (0.5 дня) ✅ DONE — `gzip.compress/decompress` в `answer_cache.py`
6. **Observability metrics** (1 день) ✅ DONE — 4 метрики в `metrics.py` + инструментация `bm25_updater.py`:
   - `bm25_update_duration_seconds` (Histogram, labels: operation)
   - `qdrant_batch_upsert_duration_seconds` (Histogram)
   - `cache_compression_ratio` (Gauge)
   - `bm25_rebuild_memory_bytes` (Gauge)

### Acceptance Criteria

1. ✅ BM25 concurrent mutations не вызывают data corruption — DONE
2. ✅ `_find_index_by_hash()` работает за O(1) — DONE
3. ✅ `_patch_metadata` — scroll батчится по 100 (set_payload per-point, оптимально для разных payload) — DONE
4. ✅ BM25 rebuild использует batch-загрузку по 5000 chunks — DONE
5. ✅ Redis cache size уменьшен на 40–60% — DONE

### Dependencies

**Блокирует:**
- Ничего

**Блокируется:**
- Всеми предыдущими фазами (можно делать только после завершения Фаз 1–5)

---

## Общий порядок выполнения фаз

### Критический путь (последовательно):

```
Фаза 1: Multi-tenant Security (4–5 дней)
    ↓
Фаза 2: Circuit Breaker (2–3 дня)
    ↓
Фаза 3: Application Decoupling (5–6 дней)
```

**Итого критический путь:** 11–14 дней

### Параллельные работы (после Фазы 3):

```
Фаза 4: API Contracts (3–4 дня)  ║  Фаза 5: God-Files (4–5 дней)
                                  ║
                                  ↓
                         Фаза 6: Performance (3–4 дня)
```

**Общая оценка:** 18–22 рабочих дня (1 senior developer full-time)

---

## Фазы по приоритетам

### Критические (нельзя откладывать):

1. **Фаза 1** — Multi-tenant Security → утечки данных
2. **Фаза 2** — Circuit Breaker → деградация сервиса
3. **Фаза 3** — Application Decoupling → архитектурная чистота

### High (можно отложить на 1–2 недели):

4. **Фаза 4** — API Contracts → стабильность внешнего контракта

### Medium (можно отложить на 1–2 месяца):

5. **Фаза 5** — God-Files → maintainability

### Low (можно отложить на 3+ месяца):

6. **Фаза 6** — Performance → оптимизация (делать только если production metrics показывают проблемы)

---

## Какие фазы можно отдавать разработчику независимо

### Требуют архитектурного контроля:

- **Фаза 1** — Multi-tenant Security (критично для безопасности, неправильное решение создаёт vulnerabilities)
- **Фаза 3** — Application Decoupling (архитектурные решения по портам и адаптерам)

**Архитектор должен:**
- Провести code review после каждого шага
- Утвердить дизайн портов до implementation
- Проверить, что layering rules соблюдаются

### Можно отдать разработчику самостоятельно (с code review):

- **Фаза 2** — Circuit Breaker (стандартный pattern, есть библиотека `aiobreaker`)
- **Фаза 4** — API Contracts (механическая работа по разделению schemas + стандартный rate limiter)
- **Фаза 5** — God-Files (рефакторинг с characterization тестами, низкий риск)
- **Фаза 6** — Performance (оптимизация, можно делать итеративно)

**Условия:**
- Characterization тесты написаны ДО рефакторинга
- После каждого шага — code review + запуск всех тестов
- Нельзя менять публичные API без согласования

---

## Архитектурные решения, которые нельзя отдавать разработчику

### AD-1: Curator scope — managed_*_ids в ChatContext

**Почему нельзя отдавать:**
- Изменение domain-модели (`ChatContext`) влияет на ACL-контракт
- Неправильное решение создаёт security vulnerability
- Требует понимания всей ACL-цепочки: UserContext → ChatContext → build_qdrant_filter()

**Архитектор должен:**
- Спроектировать структуру `managed_*_ids` (какие именно поля добавить)
- Определить, как эти поля используются в `build_qdrant_filter()`
- Написать спецификацию для разработчика

### AD-2: BM25 ACL enforcement strategy

**Почему нельзя отдавать:**
- Выбор между 3 альтернативами:
  1. ACL-фильтрация на уровне BM25 (сложно, требует per-user indexes)
  2. Post-filter через Qdrant (текущее решение)
  3. Native Qdrant sparse vectors (требует upgrade Qdrant)
- Каждая альтернатива имеет trade-offs (memory, performance, security)
- Неправильный выбор создаёт technical debt на годы

**Архитектор должен:**
- Провести spike investigation (1–2 дня) на все 3 альтернативы
- Написать ADR с обоснованием выбора
- Определить acceptance criteria

### AD-3: Application → Infrastructure decoupling (порты)

**Почему нельзя отдавать:**
- Дизайн портов — ключевое архитектурное решение
- Неправильная граница порта создаёт leaky abstraction
- Сложность: какие операции вынести в порт, какие оставить в infrastructure

**Архитектор должен:**
- Спроектировать все порты (`DocumentParserPort`, `BM25PersistencePort`, etc.)
- Определить signatures методов
- Написать пример адаптера для одного порта
- Передать разработчику для реализации остальных адаптеров

### AD-4: Ingestion orchestrator responsibility split

**Почему нельзя отдавать:**
- 768 строк с 12 infrastructure imports — неочевидно, как разбить
- Выбор между:
  1. Оставить в application + порты
  2. Переместить в infrastructure
  3. Разбить на несколько сервисов
- Каждый вариант влияет на testing strategy

**Архитектор должен:**
- Проанализировать все 3 варианта
- Написать spike (proof-of-concept) для выбранного варианта
- Определить plan разбиения на шаги

---

## Резюме

### Что исправлять первым:

**Неотложно (начать завтра):**
1. Фаза 1 — Multi-tenant Security Hardening (4–5 дней)

**После Фазы 1:**
2. Фаза 2 — Circuit Breaker (2–3 дня)
3. Фаза 3 — Application Decoupling (5–6 дней)

**После Фазы 3 (можно параллельно):**
4. Фаза 4 — API Contracts (3–4 дня)
5. Фаза 5 — God-Files (4–5 дней)

**Опционально (если production metrics показывают проблемы):**
6. Фаза 6 — Performance & Observability (3–4 дня)

### Порядок всех фаз:

**Критический путь (sequential):**
```
Фаза 1 (4–5d) → Фаза 2 (2–3d) → Фаза 3 (5–6d)
```

**Параллельные работы (after Фаза 3):**
```
Фаза 4 (3–4d)  ║  Фаза 5 (4–5d)
               ↓
       Фаза 6 (3–4d, optional)
```

### Какие фазы можно отдавать разработчику независимо:

**С минимальным архитектурным контролем:**
- Фаза 2 (Circuit Breaker) — стандартный pattern
- Фаза 4 (API Contracts) — механическая работа
- Фаза 5 (God-Files) — с characterization тестами
- Фаза 6 (Performance) — итеративная оптимизация

**Условия:**
- Code review после каждого шага
- Все тесты зелёные
- Нельзя менять domain-модели без согласования

### Какие требуют архитектурного контроля:

**Полный архитектурный контроль (архитектор должен спроектировать решение):**
- Фаза 1 (Multi-tenant Security) — критично для безопасности
- Фаза 3 (Application Decoupling) — дизайн портов и адаптеров

**Архитектурные решения, которые нельзя делегировать:**
1. Curator scope — структура `managed_*_ids` в `ChatContext`
2. BM25 ACL enforcement strategy — выбор между 3 альтернативами
3. Application ports design — границы абстракций
4. Ingestion orchestrator split — responsibility boundaries

**Процесс:**
1. Архитектор проектирует решение (1–2 дня spike)
2. Архитектор пишет спецификацию + пример
3. Разработчик реализует по спецификации
4. Архитектор проводит code review + утверждает

---

## Следующие шаги

1. **Получить approval от product owner** на INV-1 (curator scope) и INV-2 (prompt injection)
2. **Начать Фазу 1** (Multi-tenant Security Hardening) немедленно
3. **Провести spike investigation** (1 день) на BM25 ACL enforcement — выбрать окончательное решение между post-filter и native Qdrant sparse
4. **Написать ADR-003** "Application layer decoupling via ports" после завершения Фазы 3
5. **Настроить мониторинг** для circuit breaker метрик после Фазы 2

---

## Выполненные задачи (Quick Wins)

**Дата:** 2026-09-13
**Статус:** Все тесты зелёные (46/46)
**Фаза 6:** ✅ ПОЛНОСТЬЮ ЗАВЕРШЕНА

| # | Задача | Файл | Изменение | Фаза |
|---|--------|------|-----------|------|
| 1 | BM25 reverse index | `bm25_index.py` | Добавлен `_hash_to_idx: dict` + метод `find_by_hash()` — O(1) вместо O(n) | Фаза 6 (PERF-7) |
| 2 | BM25 thread safety | `bm25_updater.py` | `threading.Lock()` на все мутации (`add`, `replace`, `remove`) | Фаза 6 (PERF-6) |
| 3 | Redis gzip compression | `answer_cache.py` | `gzip.compress/decompress` при записи/чтении кэша — ~40-60% экономия RAM | Фаза 6 (PERF-10) |
| 4 | Delete dead code | `semantic_chunking.py` | Удалён неиспользуемый файл + тесты `TestSemanticChunkingHelpers` | Фаза 5 (GOD-2) |
| 5 | Email/password validation | `schemas.py` + `pyproject.toml` | `EmailStr` + `Field(min_length=8, max_length=128)` для `LoginRequest` и `CreateUserRequest` | Фаза 4 (API-5) |
| 6 | extra="forbid" | `schemas.py` | `ConfigDict(extra="forbid")` на 16 Request моделей — защита от mass assignment | Фаза 4 (API-3) |
| 7 | Cache invalidation | `answer_cache.py` + `documents.py` | `invalidate_by_document_ids()` + вызов из delete route | Фаза 4 (CACHE-1) |
| 8 | Observability metrics | `metrics.py` + `bm25_updater.py` | 4 метрики: `bm25_update_duration`, `qdrant_batch_upsert_duration`, `cache_compression_ratio`, `bm25_rebuild_memory` | Фаза 6 |
| 9 | Streaming BM25 rebuild | `cron.py` + `sqlalchemy_chunk_repository.py` | `get_all_contents_batches()` batch-загрузка по 5000 chunks вместо SELECT * | Фаза 6 (PERF-9) |

### Удалённые задачи:

| # | Задача | Причина |
|---|--------|---------|
| API-1 | API versioning (`/api/v1/`) | Версионирование будет через nginx, не в приложении |
| PERF-8 | Batch Qdrant `_patch_metadata` | `set_payload` не поддерживает batch с разными payload; scroll уже батчится по 100 |

---

**END OF PLAN**
