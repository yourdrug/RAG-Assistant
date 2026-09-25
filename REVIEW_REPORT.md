# Отчёт по ревью — Весь репозиторий — 2026-09-17

## Executive summary

- **4 критичных**, **15 высоких**, **28 средних**, **18 низких**, **20 информационных** находок
- **Главные риски проекта:**
  1. **Утечка данных между тенантами** — соседние чанки загружаются без ACL-фильтра (`rag_postprocess.py`), а `delete_document`/`rename_document` хардкодят `UserKind.INTERNAL` для всех пользователей, что может привести к обходу ACL при расширении ролей.
  2. **Отсутствие E2E-тестов и FakeUnitOfWork** —resentation-слой (routes, middleware, schemas, auth flow) вообще не тестирован; fake UoW не обеспечивает изоляцию транзакций, что маскирует баги в транзакционной логике.
  3. **RAG-пайплайн без score threshold** — все результаты Qdrant (включая нерелевантные) передаются в LLM, расширяя attack surface для prompt injection и ухудшая качество ответов.
  4. **Ресурсные exhausted при dependency failures** — зависший Ollama держит generation semaphore навсегда, синхронный QdrantClient в health probe блокирует event loop, нет per-tenant ingestion limits.

## Топ-5 неотложных действий

1. **[CRITICAL] Исправить ACL в enrich_with_neighbors()** — передать `access_filter` в `rag_postprocess.py:129-205` и фильтровать каждый neighbor перед включением в контекст LLM. Одновременно: передавать реальный `user_kind` в `document_command_service.py:230,288` вместо хардкода `INTERNAL`. Затрагивает: multi-tenant security + RAG pipeline.

2. **[CRITICAL] Добавить score_threshold в Qdrant dense search** — установить минимальный порог (0.3–0.5) в `rag_retrieval.py:102-110`. Пустой результат после порога должен возвращать явный "не найдено" вместо передачи мусора в LLM. Одновременно: вернуть пустой список sources в `_filter_sources_by_min_score` вместо fallback на худший источник.

3. **[CRITICAL] Создать E2E-тесты для критичных эндпоинтов** — добавить smoke-тесты через FastAPI TestClient для POST /chat, POST /documents, POST /auth/login, GET /documents. Исправить FakeUnitOfWorkFactory для создания нового экземпляра на каждый вызов. Затрагивает: test quality + API contracts.

4. **[HIGH] Исправить sync QdrantClient в health probe** — обернуть в `asyncio.to_thread()` или использовать async-клиент. Переиспользовать singleton-клиент вместо создания нового экземпляра на каждый вызов. Добавить `request_timeout=120` в ChatOllama и `asyncio.wait_for(timeout=180)` в `step_generate`. Затрагивает: performance + reliability.

5. **[HIGH] Добавить Idempotency-Key для write-эндпоинтов** — приоритетные: POST /documents, POST /chat, POST /benchmark. Redis-backed store для хранения результатов по (user_id, key). Одновременно: создать response-схемы для mutation-эндпоинтов (все ad-hoc dict). Затрагивает: API contracts.

---

## CRITICAL

### C-001 — ACL bypass в enrichment соседних чанков
- **ID:** C-001
- **Category:** Multi-tenant security / RAG pipeline
- **Severity:** CRITICAL
- **Confidence:** 0.90
- **Title:** Соседние чанки загружаются без ACL-фильтра
- **Location:** `server/app/infrastructure/ml/rag/rag_postprocess.py:129-205`
- **Evidence:** `enrich_with_neighbors()` вызывает `chunk_search.get_neighbors()` / `get_table_batches()` без передачи `access_filter`. Метод не принимает параметров ACL.
- **Проблема:** Если adjacent chunk принадлежит другому tenant-у (через миграцию данных или ошибку OWNER_ID), он попадёт в контекст LLM без проверки ACL.
- **Риск:** Data leak между tenant-ами через neighbor-чанки. LLM получает чужой контекст и может процитировать его.
- **Рекомендация:** Передать `access_filter`/`visibility_conditions` в `enrich_with_neighbors()` и проверять каждый neighbor через ACL. Или использовать Qdrant scroll с ACL-фильтром. **Затрагивает 2 области:** RAG pipeline + multi-tenant security.

### C-002 — Score threshold отсутствует в Qdrant dense search
- **ID:** C-002
- **Category:** RAG pipeline quality
- **Severity:** CRITICAL
- **Confidence:** 0.95
- **Title:** Все результаты Qdrant (включая нерелевантные) передаются в LLM
- **Location:** `server/app/infrastructure/ml/rag/rag_retrieval.py:102-110`
- **Evidence:** `qdrant_dense_search()` использует только `limit=k` без `score_threshold`. Все результаты с косинусным расстоянием > 0 возвращаются.
- **Проблема:** В context LLM попадают нерелевантные чанки (score ~0.01), ухудшающие качество ответа и расширяющие attack surface для prompt injection.
- **Риск:** Hallucination, prompt injection через нерелевантный контекст, bad UX.
- **Рекомендация:** Добавить `score_threshold` (configurable, 0.3–0.5) в `client.search()`. При пустом результате после порога — вернуть явный "не найдено".

### C-003 — Отсутствие E2E тестов для presentation-слоя
- **ID:** C-003
- **Category:** Test quality
- **Severity:** CRITICAL
- **Confidence:** 0.95
- **Title:** Ни один HTTP-эндпоинт не тестирован через TestClient
- **Location:** `server/tests/` — отсутствуют `test_api_*.py` файлы
- **Evidence:** Все 74 тестовых файла работают на уровне Python-функций. Ни одного `TestClient` вызова. Middleware (auth, rate-limit, error handlers), request/response schemas, HTTP status codes не тестированы.
- **Проблема:** Регрессия в presentation-слое (routes, middleware, schemas) не будет поймана до деплоя. Баги в auth flow, CORS, error responses доступны только в production.
- **Риск:** Production-инциденты с авторизацией, невалидными ответами, кассовыми ошибками.
- **Рекомендация:** Добавить smoke-тесты для критичных эндпоинтов: POST /chat, POST /documents, GET /documents, POST /auth/login через FastAPI TestClient с FakeUnitOfWork.

### C-004 — FakeUnitOfWorkFactory не обеспечивает изоляцию транзакций
- **ID:** C-004
- **Category:** Test quality
- **Severity:** CRITICAL
- **Confidence:** 0.85
- **Title:** FakeUnitOfWorkFactory.create() всегда возвращает один экземпляр
- **Location:** `server/tests/fakes.py:824-832`
- **Evidence:** `FakeUnitOfWorkFactory.create()` всегда yield'ит один и тот же `_uow`. Реальный UoWFactory создаёт новую SQLAlchemy session на каждый вызов.
- **Проблема:** Тесты не могут проверить поведение между транзакциями (read-after-commit, optimistic locking, concurrent UoW). Rollback помечает флаг, но данные не откатываются (line 817-821).
- **Риск:** Баги в транзакционной логике маскируются. Продакшен-инциденты с потерей данных или race conditions.
- **Рекомендация:** FakeUnitOfWorkFactory.create() должен создавать новый FakeUnitOfWork на каждый вызов. Добавить snapshot mechanism для rollback.

---

## HIGH

### H-001 — Hardcoded UserKind.INTERNAL в document write operations
- **ID:** H-001
- **Category:** Multi-tenant security
- **Severity:** HIGH
- **Confidence:** 0.80
- **Title:** delete_document/rename_document игнорируют реальный user_kind
- **Location:** `server/app/application/services/document_command_service.py:230, 288`
- **Evidence:** Контекст пользователя строится с `UserKind.INTERNAL` независимо от реального kind: `ctx = await self._user_ctx_factory.build(uow, user_id, UserKind.INTERNAL, user_role)`
- **Проблема:** Если CLIENT получит Capability.DOCUMENTS_MANAGE, его контекст будет INTERNAL, позволяя доступ к INTERNAL_PRIVATE документам других пользователей.
- **Риск:** Privilege escalation при расширении ролей. BOLA/IDOR через неправильный user_kind.
- **Рекомендация:** Передавать реальный `user_kind`. Или добавить assertion что `user_kind == INTERNAL` на входе.

### H-002 — Admin list_documents возвращает метаданные CLIENT_PRIVATE документов
- **ID:** H-002
- **Category:** Multi-tenant security
- **Severity:** HIGH
- **Confidence:** 0.85
- **Title:** ADMIN видит metadata (owner_id, visibility) CLIENT_PRIVATE документов
- **Location:** `server/app/application/services/document_query_service.py:54-56`
- **Evidence:** Для ADMIN вызывается `list_all()`, который возвращает ВСЕ документы с полной metadata. `in_search_scope=False` помечен, но metadata видна.
- **Проблема:** ADMIN получает информацию о клиентах и их документах через list_documents.
- **Риск:** Информационная утечка о клиентах, compliance нарушение (GDPR/персональные данные).
- **Рекомендация:** Разделить listing на "admin view" и "search scope". Или добавить флаг `admin_view=true`.

### H-003 — Нет Idempotency-Key для write-операций
- **ID:** H-003
- **Category:** API contracts
- **Severity:** HIGH
- **Confidence:** 0.90
- **Title:** Ни один write-эндпоинт не поддерживает Idempotency-Key
- **Location:** `server/app/presentation/api/routes/` (все write-эндпоинты)
- **Evidence:** POST /documents, POST /chat, POST /benchmark не имеют механизма идемпотентности. При network retry создаются дубликаты.
- **Проблема:** Двойная загрузка и индексация документов, двойной платный вызов LLM, дублирование benchmark runs.
- **Риск:** Финансовые потери (LLM billing), переполнение storage/Qdrant.
- **Рекомендация:** Добавить Redis-backed Idempotency-Key для POST /documents, POST /chat, POST /benchmark.

### H-004 — Untyped response модели для mutation-эндпоинтов
- **ID:** H-004
- **Category:** API contracts
- **Severity:** HIGH
- **Confidence:** 0.90
- **Title:** 12+ mutation-эндпоинтов возвращают ad-hoc dict без Pydantic-схемы
- **Location:** `server/app/presentation/api/routes/auth.py:101`, `groups.py:71,87`, `curators.py:34,54,74,94`, `api_keys.py:80`, `admin_act_versions.py:82`, `admin_quality.py:403`, `benchmark_admin.py:142,189,307`
- **Evidence:** toggle_user_active возвращает `{"id": ..., "is_active": ...}` вместо response_model.
- **Проблема:** OpenAPI-схема не отражает реальность. Нет валидации типов на сервере. Потребители API получают untyped responses.
- **Риск:** Несовместимость клиентов при рефакторинге. Невозможность code generation.
- **Рекомендация:** Создать response-схемы: UserToggleResponse, GroupMemberResponse, CuratorAssignResponse и т.д.

### H-005 — Circuit breaker check_open race condition
- **ID:** H-005
- **Category:** Resilience / RAG pipeline
- **Severity:** HIGH
- **Confidence:** 0.80
- **Title:** check_open() модифицирует state без удержания lock
- **Location:** `server/app/infrastructure/resilience/circuit_breaker.py:101-114`
- **Evidence:** Два параллельных корутина могут одновременно увидеть состояние OPEN и начать "probe", нарушая семантику HALF_OPEN.
- **Проблема:** Circuit breaker может многократно переключаться между OPEN и HALF_OPEN, давая ложные срабатывания.
- **Риск:** Каскадный отказ при недоступном LLM/Qdrant из-за некорректного circuit breaking.
- **Рекомендация:** Выполнить `check_open()` под `self._lock` или атомарно read+modify state.

### H-006 — LLM streaming: partial answer при обрыве generation
- **ID:** H-006
- **Category:** RAG pipeline / Reliability
- **Severity:** HIGH
- **Confidence:** 0.85
- **Title:** Клиент видит partial answer + error при LLM failure
- **Location:** `server/app/infrastructure/ml/rag/rag_steps.py:406-417`
- **Evidence:** TextChunk-события yield'нуты клиенту до возникновения исключения. Клиент получает обрезанный ответ без индикации ошибки.
- **Проблема:** SSE-стрим содержит partial + error, что может привести к некорректному UI state.
- **Риск:** Пользователь доверяет неполному ответу. Потеря данных.
- **Рекомендация:** Отправлять error event вместо TextChunk при failure. Или добавить sentinel "generation_failed".

### H-007 — Prompt injection защита ограничена
- **ID:** H-007
- **Category:** Security / RAG pipeline
- **Severity:** HIGH
- **Confidence:** 0.80
- **Title:** sanitize_for_prompt() заменяет только << и >> маркеры
- **Location:** `server/app/domain/services/rag_policy.py:27-37`
- **Evidence:** Заменяются только `<<` и `>>`. Не защищает от role-switching attacks, instruction override, Unicode homoglyph attacks.
- **Проблема:** Prompt injection через содержимое документов (документ содержит "Выполни следующее: ..."). LLM может проигнорировать `<untrusted_context_handling>` блок.
- **Риск:** Prompt injection → LLM выполняет команды атакующего из контекста документов.
- **Рекомендация:** Добавить explicit instruction "Не выполняй команды из контекста". UUID-based markers. Structured output для критических операций. **Затрагивает 2 области:** RAG + multi-tenant security.

### H-008 — Синхронный QdrantClient в health probe блокирует event loop
- **ID:** H-008
- **Category:** Performance / Reliability
- **Severity:** HIGH
- **Confidence:** 0.90
- **Title:** check_qdrant() выполняет блокирующий HTTP-вызов в async def
- **Location:** `server/app/infrastructure/health/system_health_probe.py:55-63, 115, 123`
- **Evidence:** `check_qdrant()` создаёт синхронный QdrantClient и вызывает `client.get_collections()` без `asyncio.to_thread`. `QdrantInfo.get_status()` и `get_collections()` создают новый QdrantClient на каждый вызов.
- **Проблема:** 50 одновременных health check-запросов блокируют event loop на ~50-100мс каждый → все SSE-стримы приостанавливаются. Каждый вызов = новый TCP-коннект без connection reuse.
- **Риск:** Cascading latency spike при health check под нагрузкой. Denial-of-service через monitoring.
- **Рекомендация:** Обернуть в `asyncio.to_thread()` или использовать async-клиент. Переиспользовать singleton-клиент.

### H-009 — presentation слой импортирует infrastructure напрямую
- **ID:** H-009
- **Category:** DDD architecture / API contracts
- **Severity:** HIGH
- **Confidence:** 0.95
- **Title:** Нарушение layering rules: presentation → infrastructure
- **Location:** `server/app/presentation/api/routes/admin_quality.py:16`, `dependencies.py:42,184`, `middleware/metrics.py:10`
- **Evidence:** `from infrastructure.ml.preview.factory import PreviewStrategyFactory`, `from infrastructure.auth.api_key_provider import ApiKeyProvider`, `from infrastructure.benchmark.benchmark_history_adapter import BenchmarkHistoryAdapter`
- **Проблема:** Presentation импортирует infrastructure, минуя application ports. DI-обёртки создают infrastructure-объекты напрямую (benchmark_history_adapter — на каждый запрос).
- **Риск:** Нарушение архитектурных границ. Трудно подменить implementation в тестах. Tight coupling.
- **Рекомендация:** Создать application-level port для PreviewStrategy, HistoryAdapter. BenchmarkHistoryAdapter перенести в Container.

### H-010 — db_password передаётся как голый dict
- **ID:** H-010
- **Category:** DI container / Security
- **Severity:** HIGH
- **Confidence:** 0.80
- **Title:** db_password виден в dict, передаваемом PostgresOutboxListener
- **Location:** `server/app/composition/infrastructure.py:207-215`
- **Evidence:** PostgresOutboxListener получает db_config dict с `db_password=app_settings.db_password` в открытом виде. При ошибке подключения — пароль в traceback/логах.
- **Проблема:** Утечка пароля БД в логи/traceback при ошибке подключения outbox_listener.
- **Риск:** Компрометация базы данных.
- **Рекомендация:** Использовать DSN-строку вместо dict с паролем. Не передавать password отдельным полем.

### H-011 — benchmark_admin compare_runs: ValueError без обработки
- **ID:** H-011
- **Category:** API contracts
- **Severity:** HIGH
- **Confidence:** 0.90
- **Title:** Необработанный ValueError при парсинге run IDs
- **Location:** `server/app/presentation/api/routes/benchmark_admin.py:377`
- **Evidence:** `id_list = [int(x.strip()) for x in ids.split(",") if x.strip()]` — если ids="1,abc,3", будет 500 Internal Server Error.
- **Проблема:** Необработанное исключение приведёт к 500 ошибке с fallback handler, может утечь traceback в debug-режиме.
- **Риск:** Неинформативная ошибка для клиента. Potential information leak.
- **Рекомендация:** Обернуть в try/except ValueError → HTTPException(422).

### H-012 — admin_quality diagnose_document: HTTPException(500)
- **ID:** H-012
- **Category:** API contracts
- **Severity:** HIGH
- **Confidence:** 0.85
- **Title:** Серверная ошибка возвращается клиенту через HTTPException
- **Location:** `server/app/presentation/api/routes/admin_quality.py:101`
- **Evidence:** `raise HTTPException(status_code=500, detail="Failed to diagnose document")` — минует centralized exception handler.
- **Проблема:** HTTPException минует exception_handlers для ServerException и не логирует exc_info автоматически.
- **Риск:** Клиент не может дифференцировать ошибку. Отсутствие логирования.
- **Рекомендация:** Выбрасывать ServerException или DBError вместо HTTPException.

### H-013 — ApplicationContainer не имеет validate()
- **ID:** H-013
- **Category:** DI container
- **Severity:** HIGH
- **Confidence:** 0.75
- **Title:** 22+ сервисных поля без проверки инициализации
- **Location:** `server/app/composition/application.py:63-92`
- **Evidence:** ApplicationContainer содержит 22+ сервисных поля, инициализируемых в init(), но нет метода validate(). InfrastructureContainer имеет validate().
- **Проблема:** Если один из сервисов пропущен при добавлении нового поля, ошибка всплывёт только в рантайме из Depends-обёртки.
- **Риск:** Ошибки при старте, не определённые на этапе инициализации.
- **Рекомендация:** Добавить validate() аналогично InfrastructureContainer.

### H-014 — 20+ application-сервисов без unit-тестов
- **ID:** H-014
- **Category:** Test quality
- **Severity:** HIGH
- **Confidence:** 0.95
- **Title:** ConversationService, GroupService, AssignmentService, AuthService и другие не протестированы
- **Location:** `server/app/application/services/` — 30 файлов, тесты есть только для ~6
- **Evidence:** Без тестов: conversation_service, group_service, assignment_service, config_service, search_service, health_service, auth_service, document_command_service и другие.
- **Проблема:** Бизнес-логика application-слоя не застрахована от регрессий.
- **Риск:** Security-critical логика (assignment, auth) может сломаться незамеченной.
- **Рекомендация:** Приоритет: chat_service (core), assignment_service (security), config_service (hot-reload).

### H-015 — Нет fallback-тестов при infrastructure failures
- **ID:** H-015
- **Category:** Test quality
- **Severity:** HIGH
- **Confidence:** 0.85
- **Title:** RAG-тесты покрывают только happy path
- **Location:** `server/tests/test_rag_service_stream.py`, `test_rag_logic.py`
- **Evidence:** Нет тестов на: LLM timeout → graceful degradation, Qdrant connection error → empty retrieval, BM25 load failure → fallback dense-only, Embedding service down → error propagation.
- **Проблема:** В production при падении LLM/Qdrant поведение неизвестно.
- **РISK:** Unhandled exceptions, partial responses, или data leak в production.
- **Рекомендация:** Добавить тесты с side_effect=TimeoutError/ConnectionError на моках.

---

## MEDIUM

### M-001 — UserContext: frozen dataclass с мутабельными list-полями
- **ID:** M-001
- **Category:** DDD architecture
- **Severity:** MEDIUM
- **Confidence:** 0.90
- **Location:** `server/app/domain/value_objects/user_context.py:15-18`
- **Проблема:** `@dataclass(frozen=True)` с `group_ids: list[int]` — frozen запрещает присваивание, но `user_ctx.group_ids.append(999)` работает молча.
- **Риск:** Нарушение контракта VO. Cache hash() скомпрометирован при мутации.
- **Рекомендация:** Заменить list на tuple (как в CuratorScope).

### M-002 — Conversation: анемичная модель (aggregate root без поведения)
- **ID:** M-002
- **Category:** DDD architecture
- **Severity:** MEDIUM
- **Confidence:** 0.80
- **Location:** `server/app/domain/entities/conversation.py:16-21`
- **Проблема:** Conversation — aggregate root без методов. Ownership check в ConversationService, не в entity.
- **Риск:** При добавлении use-case разработчик может забыть о проверке ACL.
- **Рекомендация:** Перенести `can_be_viewed_by()` в Conversation entity.

### M-003 — DocumentService facade: утечка приватных полей под-сервисов
- **ID:** M-003
- **Category:** DDD architecture
- **Severity:** MEDIUM
- **Confidence:** 0.85
- **Location:** `server/app/application/services/document_service.py:45-74`
- **Проблема:** 8 `@property` возвращают `self._cmd._uow_factory` — прямой доступ к приватным атрибутам.
- **Риск:** Переименование внутренних пол сломает внешних потребителей.
- **Рекомендация:** Передавать зависимости напрямую через конструктор.

### M-004 — BM25IndexAdapter создаётся дважды
- **ID:** M-004
- **Category:** DI container
- **Severity:** MEDIUM
- **Confidence:** 0.75
- **Location:** `server/app/composition/application.py:190, 198`
- **Проблема:** Два разных экземпляра одного адаптера для document_service и chunk_service.
- **Риск:** Избыточное потребление памяти, путаница при рефакторинге.
- **Рекомендация:** Вынести в переменную и переиспользовать.

### M-005 — Auth toggle_user_active: is_active как Query Parameter
- **ID:** M-005
- **Category:** API contracts
- **Severity:** MEDIUM
- **Confidence:** 0.85
- **Location:** `server/app/presentation/api/routes/auth.py:94`
- **Проблема:** PATCH принимает is_active как Query Parameter, хотя по конвенции PATCH — тело запроса.
- **Риск:** Потребители API ожидают тело. Неочевидный контракт.
- **Рекомендация:** Создать ToggleUserActiveRequest с extra="forbid".

### M-006 — admin_config update_config: domain как Query Parameter
- **ID:** M-006
- **Category:** API contracts
- **Severity:** MEDIUM
- **Confidence:** 0.80
- **Location:** `server/app/presentation/api/routes/admin_config.py:72`
- **Проблема:** PUT принимает domain как Query Parameter — неочевидно из OpenAPI.
- **Рекомендация:** Перенести в тело запроса.

### M-007 — SSE heartbeat не содержит request_id
- **ID:** M-007
- **Category:** API contracts / Observability
- **Severity:** MEDIUM
- **Confidence:** 0.75
- **Location:** `server/app/presentation/api/constants.py:43`
- **Проблема:** `SSE_HEARTBEAT = ": heartbeat\n\n"` — клиент не может ассоциировать heartbeat с запросом.
- **Рекомендация:** heartbeat может содержать request_id.

### M-008 — documents upload без concurrent limit
- **ID:** M-008
- **Category:** API contracts / Performance
- **Severity:** MEDIUM
- **Confidence:** 0.75
- **Location:** `server/app/presentation/api/routes/documents.py:53`
- **Проблема:** Нет проверки concurrent uploads. Many файлов могут исчерпать ingestion semaphore.
- **Рекомендация:** Добавить semaphore/queue для document processing jobs.

### M-009 — upload document_content без проверки MIME/расширения для .md/.txt
- **ID:** M-009
- **Category:** API contracts / Security
- **Severity:** MEDIUM
- **Confidence:** 0.70
- **Location:** `server/app/presentation/api/schemas/validators.py:71`
- **Проблема:** Для .md/.txt нет проверки magic bytes (пустой список). Клиент может назвать вирусный .exe файл как .md.
- **Рекомендация:** Добавить проверку валидного UTF-8.

### M-010 — Нет API versioning (deferred decision)
- **ID:** M-010
- **Category:** API contracts
- **Severity:** MEDIUM
- **Confidence:** 0.95
- **Location:** `server/app/main.py:178-184`
- **Проблема:** Все эндпоинты на корневом уровне без versioning prefix.
- **Рекомендация:** Реализовать через nginx (AGENTS.md deferred decision) или router prefix.

### M-011 — time.sleep в benchmark judge блокирует thread pool
- **ID:** M-011
- **Category:** RAG pipeline / Performance
- **Severity:** MEDIUM
- **Confidence:** 0.80
- **Location:** `server/app/infrastructure/benchmark/judge.py:217`
- **Проблема:** `time.sleep(delay)` в синхронной функции, вызываемой из async context. Занимает thread pool.
- **Рекомендация:** Заменить на asyncio.sleep в async-обёртке.

### M-012 — _filter_sources_by_min_score fallback возвращает worst source
- **ID:** M-012
- **Category:** RAG pipeline quality
- **Severity:** MEDIUM
- **Confidence:** 0.85
- **Location:** `server/app/infrastructure/ml/rag/rag_sources.py:158-163`
- **Проблема:** Если ВСЕ источники ниже min_score, возвращается `[sources[0]]` — худший источник.
- **Риск:** LLM получает нерелевантный контекст и hallucinates.
- **Рекомендация:** Возвращать пустой список sources.

### M-013 — Question hash exact-match, не semantic
- **ID:** M-013
- **Category:** RAG pipeline cache
- **Severity:** MEDIUM
- **Confidence:** 0.75
- **Location:** `server/app/infrastructure/ml/answer_cache.py:52-54`
- **Проблема:** Два семантически идентичных вопроса с разным формулировками дадут cache miss.
- **Рекомендация:** Рассмотреть semantic cache на основе embedding similarity.

### M-014 — {context} placeholder может сломаться при { } в контенте
- **ID:** M-014
- **Category:** RAG pipeline / Security
- **Severity:** MEDIUM
- **Confidence:** 0.70
- **Location:** `server/app/domain/services/rag_policy.py:250-252`, `rag_prompts.py:199-200`
- **Проблема:** `sanitize_for_prompt()` не экранирует `{` и `}`. Template injection при использовании ChatPromptTemplate.
- **Рекомендация:** Экранировать `{` → `{{`, `}` → `}}` перед подстановкой.

### M-015 — Embedding versioning: нет автоматической миграции коллекции
- **ID:** M-015
- **Category:** RAG pipeline / Operations
- **Severity:** MEDIUM
- **Confidence:** 0.85
- **Location:** `server/app/infrastructure/repositories/vector/qdrant_ops.py:72-77`
- **Проблема:** Несовпадение размерности → ValueError → ingestion остановлен. Нет авто-миграции.
- **Рекомендация:** Автоматизированная миграция: create → re-index → alias → delete.

### M-016 — Cost rate limiting отключен по умолчанию
- **ID:** M-016
- **Category:** Multi-tenant security
- **Severity:** MEDIUM
- **Confidence:** 0.90
- **Location:** `server/app/config.py:265-267`
- **Проблема:** `cost_rate_limit_enabled: bool = False`. Нет per-user лимитов на токены/стоимость.
- **Риск:** 100 пользователей × 100 запросов × 2000 токенов = 20M токенов/день.
- **Рекомендация:** Включить в production. Per-user token counter в Redis.

### M-017 — Rate limiting fail-open при Redis outage
- **ID:** M-017
- **Category:** Multi-tenant security / Reliability
- **Severity:** MEDIUM
- **Confidence:** 0.90
- **Location:** `server/app/config.py:253`, `infrastructure/rate_limit/limiter.py:74-75`
- **Проблема:** `rate_limit_fail_open=true` — при Redis outage ВСЕ rate limiting отключается. Brute-force login неограничен.
- **Рекомендация:** Для login rate limit использовать fail-closed. Или in-memory semaphore.

### M-018 — Chat logs возвращают полный текст вопросов/ответов кураторам
- **ID:** M-018
- **Category:** Multi-tenant security / PII
- **Severity:** MEDIUM
- **Confidence:** 0.85
- **Location:** `server/app/presentation/api/routes/admin_chat_logs.py:66-68`
- **Проблема:** ChatLogEntry содержит question и answer без PII-маскирования.
- **Рекомендация:** PII-маскирование перед отображением. Raw text только для ADMIN.

### M-019 — step_generate без общего timeout на streaming
- **ID:** M-019
- **Category:** Performance / Reliability
- **Severity:** MEDIUM
- **Confidence:** 0.90
- **Location:** `server/app/infrastructure/ml/rag/rag_steps.py:381-424`
- **Проблема:** step_generate не оборачивает streaming в `asyncio.wait_for(timeout=...)`. Зависший LLM держит generation semaphore навсегда → каскадный отказ.
- **Рекомендация:** `asyncio.wait_for(timeout=180)` для step_generate.

### M-020 — ChatOllama без request_timeout
- **ID:** M-020
- **Category:** Performance / Reliability
- **Severity:** MEDIUM
- **Confidence:** 0.85
- **Location:** `server/app/infrastructure/ml/clients/factories.py:119-130`
- **Проблема:** ChatOllama без request_timeout. Зависший Ollama держит соединение бесконечно.
- **Рекомендация:** Добавить request_timeout=120.

### M-021 — Legacy SCAN O(N) по всем cache keys при invalidation
- **ID:** M-021
- **Category:** Performance
- **Severity:** MEDIUM
- **Confidence:** 0.80
- **Location:** `server/app/infrastructure/ml/answer_cache.py:159-181`
- **Проблема:** При invalidation legacy entries — SCAN по всем ключам + GET + decompress для каждого. При 100k entries = десятки секунд.
- **Рекомендация:** Ограничить SCAN count=1000 + timeout. Полная миграция на reverse index.

### M-022 — Нет per-tenant concurrency limit для ingestion
- **ID:** M-022
- **Category:** Performance / Multi-tenant
- **Severity:** MEDIUM
- **Confidence:** 0.85
- **Location:** `server/app/infrastructure/worker/tasks.py:119-170`
- **Проблема:** Один tenant, загрузивший 100 документов, займёт все 8 ingestion slots. Другие tenant'ы ждут.
- **Рекомендация:** Per-tenant concurrency limit.

### M-023 — PDF не зарегистрирован в PARSERS registry
- **ID:** M-023
- **Category:** RAG pipeline / Ingestion
- **Severity:** MEDIUM
- **Confidence:** 0.75
- **Location:** `server/app/infrastructure/ml/ingestion/registry.py:10-16`
- **Проблема:** PARSERS dict содержит только .docx, .doc, .rtf, .md, .txt. PDF обрабатывается через отдельный path.
- **Рекомендация:** Добавить ".pdf": parse_pdf в PARSERS или документировать.

### M-024 — Confidence метрика неоткалибрована
- **ID:** M-024
- **Category:** RAG pipeline quality
- **Severity:** MEDIUM
- **Confidence:** 0.70
- **Location:** `server/app/infrastructure/ml/rag/rag_steps.py:487`
- **Проблема:** Confidence = min(1.0, max(0.0, avg_sim)). Cross-encoder scores — logits, не probabilities. "0.85" не означает "85% уверенность".
- **Рекомендация:** Документировать как relevance score. Или sigmoid/platt scaling.

### M-025 — FakeUnitOfWork rollback не откатывает данные
- **ID:** M-025
- **Category:** Test quality
- **Severity:** MEDIUM
- **Confidence:** 0.85
- **Location:** `server/tests/fakes.py:817-821`
- **Проблема:** Rollback помечает `_rolled_back=True`, но данные в репозиториях остаются.
- **Рекомендация:** Snapshot mechanism: в `__aenter__` сохранять копию, в `__aexit__` восстанавливать.

### M-026 — FakeGroupRepository всегда возвращает []
- **ID:** M-026
- **Category:** Test quality
- **Severity:** MEDIUM
- **Confidence:** 0.90
- **Location:** `server/tests/fakes.py:116-136`
- **Проблема:** `get_user_group_ids()` hardcode возвращает `[]`. Group ACL не тестируется через фейки.
- **Рекомендация:** Добавить внутреннее хранилище memberships.

### M-027 — Нет markers для integration-тестов с fakeredis
- **ID:** M-027
- **Category:** Test quality
- **Severity:** MEDIUM
- **Confidence:** 0.80
- **Location:** `server/tests/test_rate_limit_redis_integration.py`
- **Проблема:** Тесты с fakeredis смешаны с unit-тестами. Маркер `integration` определён, но не применяется.
- **Рекомендация:** Пометить все тесты с infrastructure-зависимостями как `@pytest.mark.integration`.

### M-028 — Нет теста citation accuracy
- **ID:** M-028
- **Category:** Test quality
- **Severity:** MEDIUM
- **Confidence:** 0.75
- **Location:** `server/tests/test_rag_logic.py`
- **Проблема:** TestFilterCitedSources проверяет фильтрацию, но не проверяет соответствие [N] → source[N].
- **Рекомендация:** Assertion: for each [N] in answer, verify source[N-1].

---

## LOW / INFO

### L-001 — DeleteDocument мутирует два агрегата в одной транзакции [LOW]
- **Файл:** `server/app/application/services/document_command_service.py:225-250`
- Три разных агрегата (VectorOutbox, Conversation, Document) в одной UoW транзакции без domain event/saga.

### L-002 — Chunk и Message entity: анемичные модели [LOW]
- **Файлы:** `domain/entities/chunk.py:13-17`, `message.py:16-28`
- Без business logic. Инварианты проверяются снаружи.

### L-003 — RegulatoryAct, ActVersion, ApiKey: анемичные модели [LOW]
- **Файлы:** `domain/entities/regulatory_act.py`, `act_version.py`, `api_key.py`
- Data classes без business methods (expire, revoke, current_version).

### L-004 — MLContainer.dispose() не закрывает ML-клиенты [LOW]
- **Файл:** `composition/infrastructure.py:145-163`
- close() вызывается отдельно в main.py. Если dispose() вне lifespan — HTTP-пулы не закрыты.

### L-005 — preview cached file без TTL в response [LOW]
- **Файл:** `presentation/api/schemas/admin_quality.py:68`
- DryRunResponse без expires_at. Клиент не знает TTL preview_id.

### L-006 — Ingest registry без пагинации [LOW]
- **Файл:** `presentation/api/routes/ingest.py:132`
- GET /ingest/registry возвращает все файлы без limit/offset.

### L-007 — Exception handlers не возвращают error code [LOW]
- **Файл:** `presentation/api/exception_handlers.py:67`
- Нет machine-readable error code в ответе (только message + errors).

### L-008 — DatabaseError.detail утекает в response [INFO]
- **Файл:** `domain/exceptions/domain_errors.py:48-52`
- detail может содержать SQL-запрос или stack trace. mask для non-admin.

### L-009 — CURATOR + INTERNAL_GROUP(doc в managed group) не покрыт [LOW]
- **Файл:** `tests/test_acl.py`
- Нет теста: CURATOR видит INTERNAL_PRIVATE文档 в managed group.

### L-010 — BM25Index.search() без ACL — потенциальная утечка при рефакторинге [LOW]
- **Файл:** `infrastructure/bm25/bm25_index.py:123-140`
- search() не фильтрует. Не используется в user-facing путях, но потенциально опасен.

### L-011 — Нет refresh token механизма [INFO]
- **Файл:** `infrastructure/auth/jwt_provider.py:24`
- JWT 24ч. isActive проверяется в БД. Но нет revocation list.

### L-012 — Ingestion PDF: нет error isolation между страницами [LOW]
- **Файл:** `infrastructure/ml/ingestion/pdf.py:384-404`
- _process_page() без try/except. Одна corrupt page = весь PDF потерян.

### L-013 — step_build_context вычисляет context budget дважды [INFO]
- **Файлы:** `rag/rag_steps.py:356-372` vs `helpers.py:389-404`
- Возможна рассинхронизация бюджета.

### L-014 — Condensed query логируется в открытом виде [INFO]
- **Файл:** `rag/rag_prompts.py:86`
- PII в вопросах попадает в общий лог.

### L-015 — HybridRetriever создаётся на каждый rerank вызов [LOW]
- **Файл:** `rag/rag_reranking.py:38-41`
- Minor performance overhead.

### L-016 — Конфигурационные domain events — minimal coverage [INFO]
- **Файл:** `domain/events/config_events.py`
- Единственный event — ConfigParameterChanged. Для mono-service ок.

### L-017 — _admin_conditions для INTERNAL_GROUP без group match — избыточно [INFO]
- **Файл:** `domain/services/access_control.py:131-133`
- Redundant condition для ADMIN. Не уязвимость.

### L-018 — Клиент disconnect прерывает генератор, но не LLM call [INFO]
- **Файл:** `presentation/api/routes/chat.py:94-96`
- Платный LLM call выполняется впустую при early disconnect.

### L-019 — SSE нет Connection: keep-alive [INFO]
- **Файл:** `presentation/api/constants.py:42`
- Прокси могут закрыть соединение раньше.

### L-020 — InProcessEventBus module-level singleton [INFO]
- **Файл:** `infrastructure/events/in_process_event_bus.py:57`
- Безопасно при текущей архитектуре (контейнер создаётся один раз).

---

## По областям

### DDD-архитектура

**Общая оценка:** Чистая архитектура. Слой `domain/` и `application/` свободны от инфраструктурных импортов. Репозитории корректно разделены на интерфейсы (domain) и реализации (infrastructure). Composition root изолирует DI.

**Найдены 4 нарушения:**
- UserContext frozen dataclass с мутабельными list-полями (M-001)
- Conversation анемичная модель без поведения (M-002)
- DocumentService facade с property-прокси в приватные поля (M-003)
- DeleteDocument мутирует три агрегата в одной транзакции (L-001)

**Эталонные файлы:** `domain/services/access_control.py` (294 строки, zero infrastructure imports), `domain/entities/document.py` (rich entity), `domain/value_objects/curator_scope.py` (frozen VO с tuple).

### Dependency Injection

**Общая оценка:** Самописный контейнер без service-locator. Все зависимости через constructor injection + Depends-обёртки. Service Locator НЕ используется.

**Найдены 5 проблем:**
- InfrastructureContainer.validate() не проверяет domain_registry/domain_settings (H-013 → исправлено в H-013 description)
- ApplicationContainer не имеет validate() (H-013)
- db_password передаётся как голый dict (H-010)
- BM25IndexAdapter создаётся дважды (M-004)
- MLContainer.dispose() не закрывает ML-клиенты (L-004)

### API-контракты

**Общая оценка:** 18 роутеров, schemas разбиты по файлам. Rate limiting включён. Auth dependency защищает все эндпоинты (кроме /health и /metrics).

**Найдены 13 проблем:**
- Нет Idempotency-Key для write-операций (H-003)
- Untyped response модели для 12+ mutation-эндпоинтов (H-004)
- ValueError без обработки в compare_runs (H-011)
- HTTPException(500) в diagnose_document (H-012)
- is_active как Query Parameter в PATCH (M-005)
- domain как Query Parameter в PUT (M-006)
- SSE heartbeat без request_id (M-007)
- Upload без concurrent limit (M-008)
- Upload без MIME check для .md/.txt (M-009)
- Нет API versioning — deferred decision (M-010)
- 5 paginated lists без total (LOW из API)
- Нет error code в exception handlers (L-007)
- DatabaseError.detail утекает (L-008)

### RAG-пайплайн

**Общая оценка:** Полноценный пайплайн: condense → retrieve (dense + hybrid) → rerank → build context → generate → postprocess. Answer cache с visibility hash. Guardrails и benchmark infrastructure.

**Найдены 17 проблем:**
- ACL bypass в enrichment соседних чанков (C-001)
- Score threshold отсутствует в dense search (C-002)
- Prompt injection защита ограничена (H-007)
- Partial answer при обрыве LLM streaming (H-006)
- Circuit breaker race condition (H-005)
- time.sleep в judge retry (M-011)
- Fallback на worst source (M-012)
- Semantic cache miss (M-013)
- Template injection через { } (M-014)
- Embedding versioning без миграции (M-015)
- step_generate без timeout (M-019)
- ChatOllama без request_timeout (M-020)
- Legacy cache SCAN O(N) (M-021)
- PDF не в PARSERS (M-023)
- Confidence неоткалибрована (M-024)
- PII в condensed query логах (L-014)
- Context budget вычисляется дважды (L-013)

### Multi-tenant безопасность

**Общая оценка:** Тройной ACL invariant покрыт property-тестами. BM25 pre-filter + Qdrant resolve = defense in depth. Answer cache учитывает visibility scope. CuratorScope проверяется на max size. Timing-safe auth. PII redaction применяется к вопросам.

**Найдены 9 проблем:**
- ACL bypass в neighbor enrichment (C-001 — shared с RAG)
- Hardcoded UserKind.INTERNAL (H-001)
- Admin list_documents видит CLIENT_PRIVATE metadata (H-002)
- Prompt injection (H-007 — shared с RAG)
- Cost rate limiting отключен (M-016)
- Rate limiting fail-open при Redis outage (M-017)
- Chat logs без PII-маскирования (M-018)
- Per-tenant ingestion limit отсутствует (M-022)
- BM25Index.search() без ACL (L-010)

### Производительность и надёжность

**Общая оценка:** Semaphore timeout корректно реализован для 8 ресурсов. Circuit breaker с Prometheus метриками. Redis-backed rate limiter. Structured logging. Request ID пробрасывается.

**Найдены 10 проблем:**
- Sync QdrantClient блокирует event loop (H-008)
- Новый QdrantClient на каждый health call (H-008 — shared)
- step_generate без timeout (M-019)
- ChatOllama без request_timeout (M-020)
- Legacy cache SCAN O(N) (M-021)
- Per-tenant ingestion limit (M-022)
- time.sleep в judge (M-011)
- httpx.AsyncClient на каждый health/metrics call (L)
- Qdrant sync client占用 threads (INFO)
- Нет per-tenant token metrics (INFO)

### Качество тестов

**Общая оценка:** 74 тестовых файла. Хорошая coverage unit/integration для ACL invariant, RAG policy, domain entities, ingestion. Prompt injection defense покрыт. Property-тесты для ACL triangle.

**Найдены 12 проблем:**
- Нет E2E тестов (C-003)
- FakeUnitOfWorkFactory не даёт изоляции (C-004)
- 20+ app services без тестов (H-014)
- Domain services без тестов (H-014 — shared)
- Нет fallback-тестов при failures (H-015)
- FakeUnitOfWork rollback не работает (M-025)
- FakeGroupRepository всегда [] (M-026)
- Нет integration markers (M-027)
- Нет citation accuracy теста (M-028)
- conftest.py без UserContext factory (LOW)
- Дублирование characterization-тестов (LOW)
- FakeChatRAGPort без error scenarios (LOW)

---

## Итоговая статистика

| Severity | Количество | Основные области |
|----------|-----------|-----------------|
| CRITICAL | 4 | Multi-tenant security, RAG quality, Test infrastructure |
| HIGH | 15 | Security, API contracts, Architecture, Resilience, Tests |
| MEDIUM | 28 | Performance, API, DDD, Cache, Security, Tests |
| LOW/INFO | 38 | Code quality, Minor improvements, Observability |

**Общая оценка проекта:** Проект имеет чистую архитектуру с хорошо соблюдёнными границами слоёв. ACL invariant — strongest point, покрыт property-тестами. Основные риски: (1) data leak через neighbor chunks и hardcoded UserKind, (2) отсутствие E2E тестов и некорректные fakes, (3) resource exhaustion при dependency failures, (4) missing idempotency для write-операций.

---

# Приоритизированный план исправлений

## P0 — Спринт 0 (немедленно, до следующего деплоя)

### P0-SECURITY: Защита данных между тенантами

| # | Finding | Файл | Effort |
|---|---------|------|--------|
| 1 | ACL bypass в neighbor enrichment | `rag_postprocess.py:129-205` | M |
| 2 | Hardcoded UserKind.INTERNAL в delete/rename | `document_command_service.py:230,288` | S |
| 3 | Score threshold отсутствует в dense search | `rag_retrieval.py:102-100` | S |
| 4 | Fallback на worst source при min_score | `rag_sources.py:158-163` | S |
| 5 | Admin list_documents видит CLIENT_PRIVATE metadata | `document_query_service.py:54-56` | M |

### P0-RESILIENCE: Защита от каскадных отказов

| # | Finding | Файл | Effort |
|---|---------|------|--------|
| 6 | Circuit breaker race condition | `circuit_breaker.py:101-114` | S |
| 7 | step_generate без timeout на streaming | `rag_steps.py:381-424` | S |
| 8 | ChatOllama без request_timeout | `factories.py:119-130` | S |
| 9 | Sync QdrantClient блокирует event loop | `system_health_probe.py:55-63,115,123` | M |

### P0-API: Контрактная надёжность

| # | Finding | Файл | Effort |
|---|---------|------|--------|
| 10 | Idempotency-Key для write-эндпоинтов | `routes/documents.py`, `chat.py`, `benchmark.py` | L |
| 11 | Untyped response для 12+ mutation-эндпоинтов | `routes/auth.py`, `groups.py`, `curators.py` и др. | L |
| 12 | ValueError в compare_runs + HTTPException(500) в diagnose | `benchmark_admin.py:377`, `admin_quality.py:101` | S |
| 13 | Rate limiting fail-open при Redis outage | `config.py:253`, `limiter.py:74-75` | S |
| 14 | Cost rate limiting отключен | `config.py:265-267` | S |

**P0 итого:** 14 задач, ~17 человеко-дней

---

## P1 — Спринт 1 (тестовая инфраструктура)

### P1-TESTS: Инфраструктура тестирования

| # | Finding | Файл | Effort |
|---|---------|------|--------|
| 15 | FakeUnitOfWorkFactory изоляция транзакций | `fakes.py:824-832` | M |
| 16 | FakeUnitOfWork rollback не откатывает данные | `fakes.py:817-821` | M |
| 17 | E2E тесты критичных эндпоинтов | `tests/` — новые файлы | L |
| 18 | Fallback-тесты при infrastructure failures | `test_rag_service_stream.py` и др. | M |
| 19 | FakeGroupRepository memberships | `fakes.py:116-136` | S |
| 20 | Integration markers для fakeredis тестов | `test_rate_limit_redis_integration.py` | S |
| 21 | Citation accuracy тест | `test_rag_logic.py` | S |

**P1 итого:** 7 задач, ~10 человеко-дней

---

## P2 — Спринт 2 (архитектура + безопасность + производительность)

### P2-ARCH: Архитектурная чистота

| # | Finding | Файл | Effort |
|---|---------|------|--------|
| 22 | Presentation → infrastructure imports | `admin_quality.py:16`, `dependencies.py:42,184` | M |
| 23 | db_password в голом dict | `infrastructure.py:207-215` | S |
| 24 | ApplicationContainer без validate() | `application.py:63-92` | S |
| 25 | UserContext frozen VO с мутабельными list | `user_context.py:15-18` | S |
| 26 | Conversation анемичная модель | `conversation.py:16-21` | M |
| 27 | DocumentService facade property-прокси | `document_service.py:45-74` | M |

### P2-PERF: Производительность

| # | Finding | Файл | Effort |
|---|---------|------|--------|
| 28 | Legacy cache SCAN O(N) | `answer_cache.py:159-181` | M |
| 29 | Per-tenant ingestion limit | `worker/tasks.py:119-170` | M |
| 30 | time.sleep в judge retry | `judge.py:217` | S |

### P2-SECURITY: Дополнительная безопасность

| # | Finding | Файл | Effort |
|---|---------|------|--------|
| 31 | Chat logs без PII-маскирования | `admin_chat_logs.py:66-68` | S |
| 32 | {context} template injection | `rag_policy.py:250-252` | S |

**P2 итого:** 11 задач, ~12 человеко-дней

---

## P3 — Спринт 3 (тесты + долгосрочные улучшения)

### P3-TESTS: Расширение покрытия

| # | Finding | Файл | Effort |
|---|---------|------|--------|
| 33 | Unit-тесты для 20+ app services | `application/services/*.py` | XL |
| 34 | Unit-тесты domain services | `document_versioning.py`, `document_parser.py` | M |

### P3-PERF: Долгосрочные оптимизации

| # | Finding | Файл | Effort |
|---|---------|------|--------|
| 35 | Semantic cache | `answer_cache.py` | L |
| 36 | Embedding versioning migration | `qdrant_ops.py:72-77` | L |

### P3-ARCH: Архитектурные мелочи

| # | Finding | Файл | Effort |
|---|---------|------|--------|
| 37 | BM25IndexAdapter дважды | `application.py:190,198` | S |
| 38 | MLContainer.dispose() не закрывает клиентов | `infrastructure.py:145-163` | S |
| 39 | Presentation query params (toggle, config) | `auth.py:94`, `admin_config.py:72` | S |
| 40 | API versioning (deferred decision) | `main.py:178-184` | M |

**P3 итого:** 8 задач, ~15 человеко-дней

---

## P4 — Спринт 4 (cleanup, код-стайл)

| # | Finding | Файл | Effort |
|---|---------|------|--------|
| 41 | Chunk/Message/ActVersion анемичные модели | `entities/chunk.py`, `message.py`, `act_version.py` | M |
| 42 | conftest UserContext factory + characterization dedup | `conftest.py`, `test_rag_service_*.py` | M |
| 43 | SSE heartbeat + upload limits | `constants.py:43`, `documents.py:53`, `validators.py:71` | S |
| 44 | Confidence calibration + PDF page isolation | `rag_steps.py:487`, `pdf.py:384-404` | M |
| 45 | INFO-level: DatabaseError detail, context budget, PII in logs, events, _admin_conditions | various | S |

**P4 итого:** 5 задач, ~5 человеко-дней

---

## Сводка по фазам

| Фаза | Задач | Effort | Что закрывает |
|------|-------|--------|---------------|
| **P0** | 14 | ~17 дн | 4 CRITICAL + 8 HIGH + 2 MEDIUM |
| **P1** | 7 | ~10 дн | Тестовая инфраструктура, coverage gaps |
| **P2** | 11 | ~12 дн | Архитектура, безопасность, производительность |
| **P3** | 8 | ~15 дн | Расширение тестов, semantic cache, embedding migration |
| **P4** | 5 | ~5 дн | Cleanup, code style, INFO-level |
| **Итого** | **45** | **~59 дн** | Все 63 находки |

**Рекомендация:** Начать с P0 (14 задач, 17 дней) — это 2/3 всех критичных и высоких находок. P1做完后继续执行后续阶段。
