# План устранения находок REVIEW_REPORT.md (2026-09-27)

Источник: `REVIEW_REPORT.md` — 114 находок (3 CRITICAL / 31 HIGH / 52 MEDIUM / 28 LOW-INFO).

## Принципы работы (из AGENTS.md)

- **Gate после каждой волны:** `task test` (зелёные, включая 5 инвариант-тестов из списка
  «никогда не удалять»), `task lint`, слоистые grep-проверки (`domain/` и `application/`
  без импортов infrastructure/presentation).
- **God-файлы** (`document_processor.py`, `rag_service.py`, `document_service.py`, `chunk_service.py`)
  — сначала characterization-тесты, потом правки.
- **Hotfix отдельно**, рефакторинг отдельно. Порядок внутри волны — от XS-effort к M.

---

## Волна 0 — Hotfix (1 день, отдельный патч-релиз)

Прод неработоспособен для всех ролей, при этом CI зелёный — лечим и баг, и дыру в тестах.

| # | Находка | Действие | Критерий приёмки |
|---|---|---|---|
| 0.1 | **C1** | `acl.py:73`: `user["id"]` → локальная `user_id` (уже вычислена на :47) | `build_qdrant_filter(UserContext(...))` не падает для USER/CURATOR/CLIENT/ADMIN |
| 0.2 | C1 (тесты) | 4 параметризованных теста с реальным `UserContext` без моков; перевести `TestACLInvariant` с dict на `UserContext`; сквозной тест `RagService._init_state` без патча фильтра | Тесты падают на неисправленном коде (проверить revert'ом фикса) |

**Статус:** в работе.

---

## Волна 1 — Разблокировка event loop (2–3 дня)

| # | Находка | Действие | Критерий приёмки |
|---|---|---|---|
| 1.1 | **C2** | BM25: batch-метод `remove_many(indices)` (один проход + одна пересборка, удаление в обратном порядке) + `asyncio.to_thread` в `document_command_service.py` | `DELETE` крупного документа не блокирует healthcheck (latency /health < 1s во время удаления) |
| 1.2 | **C3** | Characterization-тесты на `document_processor.py` (god-file, L2) → обернуть `parse`/`assess`/рендер/`split` в `asyncio.to_thread`; семафор «один OCR на воркер» | heartbeat воркера обновляется во время OCR 30-мин джобы |

---

## Волна 2 — Транзакционные границы и жизненный цикл (3–5 дней)

| # | Находка | Действие | Критерий приёмки |
|---|---|---|---|
| 2.1 | **H1** | S3 `upload_file`/`copy_file` вынести **до** открытия UoW (ключ детерминирован по document_id); cleanup при падении commit — через outbox, не inline-except | Нет S3-I/O между `uow.begin` и `commit`; тест «сирот в S3 при падении commit нет» |
| 2.2 | **H2** | BM25-мутации применять **после** выхода из UoW (по образцу `storage_deletes`) | Тест «rollback не оставляет следов в BM25» |
| 2.3 | **H18** | `chat_service.py`: persist вопроса сразу после `_prepare_chat`; `try/finally` вокруг SSE-цикла, persist ответа с `truncated=True` в `finally`; явный `aclose()` | Тест «обрыв клиента → оба сообщения + chat_log + токены в БД» |
| 2.4 | **H10, M9, M10** | `container.dispose()` в `_on_shutdown` воркера; `ml_clients.close()` внутрь `InfrastructureContainer.dispose()`; `_shutdown` в `finally` lifespan; async-rollback init через `dispose()`, не через приватное поле | Рестарт воркера без висящих HTTP-пулов; нет утечек при падении старта |
| 2.5 | **H19** | `@retry_on_transient` (HTTPStatusError 429/5xx, `Retry-After`) + breaker `tei_embed`/`tei_rerank`/`s3`; общий `httpx.Client` для sync TEI | Тест: транзиентный 502 → успех после retry; висящий TEI не отбирает все слоты семафора |
| 2.6 | **H20 + H31** (вместе — один контракт) | `_run_tracked_job`: транзиентные (`httpx.TransportError`, `TimeoutError`, 5xx) → `arq.Retry(defer=...)`; `document_processor.py:434` — re-raise после выставления статуса документа | Тест «исключение в process → document FAILED, job FAILED, arq видит retry»; статусы агрегатов согласованы |
| 2.7 | **H21** | `job_completion_wait=30-60s` в arq, `stop_grace_period: 60s` в compose; при `CancelledError` job → `pending`, не `failed` | Деплой во время джобы: UI не показывает ложный `failed` у переочередной задачи |

---

## Волна 3 — API-контракт и безопасность периметра (2–3 дня)

| # | Находка | Действие | Критерий приёмки |
|---|---|---|---|
| 3.1 | **H6** | Переместить `compare_runs` выше `get_run` | Маршрутный гонек-тест: `GET /admin/benchmark/runs/compare` → не 422 |
| 3.2 | **H7** | Единое правило «нет доступа → 404» в `exception_handlers` / access_control; 403 — только policy-отказы без раскрытия существования; поправить `api_keys` (200+`[]` на несуществующем id) | Тест-оракул: несуществующий и чужой id неразличимы по статусу |
| 3.3 | **H4** | `read_upload_with_limit` в `documents.py`, `admin_quality.py` (обе точки); проверка `Content-Length` до чтения | Upload 10 ГБ → 413 без аллокации |
| 3.4 | **H5** | Middleware с лимитом `Content-Length`/потоковым abort **до** auth и rate-limit | Неаутентифицированный гигабайтный POST → мгновенный отказ |
| 3.5 | **M15, M18, M22** | Заголовки `HTTPException` (`Retry-After`) не терять; `error_message` → код ошибки для пользователя, полный текст — только `/admin/*`; `status_code=201/202` | Контрактные тесты на заголовки/статусы |

---

## Волна 4 — Консистентность данных (2–3 дня)

| # | Находка | Действие | Критерий приёмки |
|---|---|---|---|
| 4.1 | **H12** | `DELETE_BY_DOCUMENT` перед upsert в CLI-путях; `extend` BM25 удаляет старые записи по source/content_hash | Тест «изменил файл → реингест → ровно N точек в Qdrant и BM25» |
| 4.2 | **H13** | `reconcile`: DONE только при `pending==0 && failed==0 && dead_letter==0`, иначе FAILED с причиной; убрать ложный `version+1` | Тест «outbox pending → документ остаётся INDEXING» |
| 4.3 | **H16** | `invalidate_by_document_ids` в тех же UoW, что мутации чанков/реингест | Тест «edit chunk → cache miss» |
| 4.4 | **H8** | Переходы статуса Document — через entity (`transition_to`/`mark_done`), как у BackgroundJob; 7 production-вызовов переключить | Инвариант «DONE/FAILED → PROCESSING запрещён» исполняется в проде; property-тест |
| 4.5 | **H24** | Outbox-dispatcher: один `UPDATE ... WHERE id = ANY` + один `count_by_document` на батч | Документ на 1000 чанков: десятки запросов вместо 130+, 1 commit вместо 32 |

---

## Волна 5 — Тестовая инфраструктура (3–5 дней, частично параллельно волнам 3–4)

| # | Находка | Действие |
|---|---|---|
| 5.0 | **M14** (предпосылка) | Убрать `@Singleton` у `Application` / `create_application(container=None)` + очистка `dependency_overrides` в фикстуре — без этого 5.1 протекает между тестами |
| 5.1 | **H25** | Smoke-набор `TestClient(create_application())` с `dependency_overrides`: 200 владелец / 403 чужая роль / 404 чужой id по `/documents`, `/chunks`, `/conversations`, `/api_keys`, `/admin/*` + гонек-тест роутов (ловит H6-класс багов) |
| 5.2 | **H29** | `Container().init()` без патчей саб-контейнеров + `assert validate() == []`; тест «PROD + issues → RuntimeError» |
| 5.3 | **H26, H27, H28, H30** | Матрица `can_be_viewed_by`; реальный `ChatService._prepare_chat` (curator_scope лимит); wiring answer-cache (два контекста → разные ключи); фейк listing'а на `for_list=True` + SQL-тесты `list_visible` |
| 5.4 | **M42** | `pytest-cov` + `task cov` (`--cov=app --cov-report=term-missing`) — baseline до рефакторингов god-файлов |
| 5.5 | **M30, M38, M41, M43, M45** | Выровнять ACL-фейк; реальный rollback в FakeUoW; патчеры через context manager; reranker-фейки читают `pairs`; `DATA_DIR` → `tmp_path_factory` |

---

## Волна 6 — Качество RAG и бенчмарка (2–3 дня)

| # | Находка | Действие |
|---|---|---|
| 6.1 | **H11** | `context_for_judge` — фактический `state.docs`, не имена файлов; регрессионный тест (context ≠ join(source names)) |
| 6.2 | **H14** | Reranker: `scores_by_index` по полю `index`, проверка полноты индексов; тест с перемешанным `index` + один прогон против живого TEI (confidence 0.65 — подтвердить порядок ответа) |
| 6.3 | **M23** | Fallback на dense+BM25 при падении reranker + `get_breaker("rerank")` + метрика `rag_rerank_fallback_total` |
| 6.4 | **M25, M26** | `gather(return_exceptions=True)` + partial save в `finally`; judge `temperature=0`, aux-модель в настройку; `judge_error` вместо `0.0` |
| 6.5 | **M24** | `sources[]` строится из того же списка и порядка, что нумерация `[N]`; убрать fail-open в `filter_cited_sources` |
| 6.6 | **H15, M27, M28, M29** | Ingestion: явный merge metadata (content_type после merge), пересчёт page ranges, не отбрасывать текст страниц с таблицей, batch-ингест классифицирует domain до сплита |

---

## Волна 7 — Надёжность и наблюдаемость (backlog, 3–4 дня)

| # | Находки | Действие |
|---|---|---|
| 7.1 | **H17** | BM25: лок у читателей или copy-on-write (атомарная замена ссылки); `last_survival_ratio` — в возвращаемое значение + concurrency-тест (M40) |
| 7.2 | **H22** | `/health`: разделить liveness/readiness, `asyncio.gather`, убрать SQL/внешний LLM-пинг из liveness, кэш 5–10s, сузить раскрываемые поля |
| 7.3 | **H23** | `statement_timeout` → 15–30s, явный `pool_timeout=5-10s`, длинные запросы — `SET LOCAL` |
| 7.4 | **M35, M36, M32** | Доверенные XFF-IP; `sanitize_for_prompt` на вопрос и историю; PII-редакция логов через `log_user_text()` |
| 7.5 | **M46–M52, M49** | tenant/role-лейблы метрик, healthcheck воркера (arq health key), сборщик `previews/`, корректная метрика памяти, двойная проверка кэша до/после condense |

---

## Волна 8 — Архитектурный долг (по возможностям, low/medium)

- **M1, L16, M7** — use-case «upload → job → enqueue» в `DocumentService.upload_and_enqueue`; порог 0.3 в domain.
- **M2, M6, M34, M20** — ACL entity-методы поверх `get_visibility_conditions`; проброс реального `kind`; полный `UserContext` в SQL-ветке RAG.
- **M5, M8, M13** — CLI через composition/порты, убрать цикл infrastructure→composition, одна фабрика `ActVersioningService`.
- **L2** — разбор оставшихся god-файлов (`document_processor.py`, `benchmark_admin.py`, `admin_quality.py`…) — только после волны 5 (характеризация) и по `docs/refactor/REFACTOR_PLAN.md`.
- **L13–L15, L21–L28** — Enum в Form/Query, `Field(max_length)`, OpenAPI-полировка, `text("true")` → `raise`, обновить AGENTS.md (L26: удалить секцию про несуществующий `_preload_models`).

---

## Резюме по срокам и зависимостям

| Волна | Содержание | Effort | Эффект |
|---|---|---|---|
| 0 | C1 hotfix | ~час | Прод оживает, регрессия под тестом |
| 1 | C2, C3 | 2–3 дн | Процесс не замораживается |
| 2 | H1,H2,H10,H18–H21,H31,M9,M10 | 3–5 дн | Нет потерь данных, ретраи работают |
| 3 | H4–H7 + API-medium | 2–3 дн | Периметр закрыт, контракт честный |
| 4 | H8,H12,H13,H16,H24 | 2–3 дн | Статусы и индексы не расходятся |
| 5 | H25–H30 + тестовая база | 3–5 дн | Дыры, породившие C1, закрыты |
| 6 | RAG/benchmark quality | 2–3 дн | Метрики бенчмарка перестают быть шумом |
| 7–8 | Backlog | по ёмкости | Наблюдаемость, техдолг |

**Зависимости:** 0 → всё; 1 перед 2.4–2.7 (воркер); M14 перед H25; волна 5 перед волной 8
(god-files); H19 желательно до H20 (ретраи arq опираются на типизацию транзиентных ошибок).
