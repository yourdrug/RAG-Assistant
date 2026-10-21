# План вертикального рефакторинга `server/app/infrastructure`

Связанный документ: `refactoring-open-questions.md` (решения, без которых нельзя удалять код и менять схему БД).

**Проверка актуальности: 2026-10-21.** Первоначальные гипотезы сверены с затронутыми Domain/Application/Composition, API/CLI и тестами. Доказательства с путями, строками и сценариями — в `refactoring-open-questions.md`, вопросы 1–24. Непроверенные детали отдельных объединений ниже остаются кандидатами: аудит не доказывает полезность каждой предлагаемой абстракции и не проверяет фактические рабочие схемы/корпус.

**Подтверждённые требования:** PostgreSQL replicas сохранить; production/dev/stage можно пересоздать; API сейчас один, rolling-деплоя нет; внешних потребителей старых импортов нет; OCR-движки сохранить. Постоянное файловое хранение — только S3-совместимое, временные файлы с cleanup допустимы. Для `.doc` нет утверждённого решения: LibreOffice не согласован, альтернативу либо отказ от формата нужно выбрать отдельно.

**Что изменилось относительно первоначального плана:**

| Пункт | Актуальный вывод |
|---|---|
| BM25 | RAG и sweep уже используют Qdrant sparse, синхронный с dense по point ID. Старый выбор между API incremental и nightly S3 rebuild снят. |
| Qdrant metadata | Пакетный `set_payload(key="metadata")` уже реализован; убрать из списка багов. |
| Chunk UNIQUE | UNIQUE `(document_id, chunk_index)` восстановлен и присутствует в ORM. Тройной UNIQUE — вопрос избыточности, не отсутствия защиты. |
| Sync-клиенты | Нужны обычному sweep и async judge через поток; удаление standalone не делает их мёртвыми. |
| История benchmark | JSONL I/O можно отделять, но `compare_runs` используется web regression-check. |
| CI | Ruff, strict mypy с baseline и import-linter включены в task lint/CI. Добавлен отдельный gate baseline upgrade/check/downgrade и bootstrap. |
| Порты | Порты распределены по application-контекстам; число реализаций не основание заменять их все на Callable. |
| Хранение | Удалить local-storage override **вместе** с LocalStorage/fallback, backend-ветками config/API и тестовыми defaults; подробнее вопрос 24. |

**Проверки аудита:** 381 целевой тест прошёл; import-linter: 26 kept, 0 broken с текущими исключениями. Полный pytest/task lint, PostgreSQL/MinIO integration, alembic check и реальные LLM/OCR не запускались. Повтор retrieval/transport вне sandbox прошёл после подтверждения проблемы `asyncio.to_thread` самого sandbox. Исходники и базы не изменялись.

## Содержание

1. [Цели и критерии готовности](#1-цели-и-критерии-готовности)
2. [Фаза 0: страховочная сетка](#2-фаза-0-страховочная-сетка)
3. [Фаза 1: сквозные примитивы](#3-фаза-1-сквозные-примитивы)
4. [Срезы A–H](#4-срезы)
5. [Целевая структура](#5-целевая-структура)
6. [Баги, найденные по дороге](#6-баги-найденные-по-дороге)
7. [Порядок, стратегия PR, риски](#7-порядок-стратегия-pr-риски)
8. [Как исполнять план](#8-как-исполнять-план)

---

## 1. Цели и критерии готовности

**Вертикальный срез** означает: один срез (ingestion, retrieval, benchmark и т. д.) проходит от модели БД и миграции до worker и метрик целиком. Каждый срез закрывается отдельной серией PR.

**Definition of done для всего рефакторинга:**

- нет `# noqa: F401 — re-exported for test patches` и «backward compat» реэкспортов;
- нет функциональных импортов внутри тел функций, которые прячут циклы;
- сохраняются действующие import-linter контракты; legacy-исключения сокращаются по мере переноса сборки из worker в Composition. Порты допустимы в соответствующих `application.<context>.ports`, `application.uow` и `domain.*`, а не только в `application.ports`. Текущие `worker/tasks.py` и `benchmark/sweep_data.py` импортируют `application.benchmark`, старые `application.services.*` уже неактуальны;
- `mypy --strict` и Ruff уже настроены, но есть baseline/исключения. Уменьшать долг затронутого кода, а не обещать полное снятие baseline одним срезом. Смешение стилей `'` и `"`, `Union[...]` и `|`, русских и английских докстрингов само по себе не основание для функционального рефакторинга;
- `alembic check` показывает 0 расхождений между моделями и миграциями;
- нет ссылок на тикеты вроде `H-7/H-8`, `C-4`, `L-15`, `AUDIT.md item 14`; вместо них нормальные описания.

---

## 2. Фаза 0: страховочная сетка

Выполняется до любого переноса кода.

1. **Characterization-тесты** на то, что нельзя сломать:
   - чанкинг: `split_documents`, `table_chunks`, `text_chunks` на золотых файлах (pdf, docx, rtf, md, txt);
   - Qdrant sparse: ACL/temporal filter до top-k, edit/delete и point identity; старый BM25 — только для оставшихся потребителей;
   - ACL: одинаковый результат адаптеров в одном режиме, с сохранением разницы listing/search (см. срез A);
   - RAG-пайплайн на фикстурах с фейковыми клиентами;
   - бюджетный леджер.
2. **Baseline схемы на пустой БД.** Владелец разрешил пересоздание всех сред: проверить новую схему, ограничения, индексы, расширения и начальные данные; не использовать stamp как замену DDL. Снимок старой схемы полезен для сверки, но перенос старых данных не является обязательным условием.
3. **Решения только для затронутого среза** (файл `refactoring-open-questions.md`). Большинство фактов о вызовах и тестах уже проверено. Незакрытый вопрос `.doc` не блокирует исправление курсоров или S3-only.
4. **CI-гейты:** Ruff, mypy и import-linter уже есть. Сохранить их и текущие characterization-тесты; точечно добавить недостающие сценарии. `alembic check` добавлять после включения служебных таблиц в metadata и устранения расхождений.

---

## 3. Фаза 1: сквозные примитивы

Это кандидаты для конкретных срезов, **не обязательная общая фаза до полезных исправлений**. Сначала доказать одинаковую семантику и проверить потребителей. Особенно cancellation, lease и token budgeting нельзя объединять по сходству синтаксиса. Не вводить SettingsPort/базовые классы/общую retry policy, пока независимый срез этого не требует.

| Что дублируется сейчас | Единая замена |
|---|---|
| Паттерн «`try: await shield(task) except CancelledError: finish_owned_task; raise`» в `run_storage_io`, `run_owned_sync` (точная копия в `answer_generators`), `persist_accounting`, `close()` у 6 классов | Одна функция `shielded(coro)` в `infrastructure/concurrency.py` |
| Lease/retire/close-when-idle: `ManagedHTTPPool`, `ManagedLLM`, `ManagedInstructor`, `ManagedSyncInstructor`, `BoundQdrantClient`, `S3Storage`, `LazyStorage`, плюс `close_llm_pools` с цепочкой `hasattr` | Базовые `LeasedResource` (async) и `SyncLeasedResource`; конкретные классы только адаптируют close |
| Выбор модели: `settings.llm_model if provider == OLLAMA else settings.openrouter_model` в `helpers`, `rag_relevance`, `rag_prompts`, `rag_steps`, `judge`, `evidence_judge` | `active_llm_model(settings)` |
| Три способа читать настройки: `settings.x`, `get_setting("rrf_k")`, `get_setting("rag.rrf_k")`; плюс `LiveXxxSettings`-порты | Один `SettingsPort` с типизированными секциями (`rag`, `chunking`, `ocr`, `llm`) и контекстными оверрайдами |
| Circuit breaker: ручные `check_open()` / `report_success()` / `report_failure()` в `assess_sufficiency`, `handle_relevance_gate`, `step_generate`; `call()` и декоратор не используются | `async with breaker.guard():` |
| Retry: `resilience/retry.py`, свой `tenacity` в `rag_prompts`, свой в `judge`, `_RETRYABLE_ERRORS` | Одна политика в `resilience/` с именованными пресетами |
| Создание `QdrantClient` в `factories`, `corpus/control`, `health`, `QdrantInfo` | Одна фабрика `raw_qdrant_client(endpoint)` |
| Два listener-а LISTEN/NOTIFY (`PostgresConfigListener`, `PostgresOutboxListener`): почти копия supervisor-цикла | `PgListener` с реализацией `on_notify` и `resync` |
| Разбор `IntegrityError` в имя constraint (дважды в `benchmark_sweep_repository`) | `constraint_name(exc)` |
| Table → markdown в `pdf`, `docx`, `rtf`; regex `(?<!\\)\|` в 6 местах; `(?:an\|n\|a)\.\.\d+` в 5 местах | `markdown_table.py`: `render(rows)`, `split_cells(line)`, `FIELD_FORMAT_RE` |
| Магический префикс `"\x00TABLE:"` в md/docx/rtf/langchain-парсере | Типизированный `Section(heading, content, kind)` |
| Разные оценки токенов: `CHARS_PER_TOKEN=4` и `bytes/3` | Сначала разделить приблизительный контекстный budget и консервативный финансовый reserve; не заменять одной формулой без доказательства нужных верхних границ |
| Строки статусов и операций | Использовать уже существующие domain enum/LifecycleStatus в рабочем коде; исторические миграции содержат фиксированные значения и не импортируют изменяемый runtime enum |
| Время: aware timestamp с разными зонами, naive `edited_at`, session Europe/Minsk | Предложение aware UTC для timestamp после проверки SQL-семантики; календарные effective dates сохраняют бизнес-смысл. Сам aware Minsk не является ошибкой |
| Логгеры `"default"` / `"detailed"`; `detailed_stdout` использует `_default_formatter` (опечатка); `LOG_FORMAT` читается через `os.getenv` мимо settings | `logging.getLogger(__name__)`, один формат, настройки из `Settings` |
| Строка «Информация не найдена в документах.» в `rag_steps` (×2), `output_scanner`, `rag_postprocess` | Константа в `domain` |

---

## 4. Срезы

### A. Данные и схема (БД, модели, миграции, репозитории)

**Миграции.** До консолидации — 56 файлов; в исходном анализе отмечены 4 merge-миграции, автоген, который удалил индексы, и восстановление через `f1a2…`, `r7s8…`; индекс `ux_documents_active_slot` пересоздавался 6 раз; есть no-op миграции («default values seeded at startup»); в докстрингах `Revises` расходится с реальным `down_revision` (`j7k8…`, `l2m3…`); три разных стиля.

План:

1. Включить в metadata служебные таблицы и устранить подтверждённые расхождения.
2. Построить baseline `0001` и применить к пустой PostgreSQL.
3. Проверить DDL, расширения, индексы/FK/CHECK и обязательные seed/data-эффекты; затем `alembic check`.
4. В согласованное окно остановить API/worker и пересоздать базы production/dev/stage (допустимость подтверждена владельцем). Reset — отдельное действие, не часть аудита.
5. Старую цепочку удалить только после проверки нового baseline. Не переписывать старые down_revision/merge-узлы выборочно на ещё используемой цепочке.

**Модели.**

- Разнести `models.py` по агрегатам.
- Описать `cost_rates`, `cost_usage`, `corpus_control`, `corpus_processes` в SQLAlchemy metadata (ORM либо Core Table). Для alembic check не требуется переписывать рабочий сырой SQL.
- Исправить подтверждённые расхождения: `ChatLogModel.user_id` (`Mapped[int]` при nullable=True), `chunks.edited_at` без tz при timezone-aware миграции. Отдельно решить контракт allowed_values/upsert и нужность FK benchmark_runs.job_id. UNIQUE `(document_id, chunk_index)` уже восстановлен; трёхколоночный индекс избыточен как гарантия, но не означает отсутствие ограничения. `BenchmarkCheckpointModel.updated_at` явно объявлен и не является ошибкой наследования.

**Репозитории.**

- `SQLAlchemyChunkRepository` — чистый делегирующий фасад над 5 подрепозиториями; заменить композицией или наследованием.
- Не вводить общий CRUD Repository только по сходству select/mutate/flush. Сохранять специфичные блокировки, ограничения, UoW и ошибки каждого агрегата; общий helper — только при подтверждённой одинаковой семантике.
- `chunk_pagination` вручную собирает **ChunkRecord**, пропуская новые metadata-поля; согласовать маппинг с текущим `chunk_mapping` и проверить API listing. Старые имена ChunkSearchResult/to_chunk_search_result не использовать как готовый контракт.
- Одна сессионная абстракция вместо смеси `AsyncSession` и `SessionProtocol`.
- Старые `get_all_contents*` уже не найдены в app/scripts; текущий корпусный метод — `iter_all_contents_with_acl`. `ConversationRepository.get` уже заменён на get_by_id. Кандидаты без найденных app/scripts вызовов: conversation save, config get_by_key, increment_evaluated, delete_internal_documents — перед удалением проверить порты/тесты. **Не удалять** group list_by_ids (вызывается identity и upload) и regulatory_acts.list_all (application/act_versioning/queries.py:33).
- `reconcile_indexing_documents`: докстринг говорит «mark DONE», код только возвращает id.

**ACL: четыре транслятора `VisibilityCondition`.** Это `build_qdrant_filter`, `build_acl_clauses`, `DocumentRepository.list_visible` (копия `build_acl_clauses`) и `BM25Index._doc_matches_acl`.

- Канонические условия и доменный predicate уже находятся в `domain/knowledge/access_control.py`; адаптеры их используют. Проверять соответствие, сохраняя listing/search: ADMIN видит CLIENT_PRIVATE в listing, но не в search. Не требовать одинакового scope этих двух режимов и не переносить SQL/Qdrant-типы в Domain.
- Контракт-тест: одни и те же пользователи и документы дают одинаковую видимость во всех бэкендах.

**Драйверы.** Два пакета, три режима: psycopg async (SQLAlchemy), asyncpg (listeners), psycopg sync (corpus control/budget). Унификация на psycopg — необязательное отдельное изменение с reconnect/NOTIFY тестами. Удаление asyncpg затрагивает integration fixtures и CI TEST_POSTGRES_URL. Sync сохраняется, пока существуют соответствующие вызовы, независимо от standalone.

**`DatabaseManager`.** Дублирует создание master в двух `_setup_*`. Кластер с репликами и round-robin **сохранить по требованию владельца**; изменяющие UoW должны явно работать с master, чтение с replica — только с допустимой stale-семантикой. Докстринг `DatabaseURL` говорит про asyncpg, а возвращает psycopg.

---

### B. Конфигурация

Параллельно живут два механизма; новый (`SettingsProviderConfiguration` + snapshot) уже победил.

**Удалить:**

- legacy-ветку в `PostgresConfigListener`: `resync_all`, `refetch_and_publish`, `values_equal`, `normalize_*`;
- `config_subscribers.apply_to_settings`, `coerce_and_set`, `_DYNAMIC_FIELDS`, no-op `invalidate_storage_cache`;
- `PostgresConfigBroadcaster._build_payload` с логикой `SENSITIVE_KEYS` и лимитом 7500 байт: новый путь игнорирует payload и просто делает `schedule_sync`, хватит константного NOTIFY;
- `DomainSettingsAdapter.set` (нужен только legacy);
- `config_migration.py` обслуживает активную CLI-команду `config migrate-env`. Сохранять её до решения по legacy env; перенос требует обновления CLI/тестов. При S3-only убрать local-ветки без ослабления проверки credentials.

**Починить:** `config_subscribers` импортирует `get_paddle_ocr` из тяжёлого `ml.ingestion` только ради `cache_clear`. Заменить событием или хуком, зарегистрированным модулем OCR.

---

### C. Ingestion (парсеры, сплиттеры, OCR, preview)

Диспетчеризация по расширению сейчас в четырёх местах: `PARSERS`, if-цепочка в `LangchainDocumentParser._parse`, `PreviewStrategyFactory._EXTENSION_MAP` с `if cls is …`, `IngestionDocumentParser.supports` плюс `settings.supported_extensions`. Реестр `PARSERS` в основном пути не используется.

**Целевая модель:** один `FormatHandler`-реестр. На формат приходится один модуль с `parse() -> list[Section]`, `preview()`, `extensions`. Отсюда:

- `.doc` сейчас ошибочно маппится на parse_docx, preview отклоняет его. LibreOffice не согласован. Вариант без него существует (Apache Tika/POI), но это отдельная зависимость и проверка качества; выбор альтернативы либо отказа от формата пока открыт (вопрос 5);
- `parse_docx` и `parse_docx_sections` обходят документ дважды, и парсер вызывает оба; свести к одному обходу, который отдаёт секции и метаданные;
- `qn` импортируется локально внутри функций при наличии модульного импорта.

**Прочее:**

- `splitting.py`: три пути `_split_structured` / `split_documents_legal` / `_split_char_general` с дублирующимся резолвом размеров (`settings_chunk_size`, `_get_domain_setting`, ручной разбор в `split_documents_legal`). Свести к одному `Chunker(profile)`.
- `GENERAL_SEPARATORS`, `LEGAL_SEPARATORS`, `SENTENCE_SEPARATORS`: проверить использование, часть, вероятно, мертва после `text_chunks`.
- `pdf_diag.py`: `PdfQualityReport` дублирует `domain.PDFQualityReport` (`MLPDFQualityAssessor` копирует поля один в один); `simple_chunk` — третья реализация чанкинга; `check_pdf` — CLI-диагностика с `ocr_hint` и `convert_hint`. Оставить только `assess_*`, диагностику вынести в `scripts/`.
- `guardrails/pdf_adapter.py` дублирует `_page_has_tables` и `classify_page`.
- OCR: сохранить PaddleOCR, Surya, PaddleOCR-VL и режим auto. В PDF auto делает fallback Paddle → Surya; в DOCX auto остаётся на Paddle. Объединять только после теста нужной семантики, сохраняя VL blocks/tables, ошибки и таймауты.
- В `ingestion/__init__.py` реэкспорт `settings` «для патча в тестах»: убрать, править тесты.

---

### D. Retrieval и RAG

**Дублирование в пайплайне:**

- `retrieve_with_decomposition` и `run_retrieval` вызываются с 11 позиционными аргументами в 4 местах; `rerank_and_enrich` принимает 18. Передавать `RagPipelineState` или `RetrievalRequest`.
- `step_self_rag` повторяет цепочку retrieve → rerank из `step_retrieve`. Вынести `RetrievalPipeline.run(query, state)`.
- `step_check_cache`: около 80 строк сборки ключа кэша живут внутри шага. Вынести в `CacheKeyBuilder`.
- `answer_cache.py` и `rag_cache.py` объединить в пакет `cache/`.

**Оставшиеся доменные правила в инфраструктуре.** Часть evidence-правил уже в `domain/rag/evidence`. Текущий профиль документов — `domain/ingestion/profiles/protocol.py::DocumentProfile`; прямой импорт его из Domain RAG запрещён действующим контрактом. Не вводить такую зависимость механическим переносом. `applicable_context`, `linked_context`, `grounded_response`, `evidence_focus`, `table_context` содержат жёстко зашитые русские формулировки конкретных постановлений («сведения, указанные в пункте N настоящего постановления», «При передаче», «В накладной»). Это бизнес-правила, их место в согласованной границе `domain/rag` либо отдельного evidence-профиля; инфраструктура остаётся механикой. Самый рискованный кусок: нужны golden-тесты на ответы.

**Sparse retrieval и оставшийся BM25:**

Текущий runtime уже Qdrant dense+sparse (`rag_retrieval.py:180`, `sparse_vectors.py:43`); sweep использует тот же sparse-путь. Сохранять одинаковые ACL/temporal filters до top-k, стабильные point IDs, обновление обоих векторов и corpus embedding identity. Старые index/snapshot/cron — отдельные кандидаты на очистку, не архитектура, которую нужно вводить заново. Tokenizer/stemmer используются новым encoder и сохраняются.

- `_build` и `add_text` дублируют подсчёт `doc_freq`.
- `remove_text` перестраивает inverted index целиком (O(n)) и `_hash_to_idx` дважды.
- Стеммер: суффиксы `ов`, `ев`, `ые`, `ие` повторяются в списке.
- В `persistence.py` остались FS save/load, при том что `load_configured_bm25_snapshot` работает только с S3.
- `bm25_updater` и `S3SparseIndexAdmin` в текущем коде отсутствуют. `cron_bm25_rebuild` остался, но не зарегистрирован CLI worker. Не восстанавливать этот cron как обязательную часть runtime и не планировать S3 rebuild вместо работающего Qdrant sparse.

**Qdrant и outbox:**

- Декораторы `@pin_corpus_call` и `@retry_on_transient` применены к методам непоследовательно (`delete_by_ids`, `set_document_id_by_source` и другие без них).
- `_patch_metadata` **уже** использует пакетный `set_payload(key="metadata", wait=True)` по фильтру. Сохранить это поведение и `test_qdrant_sparse.py::test_rename_patches_payload_without_replacing_vectors`.

**Гардрейлы.** `PIIDetector` с class-level singleton + `get_pii_detector` / `invalidate` + `PIIRedactorAdapter`: три слоя на один regex-проход.

---

### E. LLM/ML-клиенты и бюджет

**Фабрики** (`factories.py`):

- `ChatOpenAI(... http_client=PaidSync…, http_async_client=PaidAsync…)` скопирован 4 раза; дубль есть и в `benchmark/retrieval.build_llm`.
- `(model or settings).model_copy()` стоит в каждой функции.
- Модель `meta-llama/llama-3.1-8b-instruct` прошита в `factories` и в `judge`.
- Одна `build_chat_model(provider, role)` с таблицей параметров.

**Два вида стоимости.** Хардкод LLM_PRICING находится в `benchmark/metrics.py:147`, не в инфраструктурном metrics collector; case_evaluator использует его как estimated_generation_cost_usd. Это estimate, а не физический ledger. Сначала определить атрибуцию стоимости benchmark к фактическим попыткам и полноту данных; не подменять диагностическую оценку произвольным чтением общего ledger.

**Бюджет, два уровня.** `BudgetRuntime.llm_call`, `guard_paid_http`, `validate_paid_request`, `paid_attempt` против HTTP-транспорта `PaidAsyncTransport` / `PaidSyncTransport`. Блок `provider.max_price` продублирован. Если `llm_call` и `guard_paid_http*` нигде не вызываются, удалить. Оставить транспорт как единственную точку.

**Sync/async двойники.** Не удалять автоматически вместе со standalone: обычный sweep использует sync embeddings, async judge — sync Instructor в потоке, accounting script — sync embedding/rerank. Сначала мигрировать все эти пути, сохранив billing/cancellation/lease, либо оставить нужные реализации.

**`OpenRouterBillingCatalog`.** Тарифы уже хранятся в БД. Проверенные DeepInfra contracts также ограничивают форму запроса, endpoint и input_limit; до отдельного решения сохранять fail-closed allowlist, не заменять произвольно редактируемыми настройками.

---

### F. Worker, джобы, scheduler, события

- Три слоя декораторов вокруг задачи (`one_document_per_principal`, `bill_job_to_initiator`, `run_tracked_job`) со своей математикой `job_try`. Свести к одному `@job(...)`.
- `tasks.py` содержит `# noqa F401` реэкспорт cron «для обратной совместимости». Убрать, поправить импорт в worker config.
- `document_deletion.py` импортирует `run_tracked_job` из `tasks.py`. Вынести в `worker/runtime.py`.
- `run_full_ingest` и `run_single_ingest` почти идентичны. Объединить.
- Два планировщика пока сохранить; API один. Метрики и config resync процессные — каждому API/worker нужен свой snapshot и восстановление после lost NOTIFY. Возможный перенос глобальных outbox maintenance задач в Arq требует отдельной проверки блокировок, идемпотентности и safety net. `cron_bm25_rebuild` сейчас не зарегистрирован; startup log про bm25 устарел.
- `collect_infra_metrics` каждые 30 секунд ходит в Ollama `/api/ps` даже при провайдере OpenRouter и логирует warning. Добавить условие по провайдеру.
- `publish_sweep_event` создаёт новый arq-пул на каждое сообщение. Использовать общий `redis_client`.

---

### G. Benchmark и sweep

Самый большой источник мёртвого и дублированного кода.

**Удалить (после подтверждения):**

- Standalone и его retrieval.py — кандидаты без найденных production opt-in. Общая механика fusion уже в application.search.HybridRetriever; убрать benchmark-реэкспорты и тестовые импорты вместе с удаляемым путём, сохранив основной RAG. Nongrouped judge/context/evidence ветка активна при benchmark_judge_grouped_enabled=False (`case_evaluator.py:204`), sync judge нужен async wrapper/accounting script. Выбор только grouped — изменение поведения и checkpoint-контрактов, а не удаление доказанно мёртвого кода;
- файловые чекпоинты: `FileBenchmarkCheckpoints`, `checkpoint_cleanup.py`, `import_legacy_configuration`, `load_legacy_phase_a`, `latest_single_config`;
- JSONL I/O в benchmark_history отделять от используемого web regression-check `compare_runs`; adapter и comparison сохранять либо переносить с обновлением потребителей. `legacy_import.py` используется CLI. Постоянные файловые outputs/checkpoints удалять или переносить в рамках S3-only; явный CLI export из БД — отдельный пользовательский сценарий;
- Пункт про SUCCESSIVE_HALVING → Optuna уже неактуален: текущий BenchmarkStrategy содержит GRID, RANDOM, OPTUNA_TPE, такого алиаса в стратегии нет.

**Структура:**

- `SweepEngine` кладёт состояние в атрибуты вне `__init__` (`corpus`, `_checkpoints`, `_baseline_config`, `_legacy_single_config`, `_checkpoint_root`) и читает через `getattr(self, "_x", default)`. Заменить объектом `SweepRun`.
- Список исключённых из снимка полей настроек (`key/password/secret/url/dir` + 4 benchmark-имени) повторён в `sweep_engine` ×2 и `runner`. Вынести `settings_fingerprint()`.
- Таск `run_benchmark` в `tasks.py` дублирует логику полной оценки из sweep. Сделать общий `BenchmarkExecutor`.

---

### H. Мелкие подсистемы

- **`adapters/`**: проверять конкретную избыточность с потребителями. Порты сохраняют границы слоёв независимо от количества реализаций; JobEnqueuerPort содержит несколько операций. Массовая замена на Callable из плана исключена; fake redactor/extractor/history уже применяются в тестах.
- **`health`**: `OllamaProbe`, `QdrantInfo` и `SystemHealthProbe` дублируют друг друга. Общий `HttpProbe`. Убрать «admin/ re-exports for backward compat».
- **`api_key_provider`**: sentinel `__MISS__` возвращается как тот же `MISS`, что и «нет в кэше». Негативное кэширование не работает: чинить или удалить sentinel.
- **`rate_limit/policies.py`**: 6 повторяющихся блоков заменить таблицей данных.
- **`utils/utils.py::Singleton`**: проверить использование, вероятно удалить.
- **Docstring-мусор**: `infrastructure/__init__` перечисляет HuggingFace и scheduling, которых там нет; в `env.py` написано «KinTree-style». Переписать после рефакторинга.

---

## 5. Целевая структура

Необязательный набросок, а не критерий готовности. Не переносить все пакеты ради дерева каталогов; выбирать структуру только при конкретном завершённом срезе. Сохранять текущие контекстные границы Domain/Application.

```
infrastructure/
  platform/      concurrency.py, leased.py, resilience/, logging/, pg_listener.py
  persistence/   models/, migrations/(baseline), repositories/, uow.py
  settings/      provider_configuration.py, listener.py, endpoint_policy.py
  ingestion/     formats/{pdf,docx,rtf,md,txt}.py, registry.py, chunker/, ocr.py, markdown_table.py
  retrieval/     bm25/, vector/(qdrant, outbox), cache/, pipeline/, guardrails/
  llm/           factories.py, clients/, budget/(ledger, transport)
  jobs/          runtime.py, tasks/, cron.py
  benchmark/     executor.py, judge.py, sweep/, checkpoints.py
  storage/, auth/, rate_limit/, health/, metrics/
```

---

## 6. Баги, найденные по дороге

Чинить отдельными PR, с тестом на каждый, до или после соответствующего среза.

| # | Место | Описание |
|---|---|---|
| 1 | `chunk_search.build_exact_search_stmt` | `re.escape` применён к LIKE-паттерну: экранируются не те символы; `% word %` не находит слово в начале и конце чанка |
| 2 | `benchmark/retrieval.py` и `sweep_settings.py` | `get_setting("rag.rrf_k")` против `get_setting("rrf_k")`: разные ключи для одной настройки |
| 3 | `logging_config.py` | `detailed_stdout` использует `_default_formatter` вместо `_detailed_formatter` |
| 4 | `api_key_provider.get_cached` | негативный кэш не работает (sentinel возвращает `MISS`) |
| 5 | `qdrant_vector_store_repository._patch_metadata` | **Уже исправлено**: один set_payload по фильтру с key=metadata; нового fix не требуется |
| 6 | `worker/sweep.publish_sweep_event` | новый arq-пул на каждое сообщение; закрывается в finally, поэтому подтверждён overhead, а не утечка |
| 7 | `metrics._collect_ollama_metrics` | запросы в Ollama каждые 30 с даже при провайдере OpenRouter |
| 8 | `document_repository.reconcile_indexing_documents` | докстринг «mark DONE», код возвращает все INDEXING-документы |
| 9 | `config_parameter_repository.get_by_key` | потенциальный MultipleResultsFound при разных namespace, но app/scripts вызовов не найдено; кандидат на удаление, не доказанный runtime-сбой |
| 10 | `chunk_pagination._paginate_backward` | `next_cur` и `prev_cur` берутся от одного и того же элемента |
| 11 | `.doc` | парсер принимает формат, но читать не умеет; preview падает |
| 12 | `chunks` UNIQUE | **Устарело как баг уникальности**: двухколоночный UNIQUE уже восстановлен; проверить только избыточность тройного |
| 13 | tz | naive/aware расхождения (`edited_at`), `Mapped[int]` при `nullable=True` |

---

## 7. Порядок, стратегия PR, риски

**Порядок:**

1. Фаза 0: сохранить существующий baseline, закрыть пробелы конкретной первой задачи. Общая Фаза 1 и срок 1–2 недели не являются обязательством.
2. S3-only (ниже) и подтверждённые bugfix можно выполнять независимо. Срез A обязателен перед заменой истории миграций, но не перед каждым изменением приложения.
3. Срезы B/G — после выбора конкретных legacy-возможностей. G не считать автоматически безопасным: есть живые sync, CLI, checkpoint и web regression-потребители.
4. Срез C (ingestion), затем D (RAG), затем E (LLM/budget).
5. Срезы F (worker) и H (мелочи).

**Правила PR:**

- Один срез: одна ветка и серия PR ≤ ~600 строк диффа.
- Сначала baseline, потом перенос, потом удаление старого. Переходный код допустим на ограниченный срок, если нужен для проверяемой совместимости.
- Смена baseline — в срезе A с инструкцией создания пустых БД. Новые исправления БД могут требовать собственной миграции в соответствующем fix PR.
- Размер диффа не критерий корректности: characterization/security/concurrency-тесты и исправления могут добавлять строки. Не сокращать проверки ради числа строк.

**Главные риски:**

| Риск | Как снижаем |
|---|---|
| Новый baseline | Разрешено пересоздать все БД; проверить пустую схему, ограничения, индексы и seed-эффекты до удаления цепочки |
| Вынос юридических regex из `ml/rag` в доменный профиль | Golden-тесты на ответы до переноса |
| Очистка старого BM25 и возможная замена listener driver | Сохранить текущий Qdrant sparse и tokenizer/stemmer; отдельно проверить reconnect/NOTIFY |
| Потеря поведения при слиянии дублей | Characterization-тесты (Фаза 0), ревью дифов отдельным проходом |
| Изменение Redis/outbox/checkpoint contracts | Rolling-деплоя пока нет; согласованный restart API/worker, версии ключей и учёт незавершённых попыток всё равно проверить |

---

## 8. Как исполнять план

Одним промптом выполнять не стоит: объём не помещается в контекст с нужной точностью, у модели нет обратной связи от тестов и линтеров, а часть решений принимает человек.

1. **Серия маленьких задач**, по одной на PR: цель, список затронутых файлов, критерий готовности, команды проверки.
2. **Агентный режим с доступом к репозиторию и командам** (запуск тестов, `ruff`, `mypy`, `import-linter`, `alembic check`, grep по потребителям).
3. **Сначала тесты**: characterization-тесты можно сгенерировать моделью, корректность проверяет человек.
4. **Сильную модель** тратить на проектирование (`LeasedResource`, ACL-интерпретатор, вынос правил в доменный профиль) и ревью; механические замены (enum вместо строк, единый логгер) делать быстрой моделью или скриптом.
5. **Независимые срезы** (например, G и C) вести параллельно в отдельных сессиях.
6. **Ревью отдельным проходом**, в идеале другой сессией: «найди изменения поведения в этом дифе».

**Статус исполнения, 2026-10-21:**

- [x] S3-only: удалены local override/backend/fallback, выбор backend в настройках, API и миграторе; обновлены exports, defaults, fake/S3-тесты и инструкции. Старый FILE_BACKEND не включает локальную запись. Временные файлы S3 сохранены.
  Проверки: baseline 91 passed; после изменения 279 passed, 6 integration skipped; отдельно настоящий одноразовый MinIO — 3 passed, 3 skipped (Redis/PostgreSQL не подключались). `task lint` прошёл без изменения правил/baseline; `git diff --check` прошёл; `graphify update .` выполнен. Рабочие сервисы и данные не изменялись.
- [x] Негативный API-key cache: known-invalid возвращает None, повтор не читает БД до истечения TTL; regression на dependency → provider → Redis fake, проверены MISS и отказ Redis. 62 целевых теста прошли, `task lint` прошёл; baseline/правила не менялись.
- [x] Курсоры обратной страницы: next кодирует последний элемент страницы. Regression на настоящем PostgreSQL воспроизвёл оба случая до исправления; после исправления 78 целевых тестов прошли, `task lint` прошёл.
- [x] Exact search: literal regex с границами слов вместо ошибочного ILIKE; 7 PostgreSQL-регрессий воспроизвели дефект до исправления. После исправления 108 целевых тестов (ACL, temporal, API) прошли, `task lint` прошёл.
- [x] Reconcile INDEXING: переводит в DONE только документы без незавершённого document-outbox (dead-letter также блокирует), обновляет indexed_at/version и возвращает изменённые ID. PostgreSQL regression до правки падал; после правки 52 теста прошли, включая concurrent skip-locked и rollback/retry. `task lint` прошёл.
- [x] Служебные таблицы metadata: Core Table для cost_rates/cost_usage/corpus_control/corpus_processes; raw SQL адаптеров сохранён. Тест сравнивает metadata с тремя действующими миграциями, включая defaults/PK/FK/индексы и отражённые PostgreSQL CHECK. 58 целевых тестов прошли, 1 Redis restart skipped; `task lint` прошёл (763 source files, 26 import contracts), baseline/правила не менялись.
- [x] Остальные обнаруженные расхождения модели ↔ миграции: metadata описывает пять недостающих индексов и CHECK source_type; chunks.edited_at соответствует существующему timestamptz; выражение индекса identity нормализовано без изменения семантики. Миграция v2l3m4n5o6p7 заполняет только NULL creation_date реестра временем миграции и добавляет NOT NULL. ChatLogModel.user_id допускает None в аннотации. Именованный use_alter для существующего FK best_run_id устраняет предупреждение о цикле. Общая политика UTC и календарные даты не менялись.
  Проверки: полный upgrade на пустой одноразовой PostgreSQL 18.4 и `alembic check` — 0 новых операций; 32 целевых теста прошли, включая полную цепочку миграций, заполнение NULL, сохранение дат, запрет новых NULL, курсоры, exact search и reconcile. Исправлена изоляция integration create_all при наличии таблиц public. Сняты mypy baseline для models.py, I001 для models.py/env.py и baseline пустых строк env.py; единственное точечное type-ignore относится к нетипизированному конструктору JSONB сторонней SQLAlchemy. Все семь проверок task lint выполнены напрямую и прошли (mypy: 764 файла, import-linter: 26 kept); snap-команда task недоступна в окружении. `uv lock --offline` не изменил lock. Рабочие БД не менялись.
- [x] Общий schema baseline `0001`: историческая цепочка (включая промежуточную v2l3m4n5o6p7) заменена одной фиксированной миграцией без импорта runtime-моделей. На одноразовых PostgreSQL 18.4 и 16.4 сверены 25 таблиц, defaults/nullability, PK/FK/UNIQUE, 24 CHECK, физические определения индексов (включая gin_trgm_ops), последовательности и отсутствие migration seeds. Настройки, как и раньше, заполняются при старте. Старые преобразования существующих данных не применимы к пустому baseline; их тесты заменены проверками действующих ограничений и defaults. Downgrade удаляет таблицы в порядке зависимостей, сохраняя общий pg_trgm.
  Integration fixture применяет настоящий baseline в изолированной схеме вместо ORM create_all. В CI добавлен отдельный gate upgrade/check/downgrade и конкурентного idempotent bootstrap; PostgreSQL service user уже имеет CREATEDB. Каталог versions больше не исключён из Ruff; шаблон новых миграций обновлён под современный синтаксис типов, удалены baseline-записи удалённых миграций.
  Проверки: весь integration-набор — **76 passed, 9 skipped** (Redis restart/Qdrant/MinIO без внешних сервисов); дополнительный целевой прогон baseline/bootstrap/API strategies/deployment — **37 passed**. Все **3847 тестов собраны** без ошибок. Все семь lint-проверок выполнены напрямую и прошли; mypy — 764 файла, import-linter — 26 kept. `uv lock --offline` не изменил lock; полный pytest не запускался.
- [x] Срез B, listener/broadcaster: удалены legacy resync/refetch/normalization и JSON payload с параметрами. NOTIFY передаёт пустой сигнал в существующей транзакции; listener перечитывает валидируемый snapshot с writer, сохраняя lock, resync при reconnect/по расписанию, статус ошибки и cleanup задач. Snapshot-конфигурация обязательна при создании listener; Composition собирает её до EventContainer. Тесты старой ветки заменены проверками действующего пути, пустого/старого payload, больших/секретных значений без передачи в NOTIFY и остановки pending sync.
  Проверки: baseline — 81 passed; после изменения — 224 целевых теста прошли. Все семь проверок task lint выполнены напрямую и прошли (mypy — 765 файлов; import-linter — 26 kept); правила/baseline не менялись. Оставшаяся часть B: legacy subscribers/DomainSettingsAdapter.set и OCR cache hook. Полный pytest и настоящий PostgreSQL LISTEN/NOTIFY в этом шаге не запускались; рабочие сервисы и данные не менялись.
- [x] Срез B, оставшийся legacy: удалены apply_to_settings/coerce_and_set/_DYNAMIC_FIELDS, no-op storage subscriber и DomainSettingsAdapter.set; рабочие потребители уже используют валидируемый snapshot. Обработчик сброса PaddleOCR cache находится в модуле OCR и регистрируется Composition, config_subscribers больше не импортирует ingestion. Активный CLI config migrate-env сохранён.
  Проверки: baseline — 189 passed; после изменения — 262 целевых теста прошли. Покрыты старые storage events через bus → listener → snapshot, изоляция доменов/копирование snapshot, ленивое пересоздание PaddleOCR после смены языка, сохранение cache при неизменённых/посторонних/невалидных настройках. Все семь lint-проверок выполнены напрямую и прошли (mypy — 765 файлов; import-linter — 26 kept), baseline/правила не менялись. Полный pytest и реальные OCR/внешние сервисы не запускались; рабочие данные не менялись.
- [ ] Релиз на новый baseline: согласованное окно остановки API/worker и пересоздания PostgreSQL production/dev/stage остаётся отдельным действием. Существующие ревизии не переводятся через stamp; рабочие БД, S3/Qdrant/Redis и выполняющиеся задания в этой сессии не менялись.

**Предыдущая остановка по запросу владельца:** срез metadata был завершён. По новому запросу продолжена сверка модели и миграций; результаты приведены выше. `graphify update .` выполняется после каждого среза кода. Рабочие сервисы и данные не менялись.

**Первая серия задач после аудита:**

1. S3-only отдельным PR: удалить `docker-compose.local-storage.yml`, LocalStorage/fallback и backend-specific config/API branches. Обновить storage exports, enum/setting keys, мигратор env, `.env.example`, тестовые defaults/fakes и инструкции. Не удалять Docker volumes БД/Qdrant/ML caches. Временные файлы парсеров разрешены с cleanup при успехе, ошибке и cancellation. Сохранить БД для runtime history/checkpoints и S3 для документов/файловых артефактов. Сначала проверить S3 workflow без DATA_DIR и MinIO integration; рабочие данные в этом аудите не трогать.
2. Отдельные bugfix: негативный API-key cache; курсоры обратной страницы; exact LIKE; reconcile INDEXING. Каждый с регрессионным тестом по полному пути.
3. Описать служебные таблицы в metadata и проверить baseline на пустой БД. Пересоздание сред выполняется отдельным согласованным релизом; replicas сохраняются.
4. Очистка подтверждённого legacy с исправлением внутренних импортов и patch targets; sync, comparison и действующие CLI не удалять по названию модуля.
5. `.doc` — отдельное решение перед срезом C. Юридические evidence-правила — после фиксации golden-набора и решения границ, без прямой зависимости Domain RAG от ingestion profiles.
