# Открытые вопросы перед рефакторингом `server/app/infrastructure`

Документ дополняет план вертикального рефакторинга. Решения и результаты проверки репозитория зафиксированы ниже; незакрытые вопросы блокируют только связанные изменения.

**Как пользоваться:** у каждого вопроса есть контекст, варианты, моя рекомендация и поле `Решение`. Ответы фиксируем прямо в файле, чтобы они оставались в истории репозитория.

**Приоритеты:**
- 🔴 **Блокер**: без ответа нельзя начинать связанный срез.
- 🟡 **Важно**: влияет на объём работ, но срез можно начать.
- 🟢 **Уточнение**: можно решить по ходу.

**Проверка актуальности: 2026-10-21.** Проверены связанные пути в Domain, Application, Composition, API/CLI, адаптерах и тестах. Graphify использован для навигации; выводы сверены с исходниками. Фактическая схема рабочих БД, их ревизии и качество ответов на реальном корпусе не проверялись. Исходники не менялись.

**Подтверждено владельцем:** среды production/dev/stage; все базы можно пересоздать; поддержку реплик PostgreSQL сохраняем; API пока один, rolling-деплоя нет; внешних потребителей старых импортов/standalone/файловой истории нет. OCR-движки пока сохраняем. Постоянное файловое хранение — только S3-совместимое; временные файлы с очисткой допустимы. LibreOffice для `.doc` не согласован; окончательный вариант обработки `.doc` открыт.

В полях «Решение» разделены факты репозитория, подтверждённые требования и предложения, которые ещё не утверждены.

---

## Сводная таблица

| # | Вопрос | Приоритет | Срез | Статус |
|---|---|---|---|---|
| 1 | Реплики БД (cluster mode) | 🔴 | A | Решено владельцем |
| 2 | Baseline миграций и состояние стендов | 🔴 | A | Baseline допустим; проверка впереди |
| 3 | Standalone-benchmark и sync-клиенты | 🔴 | G, E | Ответ найден; sync сохранить |
| 4 | Файловые чекпоинты, legacy-импорт, JSONL-история | 🔴 | G | Частично; см. S3-only |
| 5 | Формат `.doc` | 🟡 | C | Выбор альтернативы открыт |
| 6 | OCR-движки (Paddle, Surya, VL) | 🟡 | C | Сохранить все три |
| 7 | Стратегия BM25 | 🟡 | D | Исходный вопрос устарел |
| 8 | Вынос юридических правил из `ml/rag` в доменный профиль | 🔴 | D | Частично; golden/границы открыты |
| 9 | Внешние потребители старых реэкспортов | 🔴 | все | Ответ найден |
| 10 | `config_migration.py` | 🟡 | B | Активный CLI; решение открыто |
| 11 | Два планировщика (APScheduler и Arq cron) | 🟡 | F | Ответ найден; перенос не утверждён |
| 12 | Драйверы БД: asyncpg и sync psycopg | 🟡 | A | Факты подтверждены; замена не утверждена |
| 13 | Язык кода и логов | 🟢 | все | Не утверждено |
| 14 | Тесты: что патчится через реэкспорты | 🔴 | все | Ответ найден; baseline выполнен |
| 15 | Два слоя бюджета (`llm_call` и HTTP-транспорт) | 🟡 | E | Legacy-кандидат; runtime сохранить |
| 16 | Таблицы без ORM-моделей | 🟡 | A | Пробел metadata подтверждён |
| 17 | Расхождения модель и миграция | 🟡 | A | Частично устарело |
| 18 | Единицы времени и таймзона | 🟡 | A | Предложение; внешние SQL неизвестны |
| 19 | Контракты DeepInfra и тарифы в коде | 🟢 | E | Контракты сохранить до решения |
| 20 | Негативный кэш API-ключей | 🟢 | H | Дефект подтверждён |
| 21 | Тонкие адаптеры и порты | 🟢 | H | Массовую замену исключить |
| 22 | Политика совместимости и релизов | 🔴 | все | Условия релиза уточнены |
| 23 | Баги, найденные при чтении | 🟡 | разные | Проверено; см. результаты |
| 24 | Только S3; удалить local backend | 🔴 | storage, B, C, G, H | Решено владельцем |

---

## 🔴 Решения по исходным блокерам

### 1. Реплики БД (cluster mode)

**Решение:** поддержку реплик **сохранить** (требование владельца). Реальная конфигурация реплик не проверялась. `server/app/infrastructure/uow_factory.py:107` создаёт одну сессию с `master=False` по умолчанию; `database/database.py:404` выбирает read/write engine, а не отдельную read-сессию внутри UoW. Поэтому формулировка «свои read/write sessions» в докстринге неточна. Перед изменением маршрутизации проверить каждый изменяющий сценарий: он должен явно выбирать master; ограничения и гонки проверять в PostgreSQL. Удаление cluster mode исключено из плана.

---

### 2. Baseline миграций и состояние стендов

**Решение:** production/dev/stage можно пересоздать (подтверждено владельцем). Новый baseline допустим для **пустых** баз; `stamp` не нужен как стратегия переноса данных. В каталоге сейчас 56 файлов миграций. Старую цепочку сохранять до проверки полного baseline: DDL, индексы, CHECK/FK, расширения, служебные таблицы и обязательные seed/data-эффекты. Реальные `alembic current` и схема сред неизвестны. Никакие базы в рамках аудита не пересоздавались.

---

### 3. Standalone-benchmark и sync-клиенты

**Решение:** вне репозитория standalone и старые импорты не используются (владелец). Производственные вызовы `standalone=True` поиском не обнаружены; явный opt-in проверяют `server/tests/test_worker_resource_lifecycle.py:112` и `test_solid_characterization.py:89`. Удаление standalone остаётся кандидатом, а не автоматическим удалением всего sync-кода. Обычный sweep использует `embed_query_sync` в `server/app/infrastructure/benchmark/sweep_scoring.py:272`; `judge_answer_async` сам вызывает sync Instructor через поток (`benchmark/judge.py:377`). Sync embedding/rerank/judge также проверяет `server/scripts/check_paid_accounting.py:148`. Сначала заменить эти вызовы или сохранить нужные sync-клиенты/транспорт.

---

### 4. Файловые чекпоинты, legacy-импорт и JSONL-история

**Решение:** внешних потребителей нет; необходимость сохранять данные БД снята владельцем. Но код не мёртв: `sweep_engine.py:189` выбирает файловый checkpoint при переданном root, `runner.py:150` — для resume/out_dir; `checkpoint.py:44` содержит импорт старой оплаченной работы. Web composition уже создаёт DB checkpoints **без** legacy root (`server/app/composition/service_providers.py:40`). CLI `benchmark import-legacy` активен (`presentation/cli/commands/benchmark.py:61`), локальный экспорт идёт через `BenchmarkExportService`. `BenchmarkHistoryAdapter` нужен web regression-check: `presentation/api/routes/benchmark_admin.py:467` → `application/benchmark/result_service.py:96` → `benchmark_history_adapter.py:7` → чистый `compare_runs`. Удалять JSONL I/O отдельно от сравнения. По требованию S3-only (вопрос 24) постоянные файловые checkpoints/history/reports подлежат удалению или переносу в БД/S3, с обновлением CLI и тестов.

---

### 8. Вынос юридических и регуляторных правил из `ml/rag`

**Решение:** профили general/legal/decree уже регистрируются (`server/app/infrastructure/domain_profile/__init__.py:23`), но это `DocumentProfile` из `domain/ingestion/profiles/protocol.py:18`, контракт parsing/classification без `evidence_rules`. Значительная часть evidence-правил уже в `domain/rag/evidence/*`; инфраструктурные `applicable_context.py:7`, `evidence_focus.py:7`, `grounded_response.py:5` их импортируют. Перенос в старый путь `domain/domain_profile` неактуален. Контракт запрещает прямую зависимость Domain RAG от ingestion profiles (`server/tests/test_architecture_contracts.py:141`); новое правило профилирования требует отдельного решения границ. Аннотации required_facts/expected_refusal и тесты есть (`test_benchmark_evidence.py`, `test_evidence_applicability.py`), есть `server/scripts/validate_evidence_baseline.py:74`. Наличие утверждённого golden-набора на рабочем корпусе этим не доказано. Начинать с characterization существующего поведения; не обещать перенос без изменения ответов без сравнения.

---

### 9. Внешние потребители старых реэкспортов

**Решение:** внешних потребителей нет (владелец), внутренние найдены. Список точек миграции:

- `worker.tasks` cron-реэкспорты: `server/app/presentation/cli/commands/worker.py:21`; внутренние `worker/document_deletion.py:5`, `worker/sweep.py:159` импортируют `run_tracked_job`.
- `ml.ingestion`: `composition/infrastructure.py:183`, `ml/langchain_document_parser.py:17`, `ml/config/config_subscribers.py:26`, `ml/extraction/extraction_adapter.py:8`, `ml/preview/docx_preview_strategy.py:15`, `ml/preview/rtf_preview_strategy.py:14`; тесты `test_ingestion_text.py`, `test_ingestion_parsers.py`, `test_ingestion_docx.py`, `test_chunking_characterization.py`, `test_rag_logic.py`, `test_ocr_vl.py`.
- `benchmark.runner` реэкспорт метрик: `server/tests/test_benchmark_runner.py:29`; `benchmark.__init__` используется тестами через `from infrastructure.benchmark import ...`, в том числе `test_benchmark_characterization.py`. Приватные `_safe_avg` и `_sanitize_model_name` ещё экспортируются.
- Health composition уже использует `infrastructure.health.system_health_probe` (`composition/infrastructure.py:302`); старый `admin/` как текущий путь не подтверждён.
- `domain.utils.percentile` устарел: реэкспорт сейчас `domain.benchmark.stats.percentile` в `infrastructure/benchmark/metrics.py:143`.

Менять импорты и patch targets в том же PR; список относится к проверенным старым путям, не является запретом нормальных импортов Infrastructure из Composition.

---

### 14. Тесты: что патчится через реэкспорты

**Решение:** тесты находятся в `server/tests/`, integration — в `server/tests/integration/`; команда из `Taskfile.yml:268`: `cd server && uv run pytest -v`. Characterization уже есть: chunking, document processor, RAG service/policy, benchmark и sweep. Не начинать Фазу 0 с их повторного написания. Патчи реэкспортов подтверждены: `test_ingestion_text.py:285` патчит `infrastructure.ml.ingestion.settings`; `test_rag_service_stream.py:122` и `:150` — имена в `rag_steps`. Фейки портов уже применяются (`test_chat_contracts.py:73`, `test_document_processor_characterization.py:307`). CI требует 85% только для ACL и answer_cache, это не показатель покрытия всего RAG/бюджета. До переноса выбирать конкретные сценарии и фиксировать baseline. Результаты выполненных проверок приведены в конце документа.

---

### 22. Политика совместимости и релизов

**Решение:** владелец подтвердил один API, отсутствие rolling-деплоя и возможность пересоздать все БД. Значит, для baseline допустим согласованный останов API/worker и запуск на новой схеме; обратная совместимость старого и нового worker не обязательна для текущего способа релиза. Это **не** разрешение молча потерять выполняемую работу при обычном рефакторинге. Версии Redis keys/outbox/checkpoints и corpus binding проверять отдельно: разрешение пересоздать PostgreSQL не означает разрешение удалить S3/Qdrant или отключить учёт незавершённых платных попыток. Runtime sparse больше не зависит от S3 BM25 snapshot (вопрос 7). Конкретный релиз/reset в рамках аудита не выполнялся.

---

## 🟡 Важные

### 5. Формат `.doc`

**Решение:** текущая поддержка бинарного `.doc` сломана: `config.py:222` разрешает расширение; `ml/langchain_document_parser.py:196` ведёт `.doc` в DOCX-парсер; `ml/ingestion/registry.py:12` делает тот же mapping; `ml/preview/factory.py:29` отклоняет preview. Владелец сначала попросил сохранить `.doc`, затем отказался от LibreOffice и допустил отказ от формата, если альтернативы нет. Альтернатива существует: [Apache Tika OfficeParser](https://tika.apache.org/3.2.0/formats.html) извлекает текст/метаданные старого Word через Apache POI; [CLI работает как Java jar](https://tika.apache.org/3.2.0/gettingstarted.html). Это новый компонент, не готовый адаптер проекта; таблицы, структуру, таймауты и preview надо проверять на фикстурах. Выбор Tika либо явного отказа от `.doc` **не утверждён**. Не удалять формат и не добавлять конвертер автоматически.

---

### 6. OCR-движки

**Решение:** сохранить текущие OCR-движки (владелец). Их уже **три**: PaddleOCR, Surya и PaddleOCR-VL; `auto` — режим выбора, а не отдельный движок. В PDF `auto` пробует Paddle, затем Surya при пустом тексте (`ml/ingestion/ocr.py:159`). В DOCX `auto` вызывает Paddle **без** такого fallback, а Surya находится в `else` (`ml/preview/docx_preview_strategy.py:139`); исходное утверждение «auto вызывает Surya» неверно. Общий интерфейс должен сохранить VL, разметку блоков/таблиц и различия ошибок. Реальный выбор в production неизвестен; default `config.py:140` — PaddleOCR.

---

### 7. Стратегия BM25

**Решение:** исходный выбор «incremental BM25 в API или nightly S3 rebuild» **устарел**. `ml/rag/rag_retrieval.py:180` запускает dense+sparse в Qdrant с одинаковым фильтром, sparse возвращает payload и point IDs без разрешения BM25-хешей. `repositories/vector/sparse_vectors.py:43` формирует оба вектора для одной точки; `qdrant_ops.py:127` включает IDF. Sweep использует тот же sparse-поиск (`benchmark/sweep_scoring.py:291`). `S3SparseIndexAdmin` и `bm25_updater` в текущем коде не найдены. Старые BM25 index/snapshot/registry/cron ещё существуют; `cron_bm25_rebuild` не включён в functions/cron_jobs CLI worker (`presentation/cli/commands/worker.py:60`, `:81`). Не добавлять ночную пересборку обратно ради устаревшего плана. Проверять удаление оставшихся потребителей отдельно, сохранив tokenizer/stemmer, используемые новым sparse encoder.

---

### 10. `config_migration.py` (env → БД)

**Решение:** `config_migration.py` — активный CLI-сценарий, а не бесхозный скрипт: `server/app/presentation/cli/commands/config.py:35` объявляет `config migrate-env` с dry-run/apply/trim-env и импортирует мигратор. Текущий модуль содержит более 340 строк; утверждение про незавершённые stages 3–12 в его докстринге устарело. Пересоздание БД не доказывает отсутствие legacy env на средах. До отдельного решения сохранить CLI, либо перенести реализацию с исправлением импорта. При S3-only убрать local-ветку проверки credentials и обновить `test_config_env_migration.py`.

---

### 11. Два планировщика

**Решение:** API сейчас один, rolling-деплоя нет (владелец). APScheduler по-прежнему выполняет config resync, outbox dispatch и reconcile (`infrastructure/scheduling/scheduler.py:59`). Arq cron содержит cleanup/recover, но **не** BM25 в 03:00 (`presentation/cli/commands/worker.py:81`); сообщение startup про bm25 устарело. Worker имеет собственный периодический config resync (`worker.py:144`). Переносить процессный config resync в единственный Arq cron нельзя: каждому процессу нужен свой snapshot и восстановление после потерянного NOTIFY. Для outbox перенос — отдельное решение с сохранением LISTEN и safety net. Идемпотентность и блокировки проверять по каждой задаче, не по `max_instances=1`.

---

### 12. Драйверы БД

**Решение:** psycopg async/sync и asyncpg действительно используются; это два пакета, три режима доступа. `events/postgres_config_listener.py:16` и `postgres_outbox_listener.py:15` используют asyncpg. Его удаление затрагивает ещё integration fixtures, `TEST_POSTGRES_URL` и CI (`.github/workflows/lint.yml`, `server/tests/integration/test_postgres_transactions.py:8`). Причина выбора asyncpg из репозитория не установлена; замена на psycopg — предложение, не предпосылка рефакторинга. Sync psycopg нельзя удалять вслед за standalone: он нужен budget policy и corpus control. Проверить reconnect, потерянные NOTIFY и закрытие соединений до замены драйвера.

---

### 15. Два слоя бюджета

**Решение:** поиск вызовов в app/scripts не нашёл использования `BudgetRuntime.llm_call` и `guard_paid_http*`; связанный `paid_attempt` используется только внутри `budget/runtime.py`. Но их прямо проверяет `server/tests/test_cost_budget.py:113`, `:126`, `:204`, `:309`. Это кандидат на удаление legacy entry points с переносом значимых гарантий в тесты транспорта. Сам `BudgetRuntime` **живой**: `composition/infrastructure.py:405` конфигурирует его, `PaidAsyncTransport`/`PaidSyncTransport` используют admission/policy/rates/run_sync. Нельзя удалять runtime целиком или потерять fail-closed validation, резерв, учёт физических retry, cancellation и reconciliation.

---

### 16. Таблицы без ORM-моделей

**Решение:** отсутствие cost_rates/cost_usage/corpus_control/corpus_processes в metadata подтверждено по `database/models.py`; таблицы создаются `migrations/versions/o5f6g7h8i9j0_cost_budget.py:14` и `p6g7h8i9j0_corpus_switch.py:15`, затем изменяются `q7h8i9j0k1l2_cost_attempt_audit.py`. `migrations/env.py:47` не исключает их из autogenerate, поэтому они могут предлагаться к удалению. Добавить точное описание в SQLAlchemy metadata (ORM или Core Table); переписывать рабочий сырой SQL для этого не обязательно. До baseline сверить ограничения и индексы, не только имена колонок.

---

### 17. Расхождения модель ↔ миграция

**Решение:** часть исходных замечаний неверна. `ChunkModel` уже имеет UNIQUE `(document_id, chunk_index)` (`models.py:326`), миграция `t0j1k2l3m4n5_unique_chunk_index.py:13` его восстанавливает. Трёхколоночный UNIQUE избыточен как гарантия уникальности, но наличие дублей позиций больше не следует из модели/актуальной цепочки. `BenchmarkCheckpointModel.updated_at` объявлен явно (`models.py:562`), наследование от LinkedBaseModel этому не мешает — это не баг. Подтверждены `ChatLogModel.user_id` с неверной nullable-аннотацией (`models.py:378`), naive `chunks.edited_at` (`:346`) при tz-aware миграции, отсутствие FK у benchmark_runs.job_id (`:492`). Allowed-values читаются из dict и list (`repositories/config/sqlalchemy_config_parameter_repository.py:109`); upsert обновляет только value (`:96`), но необходимость менять декларацию вместе со значением — отдельный контракт, не доказанная ошибка миграции. GIN trgm и остальные индексы сверять на пустой БД после полного upgrade, не по одной старой миграции.

---

### 18. Единицы времени и таймзона

**Решение:** aware datetime с Europe/Minsk уже обозначает момент времени; проблема не в самом `datetime.now(settings.tz)`, а в смешении с naive и сессионной семантикой. `database/basemodel.py:26` задаёт timezone-aware creation_date; `database/database.py:85` устанавливает Europe/Minsk; `models.py:346` задаёт naive edited_at. Внешние SQL/BI-зависимости репозиторий не доказывает. Для нового baseline предлагается aware UTC, но бизнес-смысл полей date/effective_from/effective_to сохранять: календарные даты не конвертировать как timestamp. Правило часового пояса ещё не утверждено владельцем.

---

### 23. Баги, найденные при чтении

**Решение:** исходные 10 гипотез проверены:

| # | Результат и сценарий |
|---|---|
| 1 | Подтверждён: API `routes/search.py:27` → `application/search/service.py:24` → chunk repository → `chunk_search.py:126`. EXACT строит ILIKE `% word %`: слово на краю не совпадает; `re.escape` не экранирует SQL `%`/`_` как литералы. |
| 2 | Подтверждён для standalone: `benchmark/retrieval.py:106`, `:161` читают `rag.rrf_k`; `config.py:26` делает getattr без перевода dotted key. Без явного override возникает AttributeError. Обычный sweep читает `rrf_k` (`sweep_scoring.py:154`). |
| 3 | Подтверждён: `logging/logging_config.py:128` выбирает default formatter для detailed_stdout. |
| 4 | Подтверждён полный путь негативного API-key cache; см. вопрос 20. |
| 5 | **Уже исправлен**: `repositories/vector/qdrant_vector_store_repository.py:192` делает один set_payload по фильтру с key=metadata и wait=True. Проверяется `test_qdrant_sparse.py:144`. |
| 6 | Подтверждён расход ресурсов: каждый `worker/sweep.py:24` создаёт и закрывает arq pool для одного progress event. Это не доказательство утечки: finally закрывает пул. |
| 7 | Подтверждён лишний вызов: `metrics/metrics.py:405` без проверки provider вызывает `_collect_ollama_metrics`; `scheduling/scheduler.py` запускает сбор периодически. |
| 8 | Подтверждён: `document/sqlalchemy_document_repository.py:387` только выбирает все INDEXING id, не проверяет outbox и не меняет статус; `vector/outbox_dispatcher.py:225` считает и логирует их как исправленные. |
| 9 | Условный дефект API репозитория: `config/sqlalchemy_config_parameter_repository.py:26` использует scalar_one_or_none без domain_key. В app/scripts текущих вызовов get_by_key не найдено; действующий путь get_by_key_and_domain фильтрует namespace. Это кандидат на удаление, не подтверждённый production-сбой. |
| 10 | Подтверждён: API chunk listing → `application/chunks/queries.py:105` → `chunk_crud.py:250` → `chunk_pagination.py:73`. Для обратной страницы [2,3,4] оба курсора кодируют 2; next должен продолжать после 4, иначе при переходе вперёд будут повторы. |

Отдельные fix PR с регрессионными тестами; исходники в аудите не исправлялись. Состояние `.doc`, UNIQUE и timezone уточнено в вопросах 5/17/18.

---

## 🟢 Уточнения

### 13. Язык кода и логов

**Решение:** выбор языка — соглашение команды, его нельзя вывести из смешанного текущего кода. Предлагается английский для нового кода/логов/комментариев, русский для продуктовых сообщений и промптов; массовый перевод не включать в функциональные срезы. Владелец это соглашение ещё не утверждал; работу по корректности оно не блокирует.

---

### 19. Контракты DeepInfra и тарифы в коде

**Решение:** три контракта DeepInfra действительно зашиты в `budget/provider_catalog.py:12`; быстрый LLM — в `ml/clients/factories.py:240` и `benchmark/judge.py:167`. **Тарифы уже данные БД**, а ModelBillingContract дополнительно ограничивает endpoint, форму запроса, operation и input_limit. Это граница безопасного биллинга, не просто список цен. Автоматический перенос в произвольные редактируемые настройки может ослабить fail-closed проверку. Сохранить проверенные контракты до отдельного решения; конфигурируемую быструю модель рассматривать отдельно. Необходимость новых моделей владелец не подтверждал.

---

### 20. Негативный кэш API-ключей

**Решение:** дефект негативного кэша подтверждён по полному пути: `auth/api_key_provider.py:65` превращает known-invalid sentinel в MISS → `presentation/api/auth_dependencies.py:58` снова вызывает lookup БД и записывает тот же sentinel. Повтор одного неверного ключа снова читает БД. Для исправления достаточно различать отсутствие записи и cached-invalid (например, возврат None при sentinel); положительный путь и отзыв ключей сохранить. Наличие существующего кэша — основание починить контракт, но не доказательство полной защиты от перебора разных ключей.

---

### 21. Тонкие адаптеры и порты

**Решение:** порты уже распределены между `application/ports`, `application/chat/ports`, `application/documents/ports`, `application/admin/ports`, `application/benchmark/ports`. Они реально подменяются: `test_chat_contracts.py:73` передаёт fake redactor, `test_document_processor_characterization.py:307` — StubExtractor, `test_benchmark_database_history.py:30` проверяет history adapter с fake UoW. `JobEnqueuerPort` содержит шесть операций (`application/ports/job_enqueuer.py:26`), а не одну функцию. Количество production-реализаций не является критерием ненужности порта: он сохраняет направление зависимостей. Массовую замену на Callable исключить из обязательного плана; упрощать конкретную границу только с её потребителями и тестами. Разрешение на произвольную смену портов не получено.

---

## Дополнительное решение: только S3

### 24. Удаление постоянного local storage

**Решение владельца:** отказаться от `docker-compose.local-storage.yml` и постоянного local backend во всех средах. Документы и файловые артефакты — S3-совместимое хранилище (в том числе MinIO); checkpoint/history рантайма остаются в PostgreSQL. Временные файлы для парсеров допустимы с гарантированной очисткой. Это требование к файловому backend приложения, не команда удалять volumes PostgreSQL/Qdrant/ML caches.

**Найденный путь:** override задаёт FILE_BACKEND=local и bind mount обоим процессам (`docker-compose.local-storage.yml:1`); default уже S3 (`server/app/config.py:215`), но `storage/file_storage.py:532` на любое другое значение молча выбирает LocalStorage. Экспорт LocalStorage есть в `storage/__init__.py:7`; `test_file_storage_async.py` тестирует local реализацию; `tests/conftest.py:39` устанавливает FILE_BACKEND=local; `test_config_env_migration.py` использует local для обхода S3 credential requirements. API preview выбирает возможности через `StorageConfig.file_backend` (`presentation/api/routes/admin_quality.py:178`, `:324`), а мигратор — credentials через backend (`infrastructure/config_migration.py:260`). Удаление одного compose-файла требование не выполняет.

**Объём отдельного изменения:** удалить override, LocalStorage и fallback; убрать выбор backend/связанные ветки enum, config, API и мигратора; обновить тестовые defaults и заменить local I/O тесты содержательными S3/fake-storage сценариями; синхронизировать `server/.env.example:36` и действующие инструкции. Старое FILE_BACKEND=local не должно незаметно включать локальную запись. Сохранить временные файлы S3 download/parsing с cleanup при ошибке и cancellation. Проверить существующие `tests/integration/test_s3_storage_minio.py` и сценарии без DATA_DIR. Само удаление в этом аудите не выполнялось.

## Проверки и оставшиеся решения

2026-10-21 выполнены:

- `cd server && UV_CACHE_DIR=/tmp/rag-audit-uv-cache uv run --frozen --no-sync pytest tests/test_acl.py tests/test_bm25_acl.py tests/test_answer_cache.py tests/test_curator_role.py tests/test_ingestion_acl_consistency.py tests/test_architecture_contracts.py -q --tb=short` — **274 passed**.
- `cd server && UV_CACHE_DIR=/tmp/rag-audit-uv-cache uv run --frozen --no-sync pytest tests/test_retrieval_acl.py tests/test_paid_transport.py tests/test_qdrant_sparse.py tests/test_chunking_characterization.py tests/test_worker_resource_lifecycle.py tests/test_benchmark_runner.py tests/test_benchmark_database_history.py -q --tb=short` — **107 passed**, 5 предупреждений DeprecationWarning от SWIG.
- `cd server && PYTHONPATH=app UV_CACHE_DIR=/tmp/rag-audit-uv-cache uv run --frozen --no-sync lint-imports` — **26 kept, 0 broken**, с существующими исключениями (28 в DDD layers).

Retrieval/transport сначала зависали в sandbox. Минимальный `asyncio.to_thread(lambda: 1)` также завис там, но выполнился вне sandbox; второй набор тестов успешно повторён вне sandbox. Это не подтверждённая регрессия проекта.

Не запускались полный pytest, полный task lint, PostgreSQL/MinIO integration, alembic check/current на рабочих базах, реальные LLM/OCR и golden-оценка корпуса. Наличие теста не приравнивается к проверке всех сценариев соответствующего среза.

До исполнения нужны решения только для соответствующей задачи: вариант `.doc` без LibreOffice; сохранение/удаление внутренних CLI legacy-функций; профильные evidence-правила и golden-набор; политика UTC/внешние SQL. Язык комментариев и конфигурируемость моделей не блокируют старт. Не требовать у владельца повторно расположение тестов, внутренние импортёры и факт использования runtime — ответы уже найдены.
