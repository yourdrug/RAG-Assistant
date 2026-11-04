# Полная ревизия RAG Assistant

Дата: 2026-10-30. База: `453aeb2`, текущее рабочее дерево. Scope: весь репозиторий с углублённой проверкой backend по семи областям `review-full`.

## Executive summary

Подтверждено **18 находок: CRITICAL — 0, HIGH — 2, MEDIUM — 13, LOW — 3**. Два HIGH-дефекта: переименование ожидающего обработки документа удаляет файл по ключу, сохранённому в задании worker; recovery ошибочно объявляет долгие работающие задачи осиротевшими, игнорируя heartbeat. Ошибки парсинга также могут завершаться статусом DONE без индексации. Эти сценарии требуют исправления до эксплуатации длительных ingestion/benchmark задач.

В проверенном пути не подтверждён обход межпользовательского ACL. Dense/sparse retrieval применяют фильтры доступа; найденные документы и кэш повторно проверяются по актуальным данным БД. При этом вспомогательная LLM вызывается до input guardrail, а API-key кэш несовместим с форматом authentication DTO. Проверенные зависимости Domain/Application направлены корректно, основной API/worker lifecycle освобождает ресурсы. Это результат ограниченного аудита, а не доказательство отсутствия иных дефектов.

Целевые тесты преимущественно проходят; один тест cancellation стабильно падает из-за устаревшего fake-контекста до проверяемого действия. Реальные PostgreSQL/Qdrant/Redis/S3/LLM интеграции и нагрузочные испытания не запускались. Исходники не изменялись; файл отчёта — единственный результат записи аудитора. Пользовательские изменения `server/app/presentation/api/routes/chat.py` и новый `server/tests/test_chat_openapi.py` сохранены.

## Топ-5 неотложных действий

1. **Защитить lifecycle документа при rename**: запретить операцию для активной обработки с блокировкой записи либо сохранить неизменяемый storage key. Дополнить компенсацию upload ошибкой commit. См. DDD-01, DDD-02.
2. **Исправить recovery задач**: учитывать последний heartbeat, проверить живую длительную задачу и действительно умерший worker. См. PERF-01.
3. **Сделать ingestion достоверным и неблокирующим**: передавать ошибки parser в итог задания/retry и вынести parse/chunking из event loop. См. PERF-02, PERF-03.
4. **Восстановить безопасный retry идемпотентных запросов** после ошибки до внешнего эффекта; отдельно обрабатывать неопределённый результат. См. API-01.
5. **Исправить раннюю RAG-дедупликацию** с учётом источника, редакции и структурного scope; добавить тест двух одинаковых фрагментов под разными условиями. См. RAG-01.

## Метод и границы

Проверки выполнили все семь специализированных reviewer-агентов: DDD, DI, API, RAG, multitenancy/security, performance/reliability, test quality. Перед поиском использовался `graphify query`; `graphify-out/wiki/index.md` отсутствует. Находки сопоставлены по корневой причине; пробел теста компенсации включён в DDD-02, а не вынесен отдельной дублирующей проблемой. Confidence относится к конкретному описанному дефекту; сетевые и concurrency-сценарии явно отделены от изолированных воспроизведений.

Фронтенд, CI и эксплуатационные скрипты охвачены ограниченно. Не проводились исчерпывающий анализ каждого файла, проверка зависимостей по базам уязвимостей и полный прогон тестов. Нет утверждения о полной безопасности приложения или о количественном покрытии.

### Находки по областям

| Область | Находки |
|---|---|
| DDD / Application / mapping | DDD-01 (HIGH), DDD-02 (MEDIUM), DDD-03 (LOW) |
| DI / CLI | DI-01 (MEDIUM) |
| API | API-01, API-02, API-03 (MEDIUM) |
| RAG | RAG-01, RAG-02 (MEDIUM), RAG-03 (LOW) |
| Security / Authentication | SEC-01, SEC-02 (MEDIUM) |
| Performance / Reliability | PERF-01 (HIGH), PERF-02, PERF-03 (MEDIUM) |
| Tests | TEST-01, TEST-03 (MEDIUM), TEST-02 (LOW) |

## CRITICAL

Подтверждённых находок нет.

## HIGH

### DDD-01 — Переименование PENDING-документа удаляет файл, ожидаемый worker

- **Category:** DDD / Application / lifecycle.
- **Severity:** HIGH. **Confidence:** 0.98.
- **Location:** `server/app/application/documents/rename.py:100`; `:135`; `:77`.
- **Evidence:** проверяется статус другого документа, занимающего новое имя, но не переименовываемого. После commit старый `source_path` удаляется. Worker берёт сохранённый `storage_key` (`server/app/infrastructure/worker/tasks.py:45`) и скачивает его в `server/app/application/documents/processor.py:111`. API вызывает эту команду из `server/app/presentation/api/routes/documents.py:207`.
- **Problem:** загрузить документ и переименовать его, пока job ожидает исполнения. Новый ключ создаётся, старый удаляется, job сохраняет старую ссылку.
- **Impact:** допустимая операция владельца ломает обработку успешной загрузки. Возможна гонка с началом обработки.
- **Recommendation:** проверять lifecycle под блокировкой документа; минимально запретить rename во время активной обработки. Рассмотреть неизменяемый ключ хранения отдельно от отображаемого имени.
- **Verification:** существующий characterization harness с PENDING-документом разрешил rename и вызвал `delete_file('old-key')`, сохранив статус pending. Реальная очередь/S3 не использовались.

### PERF-01 — Recovery объявляет живые долгие задачи осиротевшими

- **Category:** Reliability / worker lifecycle.
- **Severity:** HIGH. **Confidence:** 0.99.
- **Location:** `server/app/infrastructure/repositories/misc/sqlalchemy_background_job_repository.py:143`.
- **Evidence:** `recover_orphaned()` сравнивает `started_at`, игнорируя `heartbeat_at`. Worker обновляет heartbeat каждую минуту (`server/app/infrastructure/worker/runtime.py:35`); cron выполняет recovery и `benchmark_sweeps.fail_dead_jobs()` каждые 15 минут (`server/app/infrastructure/worker/cron.py:30`). Порог по умолчанию — 30 минут (`server/app/config.py:295`), допустимые таймауты ingestion/benchmark/sweep — 2/1/6 часов (`server/app/presentation/cli/commands/worker.py:35`).
- **Problem:** исправное задание старше порога получает FAILED независимо от свежего heartbeat.
- **Impact:** ложный провал ingestion/benchmark. Sweep также становится FAILED, хотя продолжает выполняться; ограничение активного sweep освобождается и допускает параллельный запуск нового.
- **Recommendation:** определять устаревание по `COALESCE(heartbeat_at, started_at)`; установить начальный heartbeat при запуске. Проверить живую долгую задачу и потерянный worker.
- **Verification:** перехвачен SQL реального repository: присутствует `started_at <`, отсутствует `heartbeat_at`. Реальное конкурентное PostgreSQL-воспроизведение не запускалось.

## MEDIUM

### SEC-01 — Заблокированный ввод отправляется вспомогательной LLM до проверки

- **Category:** Security / input guardrails.
- **Severity:** MEDIUM. **Confidence:** 1.00.
- **Location:** `server/app/application/rag/service.py:99`; `server/app/infrastructure/ml/rag/rag_steps.py:188`; `server/app/infrastructure/ml/rag/rag_prompts.py:93`.
- **Evidence:** prepare сначала выполняет condense, затем safety.scan. При включённой condensation и непустой history исходные question/history уже отправлены fast_llm к моменту запуска InputScanner.
- **Problem:** guardrail не защищает первый вспомогательный LLM-вызов.
- **Impact:** отклоняемый ввод достигает модели и расходует бюджет; недоверенные инструкции обрабатываются стадией переписывания вопроса. Утечка чужих документов не доказана; retrieval и финальная generation после отказа не запускаются. Condensation по умолчанию выключена.
- **Recommendation:** проверять исходные question/history до первого LLM-вызова; отдельно проверять результат condensation при необходимости.
- **Verification:** настоящий prepare и condense/scan с подменой только auxiliary вызова, payload `ignore all previous instructions`: auxiliary_condense_calls=1, последующий terminal rejection=True. Внешний LLM не вызывался.

### SEC-02 — API-key кэш и authentication используют разные форматы identity

- **Category:** Authentication / API / reliability.
- **Severity:** MEDIUM. **Confidence:** 1.00.
- **Location:** `server/app/presentation/api/auth_dependencies.py:61`; `:71`; `server/app/infrastructure/auth/api_key_provider.py:93`.
- **Evidence:** ApiKeyService возвращает ApiKeyAuthResult dataclass; dependency передаёт его в provider, который использует json.dumps и `value['api_key_id']`. Ошибка подавляется как Redis failure. На чтении provider возвращает dict, тогда как dependency ожидает result.id/result.role. Composition действительно подключает этот provider (`server/app/composition/infrastructure.py:306`).
- **Problem:** обычный успешный запрос не заполняет положительный кэш; существующая запись формата provider вызывает AttributeError при аутентификации.
- **Impact:** каждый API-key запрос обращается к БД и обновляет last_used; при положительной legacy записи возникает HTTP 500. Обход авторизации не подтверждён.
- **Recommendation:** согласовать порт/DTO и JSON mapping, восстанавливать ApiKeyAuthResult на чтении; проверить реальный provider совместно с authentication dependency.
- **Verification:** два вызова dependency с реальным provider дали positive_cache_entries=0 и db_lookups=2. После записи asdict(result) тем же provider следующий вызов дал `AttributeError: 'dict' object has no attribute 'id'`. Существующие тесты проверяют стороны раздельно (`server/tests/test_identity_service_split.py:131`, `server/tests/test_api_key_cache.py:56`) и пропускают несовместимость.

### DDD-02 — Upload не компенсирует объект хранения при ошибке commit

- **Category:** Transactions / external effects.
- **Severity:** MEDIUM. **Confidence:** 1.00.
- **Location:** `server/app/application/documents/upload.py:155`; `:182`; компенсация `:118`.
- **Evidence:** `persist_upload()` охватывает upload и `set_source_path`, но commit выполняется позднее, при выходе из внешнего UoW (`server/app/application/uow/base.py:45`). Исключение этого выхода не попадает в компенсацию.
- **Problem:** upload успешен, commit БД падает; документ откатывается, файл остаётся.
- **Impact:** orphan-объекты, расход места и сохранение загруженных данных после неуспешной операции.
- **Recommendation:** отслеживать созданный ключ за пределами UoW и компенсировать при подтверждённом rollback; учитывать неопределённый исход commit.
- **Verification:** harness с `commit.side_effect = RuntimeError('commit failed')`: один upload, ноль deletes, rollback. Тест `server/tests/test_document_commands_characterization.py:317` ломает операции до commit; проверка commit failure в этом файле имеется только для delete (`:438`), поэтому дефект пропущен.

### DI-01 — Reconcile использует клиент без актуальной генерации корпуса

- **Category:** DI / CLI / reliability.
- **Severity:** MEDIUM. **Confidence:** 0.99.
- **Location:** `server/app/presentation/cli/commands/reconcile.py:66`.
- **Evidence:** команда создаёт Qdrant-клиент без `bootstrap_configuration()`. Factory использует fallback `generation=""` (`server/app/infrastructure/ml/clients/factories.py:264`); durable corpus имеет UUID (`server/app/infrastructure/corpus/control.py:91`). Write guard сравнивает namespace с durable snapshot (`server/app/infrastructure/corpus/locks.py:43`).
- **Problem:** fresh CLI не загружает authoritative snapshot. Даже при совпадающем endpoint/alias операция `--delete` отклоняется из-за generation; после corpus switch предварительный scan может смотреть старый target.
- **Impact:** штатная очистка orphan points не работает на инициализированном корпусе; диагностика может относиться к устаревшему корпусу.
- **Recommendation:** использовать общий bootstrap/lifecycle и текущий corpus client контейнера, сохраняя generation guard и disposal.
- **Verification:** одинаковые snapshot с UUID и пустой generation воспроизвели `ValidationError: Corpus generation changed; retry using the current snapshot` в реальном `validate_write_admission()`.

### API-01 — Ошибка запроса оставляет idempotency key в IN_PROGRESS

- **Category:** API / idempotency / reliability.
- **Severity:** MEDIUM. **Confidence:** 1.00.
- **Location:** `server/app/presentation/api/routes/ingest.py:78`; `:95`; `server/app/presentation/api/idempotency.py:38`; `server/app/infrastructure/rate_limit/idempotency.py:50`.
- **Evidence:** ключ резервируется до `create_job`; терминальная ошибка или release не записывается. IN_PROGRESS хранится с TTL 3600. Аналогичный порядок используется upload/benchmark.
- **Problem:** ошибка БД до любого эффекта оставляет завершённый запрос якобы выполняющимся.
- **Impact:** retry с тем же ключом получает 409 до часа и не может восстановить операцию после кратковременного отказа.
- **Recommendation:** ввести терминальную ошибку/безопасный retry для гарантированного отсутствия эффекта. Для неопределённого исхода сохранять durable operation/job ID; не освобождать ключ вслепую.
- **Verification:** настоящий IdempotencyStore с fakeredis и `ingest_documents()`: первый `create_job` выбросил ConnectionError до эффекта, повтор после восстановления получил 409 и не вызвал `create_job`; TTL=3600.

### API-02 — Экспорт benchmark теряет историю, дату среза и заметки

- **Category:** API / data portability.
- **Severity:** MEDIUM. **Confidence:** 1.00.
- **Location:** `server/app/presentation/api/routes/benchmark_admin.py:180`; `server/app/application/benchmark/services.py:125`.
- **Evidence:** export исключает `history`, `as_of_date`, `notes`, поддерживаемые create/import схемой. `bulk_create()` также не передаёт notes в entity.
- **Problem:** export/import не сохраняет контекст многотуровых и временных сценариев.
- **Impact:** восстановленный датасет исполняется иначе, результаты benchmark меняются без изменения текста вопроса.
- **Recommendation:** экспортировать полный переносимый набор через общую схему, сохранить notes в bulk_create; проверить round-trip.
- **Verification:** экспорт вопроса с history, датой 2024-01-01 и notes с последующей `BenchmarkQuestionCreate.model_validate()` дал None для всех трёх полей.

### API-03 — Benchmark history игнорирует диапазон и возвращает старые записи

- **Category:** API / filtering.
- **Severity:** MEDIUM. **Confidence:** 1.00.
- **Location:** `server/app/presentation/api/routes/benchmark_admin.py:437`; `:447`.
- **Evidence:** `days` и `metric` не используются; запрос выполняется с `sort_order="asc"` и limit, без границы даты.
- **Problem:** `history?days=1` включает давние результаты; при превышении limit последние запуски исключаются.
- **Impact:** график тренда скрывает текущие улучшения и регрессии.
- **Recommendation:** передавать cutoff в repository, выбирать последние limit и затем упорядочивать для графика; определить семантику metric.
- **Verification:** прямой вызов с days=1 вернул запись 2000-01-01; нижний слой получил только dataset, asc sort и limit.

### RAG-01 — Текстовая дедупликация удаляет разные условия и источники

- **Category:** RAG / retrieval correctness.
- **Severity:** MEDIUM. **Confidence:** 1.00.
- **Location:** `server/app/domain/rag/ranking.py:109`; `server/app/infrastructure/ml/rag/rag_reranking.py:20`.
- **Evidence:** `HybridRetriever.merge_and_dedup()` (`server/app/application/search/hybrid.py:76`) и объединение decomposition (`server/app/infrastructure/ml/rag/retrieval_pipeline.py:204`) дедуплицируют текст до scope filtering, игнорируя document_id, parent_units, редакцию и источник. Более точная identity позднего context_selection не восстанавливает удалённые кандидаты.
- **Problem:** одинаковое предложение о размере выплаты под условиями «подразделение А» и «подразделение Б» становится одним доказательством; нужный запросу вариант может исчезнуть.
- **Impact:** неполный ответ, необоснованный отказ или потеря подтверждающего источника.
- **Recommendation:** согласовать раннюю дедупликацию с identity контекста, учитывая документ, редакцию и структурный scope.
- **Verification:** оба реальных helper с одинаковым текстом и разными document_id/parent_units оставили только document_id=1; восстановленные parent scope различались.

### RAG-02 — Condensation получает историю без бюджета контекстного окна

- **Category:** RAG / context budgeting.
- **Severity:** MEDIUM. **Confidence:** 0.99.
- **Location:** `server/app/infrastructure/ml/rag/rag_prompts.py:93`.
- **Evidence:** ChatService ограничивает историю числом сообщений (`server/app/application/chat/service.py:97`), а schema допускает вопрос до 16000 символов. При включённом condense все сообщения передаются auxiliary LLM без token-budget. Generation budgeting выполняется позднее (`server/app/infrastructure/ml/rag/rag_steps.py:427`).
- **Problem:** защищённая от переполнения generation следует за незащищённой condensation.
- **Impact:** несколько длинных допустимых сообщений могут вызвать ошибку провайдера либо усечение, меняющее поисковый запрос. Condensation по умолчанию выключена.
- **Recommendation:** ограничивать полный auxiliary prompt с резервом ответа; использовать summary и ограниченную историю либо явную деградацию к исходному вопросу.
- **Verification:** RunnableLambda вместо LLM получил все восемь сообщений; собственный estimator проекта оценил полный prompt в 72832 токена. Сетевых/платных вызовов не было.

### PERF-02 — Parse и chunking ingestion блокируют event loop worker

- **Category:** Performance / async.
- **Severity:** MEDIUM. **Confidence:** 1.00.
- **Location:** `server/app/application/ingestion/loader.py:199`; `server/app/application/ingestion/batch.py:65`; `server/app/application/ingestion/single_file.py:89`.
- **Evidence:** async load_file напрямую вызывает sync parse_file → parser → format handler; batch/single-file синхронно выполняют split/index. Composition передаёт обычные sync adapters (`server/app/composition/service_providers.py:127`).
- **Problem:** PDF/OCR/тяжёлый chunking исполняются в общем loop ARQ.
- **Impact:** задерживаются другие задачи, heartbeat, cron, отмена и таймауты в том же worker.
- **Recommendation:** вынести тяжёлую работу в ограниченный executor, учитывая завершение при отмене; использовать существующую политику upload processing.
- **Verification:** настоящий load_file с sync parser длительностью 150 ms задержал параллельный asyncio ticker с шагом 10 ms до 151 ms.

### PERF-03 — Ошибка parsing завершает ingestion-job статусом DONE

- **Category:** Reliability / ingestion outcomes.
- **Severity:** MEDIUM. **Confidence:** 1.00.
- **Location:** `server/app/application/ingestion/single_file.py:79`; `server/app/application/ingestion/loader.py:68`.
- **Evidence:** loader превращает исключение parser в None; single-file пишет ошибку и нормально возвращается. `run_tracked_job()` вызывает mark_done (`server/app/infrastructure/worker/runtime.py:104`). Batch append также не передаёт число parsing failures наружу (`loader.py:185`, `batch.py:51`). Reset-путь, отменяющий удаление при parse failure, не относится к этой находке.
- **Problem:** неиндексированный повреждённый документ или временный отказ parser выглядят как успешное выполнение.
- **Impact:** пользователь видит DONE, retry не запускается; восстановление требует ручной диагностики.
- **Recommendation:** различать skip и failure явным результатом/исключением; отражать partial batch failure и передавать временные ошибки retry-политике.
- **Verification:** настоящие loader → workflow → tracked job с parser ValueError: mark_done=1, mark_failed=0, sync_documents_to_db=0.

### TEST-01 — Параметризованный prompt injection тест не использует payload

- **Category:** Tests / RAG guardrails.
- **Severity:** MEDIUM. **Confidence:** 1.00.
- **Location:** `server/tests/test_prompt_injection.py:115`.
- **Evidence:** все 15 payload вызывают один build_system_prompt и проверяют два неизменных маркера; параметр не передаётся в код.
- **Problem:** тест «injection ignored» не помещает атакующий текст в реальный контекст и не проверяет sanitizer/роли сообщений.
- **Impact:** регрессия обработки извлечённого текста пройдёт тесты при сохранении маркеров. Успешная production injection этим не доказана.
- **Recommendation:** прогонять payload через сборку контекста и проверять роли, границы и sanitizer; поведение реальной LLM проверять отдельной adversarial evaluation.
- **Verification:** полный тестовый набор файла проходит, но параметр не используется ни в одном проверяемом действии.

### TEST-03 — Cancellation regression тест падает до проверяемой отмены

- **Category:** Tests / stale fixture / cancellation.
- **Severity:** MEDIUM. **Confidence:** 1.00.
- **Location:** `server/tests/test_solid_regressions.py:138`.
- **Evidence:** тест передаёт пустой SimpleNamespace вместо RagContext; generator обращается к `ctx.as_of_date` при создании evidence (`server/app/infrastructure/benchmark/evaluation/answer_generators.py:69`).
- **Problem:** AttributeError возникает до fake invoke, который должен выбросить CancelledError, и до проверки восстановления settings scope.
- **Impact:** целевой набор красный; проверка cancellation/context restoration фактически не выполняется.
- **Recommendation:** передавать валидный RagContext с UserContext; сохранить проверку отмены и восстановления исходных overrides.
- **Verification:** падение получено в наборе performance reviewer и повторено root отдельным запуском этого теста: 1 failed с указанным AttributeError.

## LOW

### DDD-03 — Mapper теряет Document.version

- **Category:** Repository mapping / aggregate state.
- **Severity:** LOW. **Confidence:** 1.00.
- **Location:** `server/app/infrastructure/repositories/document/sqlalchemy_document_repository.py:460`.
- **Evidence:** mapper не передаёт ORM version; domain default — 0 (`server/app/domain/knowledge/document.py:48`). `update_status(expected_version=...)` сравнивает счётчик с БД (`sqlalchemy_document_repository.py:109`).
- **Problem:** сохранённая version=7 восстанавливается как 0; предусмотренное контрактом обновление с doc.version получит ложный конфликт.
- **Impact:** неполное восстановление агрегата и ненадёжный optimistic update contract. В проверенном production processing expected_version сейчас не передаётся; пользовательский сбой на этом пути не заявляется. Поэтому при консолидации severity снижена с MEDIUM до LOW.
- **Recommendation:** восстанавливать version из ORM; проверить round-trip и актуальную/устаревшую optimistic version.
- **Verification:** прямой mapper probe: persisted_version=7, entity_version=0.

### RAG-03 — ttft_sec измеряет всю generation

- **Category:** RAG / observability.
- **Severity:** LOW. **Confidence:** 1.00.
- **Location:** `server/app/infrastructure/ml/rag/rag_steps.py:581`.
- **Evidence:** timestamp берётся до semaphore, затем целиком вычитывается astream, выполняются sanitization/PII redaction, и только потом duration записывается в ttft_sec. Grounded response без LLM также заполняет поле.
- **Problem:** метрика time-to-first-token не фиксирует первый непустой chunk.
- **Impact:** benchmark и диагностика смешивают очередь, первый токен, декодирование и postprocessing.
- **Recommendation:** отдельно фиксировать первый provider chunk и определить начало отсчёта; отделить grounded/cache ответы. Сохранить буферизацию, необходимую для полного output scanning.
- **Verification:** прослежен путь в PipelineMetaEvent/RagResult; отдельный latency benchmark не запускался.

### TEST-02 — FakeChunkRepository игнорирует content_hashes

- **Category:** Tests / repository fakes.
- **Severity:** LOW. **Confidence:** 1.00.
- **Location:** `server/tests/fakes.py:240`.
- **Evidence:** метод принимает content_hashes, но отбирает только document_id и считает total до hash-фильтра, которого нет. Application передаёт фильтр (`server/app/application/chunks/queries.py:71`).
- **Problem:** общий fake расходится с контрактом выборки и пагинации по hash.
- **Impact:** unit-тесты не могут надёжно проверять этот сценарий и могут закреплять отличающееся от БД поведение.
- **Recommendation:** применять hash-фильтр до total/pagination; общий contract case для fake и SQL adapter.
- **Verification:** сохранены hashes a,b; list_for_document с content_hashes=[b] вернул оба, total=2.

## INFO — Проверенные свойства

- В проверенном графе отсутствуют обратные импорты Domain → Application/Infrastructure и Application → Infrastructure. ActVersionTimeline содержит чистые предметные правила; UoW имеет типизированные порты.
- Основной container передаёт зависимости через конструкторы/Depends, не использует универсальный resolve внутри бизнес-кода. UoW factory создаёт отдельные AsyncSession; ML families защищены lease/ContextVar. API/worker выполняют teardown и при частичной ошибке startup.
- Dense/sparse Qdrant получают ACL и temporal filter; hash resolver требует access_filter. Перед rerank, после enrichment/neighbors и при cache hit выполняется authoritative проверка документов. Sources формируются из выбранного контекста.
- JWT identity обновляется через БД; UserContextFactory загружает текущие группы и назначения CURATOR. Проверенные document/chunk query и conversation history контролируют доступ к объекту. ADMIN listing/search различаются, CLIENT search ограничен owner_id; cache key включает kind, role, user_id, группы и curator scope.
- Есть отрицательные ACL-тесты, проверки stale index/cache, реальные PostgreSQL outbox/concurrency тесты с отдельной disposable schema. Наличие интеграционных тестов проверено; их фактический запуск в этом аудите не выполнялся.

## Выполненные проверки

Все команды запускались из `server/` с `.venv/bin/pytest -q -o addopts=''` и перечисленными файлами. Часть файлов пересекается между запусками, поэтому результаты нельзя складывать в число уникальных тестов.

| Набор | Результат |
|---|---|
| test_acl.py, test_bm25_acl.py, test_answer_cache.py, test_curator_role.py | 143 passed |
| test_retrieval_acl.py, test_ingestion_acl_consistency.py | 38 passed |
| test_container.py, test_review_bug_regressions.py, test_storage_lifecycle.py, test_corpus_resources.py | 47 passed |
| test_api_asgi.py, test_chat_contracts.py, test_benchmark_question_import.py, test_idempotency_store.py | 77 passed |
| test_prompt_injection.py, test_document_commands_characterization.py, test_review_contracts.py, test_uow.py, test_rag_authoritative_access.py | 154 passed |
| test_rag_pipeline.py, test_context_selection.py, test_rag_service_stream.py, test_prompt_injection.py | 123 passed |
| test_worker_resource_lifecycle.py, test_solid_regressions.py, test_partial_ingestion_reset.py | 29 passed, 1 failed (TEST-03) |
| test_solid_regressions.py::test_rag_generator_preserves_sweep_scope_and_restores_it_on_cancellation | 1 failed, отдельное подтверждение TEST-03 |
| test_api_key_cache.py, test_identity_service_split.py, test_rag_application_stages.py | 29 passed |

Изолированные воспроизведения указаны у находок: моки/fakeredis и реальные локальные функции без внешних сервисов. Наблюдались DeprecationWarning SWIG, они не являются основанием находок.
