# Async S3 — этапы A и B

Текущий итог (2026-10-13): задачи 1–8 выполнены; исправлена потеря отмены при ошибке потока, финальные unit/MinIO/Redis проверки пройдены, server и worker последовательно выпущены на A → B. Boto3 удалён. Исторические результаты этапов ниже сохранены с исходными датами.

## Этап A — история реализации

Реализованы задачи 1–5 плана `docs/superpowers/plans/2026-10-12-async-s3-migration.md`.
Документы/ingestion используют aiobotocore; BM25 snapshot пока читает отдельный
container/registry-owned boto3 reader только в потоке. На момент завершения A
этап B ещё не был реализован; reader удалён последующими изменениями B.

## Контракты

- `FileStorage.list_files`, `get_file_info`, `IngestionRegistry.collect_entries` — async.
- `create_storage()` создаёт лёгкий backend без глобального cache; `LazyStorage`
  удерживает current/retired backends до shutdown. Смена S3-config инвалидирует reader
  и generation BM25. In-flight запросы сохраняют исходные настройки.
- Один async client/pool на backend/loop. `S3_MAX_CONCURRENT=10` ограничивает pool
  и операции; очередь ограничена `SEMAPHORE_ACQUIRE_TIMEOUT`. Shutdown запрещает
  новые запросы, ожидает активные операции и закрывает каждый backend даже при
  ошибке другого close. Listener закрывается раньше registry, registry — раньше storage.
- GET читает по 1 MiB; файловые операции выполняются в потоке. При отмене ожидается
  уже начатая запись/close, частичный или не возвращённый файл удаляется.
- HEAD возвращает `None` только для missing key; permission/bucket/network ошибки
  распространяются. COPY failure не удаляет source. Keys, ISO metadata/hash и
  BM25 JSON/ACL сохраняются. Порядок sync commit → sparse persistence → registry
  остаётся прежним. BM25 remove при delete остаётся до commit, как в baseline.
- BM25LoadError не становится cached missing. RAG/benchmark сохраняют dense-only
  fallback с исходным ACL. Следующий вызов повторяет загрузку; отмена распространяется.
- Standalone benchmark владеет registry и его sync-reader; async document storage
  ему на этапе A не требуется. Владение async snapshot storage добавляется на этапе B.

## Проверки (2026-10-12)

Baseline commit: `c3dbf15`; server/worker image: `rag-server:development-cpu-0.13.0`;
client: `rag-client:0.13.0`; MinIO: `RELEASE.2024-09-22T00-33-43Z`.
Работающие server/worker не обновлялись; MinIO bucket не очищался.

- Characterization baseline: **132 passed**.
- Regression ingestion/transactions/CLI/worker/storage/RAG: **276 passed**.
- Финальный полный unit-набор после исправлений ревью: **2593 passed, 81 deselected**
  за 68.98 s. Дополнительные lifecycle/review проверки: **62 passed**.
- Ruff: проходит. Обязательные layer checks: оба пустые.
- `task lint`: ruff проходит, mypy — **29 ошибок в 13 файлах**. На baseline-копии
  `c3dbf15` тот же набор diagnostic messages, новых ошибок нет.
- Format check: исходный `tests/test_chunk_search_acl.py` требует форматирования;
  файл не относится к миграции. Изменённые файлы отформатированы.
- Финальный MinIO smoke после review fixes: **2 passed** за 3.46 s, без warnings. PUT/HEAD/LIST/GET/COPY/RENAME/DELETE,
  Unicode/пробелы, пустой объект, SHA-256, pagination 1001 объектов и пять ротаций.
  Cleanup удаляет только заранее записанные keys под UUID test prefix.
- Delayed fake LIST/HEAD/body-read удерживаются 1 s; heartbeat 50 ms продолжает
  работать. Max lag: HEAD **1.754 ms**, LIST **1.155 ms**, GET **1.760 ms** (<200 ms).
- Peak RSS smoke: **95112 → 121640 KiB** (включая TLS/client init и 1001 coroutines).
  FD: **16 → 26** с открытым pool; пять retired pools: **16 → 21 → 16** после close.
  Unclosed client/session warnings отсутствуют. Это smoke, не production SLA/loadtest.
- Полный suite показал отдельный pending `pyrate_limiter.Leaker` teardown warning;
  он не относится к S3, не скрывался и не исправлялся в этой миграции.

Lock: aiobotocore **3.9.2**, botocore **1.43.106**, aiohttp **3.14.1**;
новые транзитивные aioitertools **0.13.0**, wrapt **2.5.0**.
boto3 **1.43.51** / s3transfer **0.19.1** сохранены для этапа A.

## Повторить проверки

Из `server/`:

```bash
uv run --group testing pytest tests/test_file_storage_async.py tests/test_s3_storage_async.py tests/test_s3_storage_download.py tests/test_storage_lifecycle.py tests/test_bm25_async_loading.py tests/test_bm25_lifecycle.py -q
uv run --group testing pytest -q -m 'not integration and not slow'
```

Opt-in smoke требует отдельного разрешения оператора. Credentials и S3_BUCKET
передать окружением через штатный secret provisioning, не печатать их в shell/logs:

```bash
RAG_S3_SMOKE=1 S3_SMOKE_ENDPOINT=http://127.0.0.1:9000 uv run --group testing pytest tests/integration/test_s3_storage_minio.py -m integration -q
```

По умолчанию integration fixture пропускается и не обращается к сети.
В этой сессии host loopback/Docker IP из command environment недоступны. Smoke
выполнен в одноразовом контейнере с `--network container:rag-minio`, read-only
checkout/venv и уже закреплёнными зависимостями. Working services не изменялись.

Client lifecycle/streaming реализованы согласно [aiobotocore S3 example](https://aiobotocore.aio-libs.org/en/stable/examples/s3/basic_usage.html).

## Выпуск и откат

Deploy не выполнялся. При отдельном запросе выпуска собирать один новый server
image и использовать его для server и worker. Предыдущий image для обоих:
`rag-server:development-cpu-0.13.0`. До смены дождаться in-flight ingestion/BM25
jobs и завершить процессы с cleanup. MinIO/Qdrant/models одновременно не менять.

Откат: вернуть предыдущий server image одновременно server/worker. Не нужны
migration downgrade, очистка bucket или переиндексация. Failed sparse persistence
не фиксирует success registry row, штатный ingestion retry остаётся доступен.
Парсинг/OCR остаются отдельным источником блокирующей CPU/I/O работы; async S3
не устраняет весь event-loop lag. Retired pools живут до shutdown.

## Решения исполнения и финальное ревью

Независимый read-only reviewer проверил `c3dbf15..1962607`; Critical/Minor не найдено.
Два Important исправлены в одном проходе с RED→GREEN и полным зелёным suite:
- Реальный aiobotocore ClientCreatorContext нельзя закрывать, если его вход не
  завершился. Исходная ошибка/CancelledError сохраняются, повторная инициализация
  доступна. Проверено настоящим SDK context без сети.
- Валидный JSON с неверными texts/ACL/hashes/numeric fields отклоняется явной
  validation как BM25LoadError. Программные ошибки не поглощаются broad catch.
Дополнительно закрыт обход shutdown через публичный ensure_client().

Решения, принятые при исполнении (и цена, если они не подходят):
1. Отдельная ветка в текущем checkout вместо worktree: сохранены локальные ignored
   plan/config и готовая среда; изменения видны в основном рабочем каталоге.
2. Characterization expected GREEN без production changes: они фиксируют текущее
   поведение; RED→GREEN применён к новому поведению. Ограничение — полнота выбранных
   сценариев baseline, а не доказательство, что текущая бизнес-логика идеальна.
3. Узкие BM25LoadError consumer fallback границы из задачи 7 перенесены в A:
   сохраняют старую dense деградацию; цена — чуть больший scope consumer diff в A.
4. Standalone владеет sync-reader через свою registry, без неиспользуемого async
   storage; async storage composition для standalone остаётся работой этапа B.
5. Async BM25 loading/mutations и удаление boto3 исключены из этого этапа по запросу
   пользователя; для полного перехода требуется этап B.
6. Существующие broad catches mutation adapter/отмена LocalStorage download не
   менялись; отменённый local download по-прежнему может оставить temp-файл.
7. Production SLA/throughput и deployment не проверялись: контролируемый smoke не
   доказывает производственную производительность; работающие сервисы не менялись.
8. Изменения после reviewed commit проверены автором с RED→GREEN и полным suite,
   без повторного reviewer, согласно executing-plans; независимого второго ревью
   именно финальных исправлений нет.

Deferred minors: отсутствуют. Ни push, ни merge не выполнялись.

## Исправления по повторной проверке этапа A (2026-10-12)

- `MLClientRegistry.close()` теперь ожидает принадлежащую registry cleanup-задачу
  даже при повторной отмене caller; reader, ML HTTP pools и Qdrant закрываются до
  распространения `CancelledError`. Повторное закрытие не запускает cleanup снова.
  Regression сначала воспроизвёл пропуск ML/Qdrant cleanup, затем прошёл после исправления.
- Добавлен успешный COPY → DELETE rename при `S3_MAX_CONCURRENT=1` с тайм-аутом
  против deadlock. Ротация проверяет отдельные endpoint/access key/secret key
  прежнего и нового backend; config events проверяются для bucket, endpoint,
  credentials и region.
- В затронутых BM25/storage тестах импорты перенесены на уровень модулей.
- Целевые storage/BM25/worker lifecycle проверки: **71 passed**. Ruff для `app tests`
  и format check пяти изменённых Python-файлов проходят; `git diff --check` пустой.
- Mypy: **29 ошибок в 13 файлах**, изменённый production-файл среди diagnostics
  отсутствует. Полный sandbox run завис на ASGI; отдельные ASGI tests вне sandbox:
  **36 passed**. MinIO smoke и deploy при этом исправлении не выполнялись.
- Полный unit-suite вне sandbox (`.venv/bin/pytest -q -m 'not integration and not slow'`):
  **2599 passed, 81 deselected, 13 warnings** за 71.48 s.

## Этап B — async BM25 (2026-10-12)

Задачи 6–7 реализованы; независимое финальное ревью завершено. Два Important исправлены с RED→GREEN; Critical/Minor не найдено. Документированное
состояние этапа A выше сохранено как история. Работающие процессы и image tags
не менялись, deploy и повторный MinIO smoke для B не выполнялись.

- Registry получает async loader через constructor; composition привязывает его
  к своему LazyStorage. Один load на generation; устаревший результат при
  invalidation отбрасывается. Missing snapshot и failed load различаются.
- `bm25_index()` и `bm25_update()` работают только с памятью. Unloaded state даёт
  `BM25SnapshotNotReady`; ошибка определена рядом с BM25LoadError в persistence
  и доступна через импорт client_registry, чтобы избежать package import cycle.
- Порт BM25 mutations асинхронный. Adapter сначала готовит snapshot, затем
  выполняет CPU mutation в принадлежащем ему потоке. Повторяется только гонка
  invalidation; failed network load не запускает бесконечный retry и не создаёт
  неподтверждённый пустой index. Позиции mutations относительно commit сохранены.
- JSON/read/from_dict и unlink offloaded; cancelled parse дожидается CPU job до
  удаления temp-файла. Registry shutdown дожидается load/mutation до close pools;
  composition затем закрывает snapshot storage. Отмена не кешируется как missing.
- Standalone generator сохранил существующую ответственность за свои ресурсы:
  теперь владеет LazyStorage и registry с явным async loader; close registry перед
  storage. Это инфраструктурный владелец, без нового импорта composition из
  infrastructure. Sweep без registry владеет storage в пределах cache_candidates.
  Metrics без registry требует явного async loader, глобальных S3 factories нет.
- CPU benchmark retrieval получает `bm25_index` keyword-only; None означает
  отсутствие sparse после async подготовки/degradation. ACL передаётся прежним путём.
- Sync reader и sync S3 loader удалены. `task lock` удалил boto3 1.43.51 и
  s3transfer 0.19.1; остальные версии не изменились. Aiohttp 3.14.1 объявлен прямой
  зависимостью для обработки ошибок streaming body, уже был в lock.

Фактические проверки:

- Повторный baseline A: **191 passed** за 6.28 s.
- Новый core RED: 10 failed / 3 passed; после реализации GREEN: **13 passed**.
- Новый consumer RED: 7 failed; итоговый regression BM25/transactions/lifecycle:
  **177 passed**. Обязательные ACL/contracts/RAG: **201 passed**.
- Первое полное выполнение выявило 4 устаревших sync mocks (review contracts и
  standalone characterization); они адаптированы, проверки commit/ACL сохранены.
- Полный unit-набор до исправлений финального ревью: **2616 passed, 81 deselected, 13 warnings** за 71.89 s. После исправлений: **2619 passed, 81 deselected, 13 warnings** за 75.46 s.
- Ruff проходит. Layer checks и поиск старых sync S3 API пустые. Cold imports
  metrics/registry/composition/standalone успешны. `uv tree --locked`: boto3 и
  s3transfer отсутствуют.
- Mypy: **29 ошибок в 13 файлах**; точное сравнение диагностик с архивом HEAD
  (без номеров строк) не обнаружило новых или исчезнувших ошибок.
- Прежний pending `pyrate_limiter.Leaker` teardown warning появился после полного
  suite; к S3 он не относится. Sandbox execution зависает на async/thread tests,
  те же unit-команды вне sandbox проходят. Один повторный запуск не стартовал
  из-за тайм-аута auto-review разрешения; разрешённый повтор успешно выполнен.

Повторить unit-проверки из server:

```bash
.venv/bin/pytest tests/test_bm25_async_loading.py tests/test_bm25_consumers_async.py tests/test_bm25_lifecycle.py tests/test_bm25_incremental.py tests/test_chunk_service.py tests/test_document_commands_characterization.py -q
.venv/bin/pytest -q -m 'not integration and not slow'
```

Перед выпуском B остаются повторный MinIO smoke для final code, реальные cold/warm
BM25 и Redis invalidation, image build/tag и согласованный restart server/worker.
Предыдущий image и процедура отката указаны выше; schema/bucket migration нет.
Изменения оставлены в working tree без commit/push/merge, поскольку затронутые
migration-файлы уже содержали пользовательские изменения при начале проверки.


### Итог независимого ревью B

Два Important исправлены в одном проходе:

1. `FileNotFoundError` после успешного GET (например, исчезнувшая temp-директория)
   больше не означает missing snapshot. Теперь это BM25LoadError с сохранением
   причины; состояние остаётся unloaded и следующий запрос повторяет загрузку.
2. `scripts/compound_retrieval_experiment.py::compare()` получил собственные
   LazyStorage и async loader: существующий snapshot не превращается незаметно
   в dense-only поиск. Registry закрывается перед storage при успехе и ошибке.

Regression на эти сценарии: сначала **3 failed / 19 passed**, затем **22 passed**.
Последний полный suite: **2619 passed / 81 deselected**, новых mypy diagnostics
по сравнению с HEAD нет. Дополнительный повторный reviewer не запускался:
исправления проверены RED→GREEN и полным набором согласно executing-plans.

Решения исполнения B и их ограничения:

- Работа продолжена в существующем feature checkout, задачи 6–7 выполнены одним
  согласованным diff. Коммитов нет из-за исходных пользовательских правок в тех
  же файлах; изменения видны в текущем checkout и требуют будущего commit.
- Task helper не распознал русские заголовки «Задача»: exact briefs извлечены
  вручную, progress/logs сохранены локально. Ограничение — ручной учёт.
- BM25SnapshotNotReady определён в persistence и доступен через client_registry;
  это устраняет import cycle, но canonical location отличается от текста плана.
- Standalone сохранил прежнего инфраструктурного владельца вместо нового импорта
  composition; server/worker по-прежнему связываются в composition. Ограничение —
  standalone wiring расположен отдельно.
- Aiohttp из существующего lock объявлен прямо для streaming exceptions;
  последующее обновление должно учитывать совместимость с aiobotocore.
- Сохранены прежнее ожидание state lock при synchronous invalidation, non-S3
  Qdrant/embedding cleanup sweep fallback и модель замены in-memory mutations
  persisted snapshot при Redis invalidation. Это известное поведение HEAD;
  миграция не обещает устранить соответствующие latency/freshness ограничения.
- Read-only reviewer не запускал тесты/сервисы; root выполнил unit/lint. Реальные
  MinIO, производственная отзывчивость и deploy фазы B остаются непроверенными.

Deferred minors: отсутствуют. Полный format check по-прежнему указывает только
исходный `tests/test_chunk_search_acl.py`; изменённые файлы отформатированы.


## Финальная проверка и выпуск (2026-10-13)

- `run_storage_io` сохраняет исходный `CancelledError` даже когда ожидаемый thread job завершается `ValueError`/`OSError`. Regression loader/mutation: RED **4 failed**, GREEN входит в **75 passed** storage/BM25 проверок. Проверяется удаление temp-файла и отсутствие опубликованного snapshot.
- Полный unit: **2628 passed, 81 deselected, 13 warnings**, 72.58 s (`.venv/bin/pytest -q -m 'not integration and not slow'`, вне sandbox). В него входят controlled CLI upload/ingest/list, standalone benchmark, worker lifecycle и ACL/transactions. Реальный LLM benchmark и ingestion пользовательских документов не запускались.
- Ruff и layer checks чистые; изменённые файлы отформатированы. Полный format check по-прежнему указывает только исходный `test_chunk_search_acl.py`. Mypy: **29 ошибок в 13 файлах**, точный набор диагностик после нормализации номеров строк совпадает с baseline.
- Финальный MinIO/Redis smoke: **4 passed**, 6.84 s. PUT/HEAD/LIST/GET/COPY/RENAME/DELETE, content/hash, pagination, pools cleanup; 10 cold BM25 callers, warm reuse, refresh; реальная смена endpoint IP/DNS между запросами; pub/sub на уникальном канале. Существующий `_system/bm25_index.json` не перезаписывался; cleanup только UUID test-prefix.
- Smoke выполнялся в одноразовом контейнере с read-only checkout/venv и `--network container:rag-server`: host environment не достигал MinIO; namespace MinIO не имеет маршрута к Redis backend-сети. Credentials загружались из `.env`, затем `.env.secrets` без вывода. Первоначальная попытка без secrets вернула InvalidAccessKeyId и не создала объектов.
- В отдельном Python-процессе финального worker проверен unavailable-S3 endpoint: lazy storage constructor — **0 network calls**, первая HEAD завершается явным BotoCoreError, heartbeat продолжает работать (**566 ticks**), клиент закрывается. Настройки основного worker не изменялись. Это проверка owned storage lifecycle в рабочем образе, а не отключение S3 для всего стека.
- Responsiveness HEAD/LIST/GET подтверждена regression-тестами: heartbeat продолжает работать при задержанном network fake, максимальный lag <200 ms. Это controlled проверка, не оценка production SLA.
- Отдельный финальный read-only reviewer новых изменений: подтверждённых ошибок нет.

### Образы и последовательный выпуск

Предыдущий образ сохранён: `rag-server:development-cpu-0.13.0`.

| Этап | Tag | Image SHA-256 |
|---|---|---|
| A (коммит `9ca9bce`) | `rag-server:development-cpu-0.13.0-async-s3-a-20261013` | `29180b20f2465a726a5e9a0e86fcc1a9ee6c29cd533ea8e705e9f0a0d1cbab6e` |
| B (финальное working tree) | `rag-server:development-cpu-0.13.0-async-s3-b-20261013` | `6d89946740bfb010aa6a393454e2f4eb7c8e7382acf5a6e424f6e91248a08d20` |

Для A source извлечён `git archive 9ca9bce` в `/tmp/rag-async-s3-stage-a`; dev bind mount app временно заменён на этот snapshot. Перед выпуском очередь Arq пуста. Единственный остаточный in-progress key без job payload — короткоживущий completed-cron marker (проверен контракт Arq `finish_job`, пользовательские аргументы не выводились). Server/worker остановлены с штатными grace periods; затем A запущен через `up -d --no-deps --no-build --pull never server worker`. API ready; в отдельных процессах обоих контейнеров успешны S3 round-trip собственного test-object и cold/warm загрузка существующего BM25 (**4037 документов**).

Затем оба процесса корректно завершены и выпущены на B той же командой без временного source override. У обоих единый tag B; API healthy, restart count **0**; worker running. Реальный round-trip и cold/warm BM25 успешны в обоих контейнерах. `aiobotocore=3.9.2`, `botocore=1.43.106`, `aiohttp=3.14.1`; boto3/s3transfer отсутствуют. В startup logs нет ERROR/Traceback/Unclosed session сообщений. MinIO/Qdrant/Redis/Postgres/Ollama/client не пересоздавались, модели и schema не менялись.

`SERVER_IMAGE` в локальном `server/.env` закреплён на B, чтобы последующие Task-команды не вернули прежний образ. VERSION не изменялся. Commit/push/merge не выполнялись; B содержит уже существовавшие пользовательские изменения текущего checkout.

### Повторение smoke

Загрузить credentials из штатных `.env`/`.env.secrets` в окружение без печати, затем из server:

```bash
RAG_S3_SMOKE=1 RAG_S3_REDIS_SMOKE=1 S3_SMOKE_ENDPOINT=http://minio:9000 S3_SMOKE_ALTERNATE_ENDPOINT=http://192.168.16.3:9000 .venv/bin/pytest tests/integration/test_s3_storage_minio.py -m integration -q -o 'addopts=--strict-markers --tb=short'
```

Адреса — снимок текущей Docker-сети; при пересоздании определить их заново. Для Redis smoke нужен доступ к обеим сетям. Redis канал уникален для теста и не инвалидирует кэши работающих процессов. Штатный integration запуск без opt-in пропускается.

### Откат dev-выпуска

Dev bind mounts означают, что одного старого image tag недостаточно: нужно одновременно вернуть source snapshot. Подготовленный snapshot — `/tmp/rag-async-s3-rollback` (коммит `c3dbf15`), override — `/tmp/rag-async-s3-rollback.override.yml`. Если `/tmp` очищен, восстановить snapshot через `git archive c3dbf15` и переопределить app mount для обоих сервисов. Перед откатом остановить приём запросов/дождаться jobs и завершить процессы штатно.

В этом выпуске использован временный launcher `/tmp/rag_async_s3_compose.py`: он читает dotenv в порядке Taskfile, задаёт VERSION и выбранный SERVER_IMAGE, затем вызывает compose с base + dev override. Примеры команд отката в текущей сессии:

```bash
server/.venv/bin/python /tmp/rag_async_s3_compose.py rag-server:development-cpu-0.13.0 stop server worker
server/.venv/bin/python /tmp/rag_async_s3_compose.py rag-server:development-cpu-0.13.0 -f /tmp/rag-async-s3-rollback.override.yml up -d --no-deps --no-build --pull never server worker
```

После отката также закрепить предыдущий SERVER_IMAGE в server/.env. Не очищать bucket, не делать downgrade БД и не переиндексировать. Сохранён и промежуточный образ A для более узкого отката B; ему нужен source snapshot `9ca9bce`.

Ограничения: это controlled smoke/lifecycle проверки, не production load/SLA; запуск реального LLM benchmark или ingestion пользовательских документов не требовался. Mypy/format baseline замечания остаются отдельной работой.


Для восстановления временного compose launcher после очистки /tmp (из корня):

```bash
cat > /tmp/rag_async_s3_compose.py <<'PYCODE'
import os
import subprocess
import sys
from dotenv import dotenv_values
for name in ('server/.env', 'server/.env.secrets', 'client/.env'):
    for key, value in dotenv_values(name).items():
        if value is not None:
            os.environ[key] = value
os.environ['VERSION'] = open('VERSION').read().strip()
os.environ['SERVER_IMAGE'] = sys.argv[1]
raise SystemExit(subprocess.call(['docker', 'compose', '-f', 'docker-compose.yml', '-f', 'docker-compose.override.yml', *sys.argv[2:]]))
PYCODE
mkdir -p /tmp/rag-async-s3-rollback
git archive c3dbf15 | tar -x -C /tmp/rag-async-s3-rollback
cat > /tmp/rag-async-s3-rollback.override.yml <<'YAML'
services:
  server:
    volumes:
      - type: bind
        source: /tmp/rag-async-s3-rollback/server/app
        target: /code/project/app
        read_only: true
  worker:
    volumes:
      - type: bind
        source: /tmp/rag-async-s3-rollback/server/app
        target: /code/project/app
        read_only: true
YAML
```

## Исправление baseline lint/type замечаний (2026-10-13)

По отдельному запросу устранены все 29 mypy ошибок в 13 файлах: уточнены локальные типы коллекций, типы checkpoint payload в местах чтения, TypedDict resume options и nullable guards; сохранены opaque checkpoint port и существующий формат данных. DeleteDocumentResponse получает канонический enum, admin get-job использует проверенный lookup ID. Форматирование `test_chunk_search_acl.py` исправлено.

Проверки: `task lint` — ruff чистый, mypy **0 ошибок в 521 source files**; `ruff format --check app tests` — **737 файлов**, замечаний нет; layer checks и `git diff --check` чистые. Целевые benchmark/checkpoint/RAG/ACL тесты: **148 passed**. Полный unit: **2628 passed, 83 deselected, 13 warnings**, 74.76 s. Integration/slow проверки не повторялись: эта правка не меняет storage/network dependencies. Образы после этой правки не пересобирались; commit/push не выполнялись.
