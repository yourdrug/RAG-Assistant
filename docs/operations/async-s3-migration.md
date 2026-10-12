# Async S3 — этап A

Реализованы задачи 1–5 плана `docs/superpowers/plans/2026-10-12-async-s3-migration.md`.
Документы/ingestion используют aiobotocore; BM25 snapshot пока читает отдельный
container/registry-owned boto3 reader только в потоке. Этап B не реализован.

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
- Полный unit-набор: **2580 passed, 81 deselected**; дополнительные lifecycle
  тесты после этого: **22 passed**. Последующий итоговый прогон записывается ниже.
- Ruff: проходит. Обязательные layer checks: оба пустые.
- `task lint`: ruff проходит, mypy — **29 ошибок в 13 файлах**. На baseline-копии
  `c3dbf15` тот же набор diagnostic messages, новых ошибок нет.
- Format check: исходный `tests/test_chunk_search_acl.py` требует форматирования;
  файл не относится к миграции. Изменённые файлы отформатированы.
- MinIO smoke: **2 passed**, без warnings. PUT/HEAD/LIST/GET/COPY/RENAME/DELETE,
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
uv run --group testing pytest tests/test_file_storage_async.py tests/test_s3_storage_async.py tests/test_s3_storage_download.py tests/test_storage_lifecycle.py tests/test_s3_snapshot_reader.py tests/test_bm25_lifecycle.py -q
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
