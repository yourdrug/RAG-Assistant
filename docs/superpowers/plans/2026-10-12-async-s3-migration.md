# План миграции S3-хранилища на aiobotocore

> **Для исполнителя:** выполнять задачи последовательно с помощью `superpowers:executing-plans`; при выбранном пользователем делегировании — `superpowers:subagent-driven-development`. Чекбоксы отражают выполнение, а не только написание кода. При подготовке документа реализация и тесты не выполнялись. По последующему запросу выполнен этап A; работающие server/worker не обновлялись.

**Goal:** перевести сетевые операции S3 в server, Arq worker и CLI на aiobotocore, устранить PERF-001 и затем полностью удалить boto3, сохранив ACL, транзакционные границы и согласованность BM25.

**Architecture:** application зависит от асинхронного `FileStorage`, infrastructure владеет aiobotocore-клиентом, composition связывает зависимости и управляет их временем жизни. Миграция разделена на две поставляемые части: сначала документы/ingestion с временным синхронным reader для BM25; затем асинхронная загрузка BM25 и удаление boto3. CPU-операции и файловая система продолжают выполняться вне event loop.

**Tech Stack:** Python 3.11, asyncio, aiobotocore 3.9.2, botocore 1.43.106, aiohttp 3.14.1, FastAPI, Arq, MinIO, pytest/pytest-asyncio, uv, ruff/mypy.

**Spec:** пользовательский запрос о миграции async-клиента; исходный PERF-001; [проверка совместимости](../../research/aiobotocore-compatibility.md); [правила проекта](../../../AGENTS.md). Проектные решения и критерии поведения зафиксированы ниже, поскольку отдельной согласованной спецификации миграции пока нет.

**Дата:** 2026-10-12. **Статус:** этап A (задачи 1–5) реализован и проверен; этап B не начат. Фактические результаты: [runbook](../../operations/async-s3-migration.md).

Документы под `docs/*` сейчас исключены правилом `.gitignore:21`. Файл существует локально; при включении плана/отчёта в PR добавить именно эти документы через `git add -f`, не менять правила игнорирования всей папки.

## 1. Границы и ограничения

- Python остаётся `==3.11.*`. На первом этапе добавить `aiobotocore==3.9.2` и прямую зависимость `botocore>=1.43.101,<1.43.107`: `Config` импортируется непосредственно проектом. boto3 остаётся только у переходного reader; после его удаления зависимость убрать.
- После каждого изменения `server/pyproject.toml` выполнить `task lock`, проверить diff `server/uv.lock`; не обновлять весь стек через `--upgrade`.
- Не добавлять aioboto3, aiofiles, новый DI-framework или отдельный event loop в потоке. Для локальных файлов достаточно asyncio и стандартной библиотеки.
- Domain не знает об S3, boto3, aiobotocore, async HTTP или конфигурации. Application не импортирует infrastructure. Привязка конкретного storage/loader выполняется в composition.
- Новые функции и обычные методы именовать без ведущего `_`; imports — на уровне модуля; production-код без `assert`; роли/видимости/статусы брать из существующих enum.
- До изменения публичного поведения файла из god-file backlog добавить characterization-тесты затрагиваемого публичного метода и получить зелёный baseline. Существующие тесты можно расширять; создавать параллельный дублирующий набор не нужно.
- Сохранить object keys, `s3://...` source, формат `FileItem`, hash `size_bytes_last_modified`, BM25 JSON и `BM25_S3_KEY`. Смена клиента сама по себе не требует миграции БД или переиндексации.
- Сохранить порядок batch ingestion: document/chunk sync commit → sparse persistence → registry upsert. Ошибка sparse persistence не должна помечать файл успешно индексированным.
- Не переносить storage-компенсации или BM25-обновления через существующие transaction boundaries. В частности, chunk mutations остаются после успешного commit; текущий порядок удаления документа сначала фиксируется characterization-тестом.
- Сохранить ACL-метаданные BM25, pre-filter и вторую проверку Qdrant resolve. Invariant-тесты не удалять.
- План не включает перевод Qdrant на async, замену парсеров/OCR, новую схему фоновых задач, переработку общего API ошибок или устранение всех блокирующих операций ingestion.
- Запуск сервисов, интеграционных и нагрузочных проверок во время реализации требует прямого запроса пользователя согласно AGENTS.md. Команды ниже — инструкция для будущего выполнения, не выполненные проверки.

## 2. Что уже проверено

В копии `server/pyproject.toml` и `server/uv.lock` в `/tmp` проверено разрешение зависимостей без установки пакетов:

| Пакет | Исходный lock | Lock с aiobotocore |
|---|---|---|
| aiobotocore | отсутствует | 3.9.2 |
| botocore | 1.43.51 | 1.43.106 |
| aiohttp | 3.14.1 | 3.14.1 |
| aioitertools | отсутствует | 0.13.0 |
| wrapt | отсутствует | 2.5.0 |
| boto3 | 1.43.51 | 1.43.51, временно сохраняется |

Вариант без boto3 тоже разрешился; вместе с ним удалился s3transfer 0.19.1. Версии остальных существующих пакетов сохранились. Это подтверждает совместимость dependency graph, но не работу клиента с нашим MinIO.

MinIO в compose: `quay.io/minio/minio:RELEASE.2024-09-22T00-33-43Z`. Реальные запросы к нему с aiobotocore, cancellation и shutdown пока не проверены.

## 3. Основные проектные решения

### 3.1. Разделить миграцию на два этапа

**Этап A:** все операции с документами и ingestion становятся async; синхронный boto3 остаётся исключительно у BM25-reader, вызываемого из рабочего потока. Это первая самостоятельная поставка, закрывающая PERF-001.

**Этап B:** BM25 загружает S3 snapshot асинхронно; синхронные CPU-обновления работают только с памятью. После перевода всех потребителей удаляются переходный reader, sync storage factory и boto3.

Промежуточное наличие двух клиентов допустимо только с явными владельцами, закрытием соединений и разделением вызовов. Нельзя оставить boto3 скрытым fallback внутри async adapter.

### 3.2. Асинхронный storage-порт

Сохранить существующие аргументы и типы; добавить `async` двум metadata-методам и общий async teardown:

```python
class FileStorage(Protocol):
    async def list_files(self, prefix: str) -> list[FileItem]: ...
    async def get_file_info(self, key: str) -> FileItem | None: ...
    async def download_to_temp(self, key: str) -> Path: ...
    async def upload_file(self, key: str, data: bytes) -> None: ...
    async def delete_file(self, key: str) -> None: ...
    async def rename_file(self, old_key: str, new_key: str) -> None: ...
    async def copy_file(self, old_key: str, new_key: str) -> None: ...
    async def aclose(self) -> None: ...

    @property
    def supported_extensions(self) -> tuple[str, ...]: ...
```

`download_bytes()` не добавлять в общий порт: скачивание через temp-файл уже подходит BM25 и не расширяет весь интерфейс ради одного потребителя.

На первой итерации `IngestionRegistry.collect_entries(docs, chunks, source_chars)` становится async и ожидает HEAD. HEAD остаются последовательными: устраняется блокировка event loop, но не обещается сокращение времени batch. Использование metadata из LIST — отдельная оптимизация после определения правил object-version consistency; не смешивать её с заменой клиента.

### 3.3. Владение клиентом и смена настроек

- Один `LazyStorage` принадлежит одному InfrastructureContainer, а его async-клиент — одному event loop. Server и worker имеют собственные экземпляры; CLI создаёт и закрывает свой контейнер внутри одного `asyncio.run()`.
- Сохранить лёгкое синхронное создание `S3Storage`: constructor только фиксирует snapshot параметров и состояние; сеть/HTTP session не открывает. Первое async-действие вызывает `ensure_client()` под asyncio.Lock; клиент входит в async context ровно один раз.
- Вынести создание backend в `create_storage() -> FileStorage` без process-global `lru_cache`. `LazyStorage` может синхронно создать лёгкий backend; его методы и `aclose()` явно соответствуют порту. Не сохранять зависимости от async-клиента через `__getattr__` в другом потоке.
- Для `LazyStorage.invalidate()` сохранить модель retirement: новые действия получают новый backend, уже начатые завершаются на старом. Retired backends закрываются при shutdown, как в текущей модели. Конкретный backend выбирается один раз на составную `rename_file()` операцию.
- Backend сохраняет endpoint/bucket/credentials/region на момент создания. Смена глобальных settings не меняет параметры уже начатой операции.
- `aclose()` идемпотентен, запрещает новые операции, ожидает завершение или отмену собственных активных операций, закрывает текущий и retired клиенты на их loop. Закрытие одного клиента не должно мешать попытке закрыть остальные; первая ошибка возвращается после cleanup.
- При init failure не сохранять полуоткрытый client/context: следующая операция может повторить инициализацию. Shutdown не должен впервые создавать клиент.
- Прямые `get_storage()` в CLI, cron, metrics и BM25 должны исчезнуть к завершению этапа B. Composition инжектирует зависимости; lifecycle не определяется случайным первым вызовом глобальной фабрики.

### 3.4. Ограничение конкурентности и сетевые параметры

Сохранить существующие `connect_timeout=10`, `read_timeout=60`, `retries={"max_attempts": 3, "mode": "standard"}`. Не добавлять внешний retry вокруг каждого boto-operation: это умножает число попыток.

Ввести положительное инфраструктурное setting `S3_MAX_CONCURRENT=10`, отражённое в `server/.env.example` и используемое также для `max_pool_connections`. Semaphore ограничивает запросы в одном процессе; это не глобальный лимит между server и worker. Для GET слот удерживается до закрытия response body; для LIST — до завершения листинга; для COPY/DELETE — до завершения запроса. Acquisition отменяемый. Bound для очереди ожидания использовать существующий `SEMAPHORE_ACQUIRE_TIMEOUT`, без добавления ещё одного похожего timeout setting.

Не держать semaphore повторно вокруг `rename_file()`, если внутренние copy/delete уже его захватывают: при лимите 1 получится deadlock. Не создавать отдельный клиент на каждый HEAD/GET.

Для custom endpoint проверить path-style addressing; сохранить текущие credentials, region и подпись. Не отключать TLS verification или checksum validation. Если закреплённый MinIO несовместим с новым checksum-поведением, зафиксировать воспроизводимый запрос и принять отдельное решение о конфигурации/обновлении MinIO.

### 3.5. Скачать файл без блокировки loop и без неограниченного буфера

`download_fileobj()` в aiobotocore отсутствует. Использовать `await get_object()` и async response body с чтением блоками; именованная инфраструктурная константа `S3_DOWNLOAD_CHUNK_BYTES = 1024 * 1024`. В каждый момент удерживается один блок, ожидается его запись, затем читается следующий. Создание temp-файла, запись, close и unlink выполняются через `asyncio.to_thread()`.

Правила cleanup:

1. Успех возвращает существующий закрытый temp-файл; caller сохраняет ответственность за последующее удаление.
2. Ошибка GET/read/write удаляет частичный temp-файл и закрывает body.
3. `CancelledError` не превращается в `None`: cleanup завершается, затем отмена распространяется.
4. Отмена `to_thread(write)` не останавливает поток. Прежде чем закрывать файл или удалять его, дождаться завершения уже начатой записи; тест должен воспроизвести эту гонку.
5. Не возобновлять оборванный GET автоматически с середины или поверх частичного файла. Поведение managed transfer меняется; bounded stream retry потребует отдельного решения с проверкой версии объекта.

### 3.6. Явная политика ошибок

Для HEAD `None` означает подтверждённое отсутствие объекта, а не AccessDenied или недоступный S3. Разрешённые missing-коды для HEAD вынести в именованную константу, проверить `404` / `NoSuchKey` / `NotFound`. `NoSuchBucket` и `403` не маскировать под отсутствующий файл. Для COPY сохранить текущую обработку `NoSuchKey`, остальные ошибки распространять. Ошибка copy не запускает delete в rename.

Это осознанное уточнение текущего поведения `get_file_info()`, которое сейчас скрывает все ClientError. Проверить реакции single-file ingestion и worker на ошибку backend. Не менять HTTP error envelope этой миграцией.

Для BM25 подтверждённое отсутствие snapshot оставляет существующую возможность работать без sparse index. Ошибки сети/авторизации и повреждённый snapshot оборачиваются в инфраструктурный `BM25LoadError` с сохранением причины; loader и registry распространяют его, `_bm25_loaded` остаётся false. В существующих best-effort границах RAG/benchmark и mutation adapter ловить именно `BM25LoadError`, логировать причину и сохранять текущую деградацию: sparse results пусты, неуспешная mutation не выдаётся за успешную. Следующая операция снова пробует загрузку. Не повторять failed mutation бесконечно внутри того же запроса. Отмена всегда распространяется; ошибка ACL или программная ошибка не попадает в этот fallback. Это сохраняет текущую возможность работы без sparse при недоступном snapshot, но устраняет кэширование transient failure как успешного missing.

## 4. Карта файлов

| Файлы | Ответственность |
|---|---|
| `server/app/application/ports/file_storage.py` | Async metadata и teardown контракт |
| `server/app/infrastructure/storage/file_storage.py` | LocalStorage, async S3Storage, LazyStorage, backend factory |
| `server/app/infrastructure/storage/__init__.py` | Актуальные exports без устаревшей глобальной фабрики |
| `server/app/config.py`, `server/.env.example` | S3 concurrency setting |
| `server/app/application/services/document_loader.py` | Await LIST |
| `server/app/application/services/ingestion_registry.py` | Async collection с await HEAD |
| `server/app/application/services/batch_ingestion.py` | Await collection; сохранить порядок commit/sparse/registry |
| `server/app/application/services/single_file_ingestion.py` | Await HEAD |
| `server/app/composition/infrastructure.py`, `service_providers.py`, `application.py`, `container.py` | Storage ownership, инъекция BM25-loader, retirement/teardown |
| `server/app/infrastructure/ml/config/config_subscribers.py` | Ротация owned storage и переходного reader |
| `server/app/presentation/cli/commands/ingest.py`, `worker.py` | Async lifecycle CLI и worker |
| `server/app/infrastructure/worker/cron.py` | Использование container-owned storage |
| Новый `server/app/infrastructure/bm25/s3_snapshot_reader.py` | Переходный sync reader на этапе A; удалить на этапе B |
| `server/app/infrastructure/bm25/persistence.py`, `ml/clients/factories.py`, `ml/clients/client_registry.py` | Async BM25 loading; удалить синхронную S3-загрузку |
| `server/app/application/ports/bm25_index.py`, `ml_clients.py` | Async mutation adapter; уточнить cached-only accessor |
| `server/app/infrastructure/bm25/bm25_updater.py` | Async orchestration загрузки, CPU mutation в потоке |
| `server/app/application/services/chunk_mutation_service.py`, `document_delete_command.py` | Await BM25 port вместо to_thread(async_method) |
| `server/app/infrastructure/metrics/metrics.py` | Async BM25 loading через injected registry |
| `server/app/infrastructure/benchmark/answer_generators.py`, `retrieval.py`, `sweep_scoring.py` | Подготовка BM25 snapshot до sync scoring |
| `server/app/infrastructure/ml/rag/rag_steps.py`, `rag/rag_retrieval.py` | Явный BM25LoadError fallback при сохранении текущей sparse-деградации |
| `server/tests/fakes.py`, существующие ingestion/BM25/CLI/worker tests | Новые async-контракты без ослабления assertions |
| `server/pyproject.toml`, `server/uv.lock` | Прямые зависимости и удаление boto3 |

Новые test-файлы ниже имеют точные предлагаемые имена; перед созданием проверить, не появился ли соответствующий набор в параллельных изменениях. Не переименовывать связанные production-модули только ради этого плана.

## 5. Review Focus

Пять ситуаций требуют отдельных тестов:

1. Отмена скачивания во время thread-write: закрытый body, отсутствие temp-файла, отсутствие записи в уже закрытый дескриптор — задача 3.
2. Смена S3-настроек во время запроса: старый запрос заканчивается на прежнем backend, новый использует новый, оба закрываются — задача 4.
3. Invalidation BM25 между загрузкой и CPU mutation: обновление не теряется и не применяется к retired snapshot — задача 6.
4. Отмена первой BM25-загрузки при нескольких ожидающих: последующий caller повторяет загрузку, lock не остаётся захваченным — задача 6.
5. S3 возвращает 403/timeout вместо отсутствующего ключа: ingestion не сообщает «файл отсутствует», retry не пропускает файл как успешно обработанный — задачи 2, 5 и 6.

## 6. Последовательность задач

### Задача 1. Зафиксировать baseline и инварианты

**Файлы:** существующие `server/tests/test_ingestion_workflows.py`, `test_ingestion_service_pipeline.py`, `test_bm25_lifecycle.py`, `test_chunk_service.py`, `test_review_contracts.py`, `test_document_commands_characterization.py`; при необходимости расширить characterization для публичных chunk/document методов.

**Consumes:** текущие sync FileStorage metadata и sync BM25 mutations. **Produces:** зелёный baseline с проверкой результата и порядка побочных эффектов; никаких новых production-зависимостей.

- [x] Зафиксировать git status, текущие image tags/commit и lock; не включать чужие изменения в коммиты.
- [x] Дополнить characterization для успешной/ошибочной batch и single-file ingestion: registry не обновляется после failed sync/sparse, hashes и ACL сохраняются.
- [x] Зафиксировать порядок BM25/remove/storage cleanup в delete и commit-before-mutation в chunk add/edit/delete; не подгонять ожидания под будущую реализацию.
- [x] Проверить существующие single-load, generation invalidation, конкурентную sync mutation + async load, mutation после отсутствующего snapshot.
- [x] Выполнить выбранный baseline после прямого запроса на тесты; сохранить команду и результат. Если baseline красный, отделить исходную ошибку от миграции, не продолжать рефакторинг затронутого god-file.

Команда baseline из `server/`:

```bash
uv run pytest tests/test_ingestion_workflows.py tests/test_ingestion_service_pipeline.py tests/test_bm25_lifecycle.py tests/test_chunk_service.py tests/test_review_contracts.py tests/test_document_commands_characterization.py tests/test_reingestion_consistency.py -q
```

**Готовность:** public behavior закреплён; каждая исходная ошибка объяснена. Коммит: `test: characterize storage and BM25 migration contracts`.

### Задача 2. Подготовить async-контракты без изменения BM25-loader

**Файлы:** FileStorage port, LocalStorage/S3Storage, loader/registry/batch/single-file, ingestion fixtures; новый `server/tests/test_file_storage_async.py`.

**Consumes:** baseline задачи 1. **Produces:** async `list_files`, `get_file_info`, `collect_entries`; LocalStorage.aclose; metadata временно выполняется через to_thread в старом S3 adapter. Это короткий совместимый мост до native async adapter.

- [x] Написать `test_list_and_head_yield_to_other_coroutines`: задержанный sync fake должен позволять второй корутине сигнализировать event до завершения metadata-вызова. Для детерминизма использовать Events, не только sleep/timing.
- [x] Написать HEAD error mapping: missing → None; AccessDenied/NoSuchBucket/timeout → ошибка; cancelled call → отмена.
- [x] Изменить port и все четыре ingestion callers вместе; `collect_entries` возвращает прежний dict, hash и chunks_count.
- [x] LocalStorage LIST/stat offload целиком; S3 metadata временно offload целиком, включая pagination loop. Убрать существующие вспомогательные имена с ведущим `_` только в изменяемом storage-компоненте.
- [x] Заменить metadata/collection MagicMock на AsyncMock в fixtures и всех найденных вызовах, включая `test_reingestion_consistency.py`. Использовать `assert_awaited*` там, где важно ожидание.
- [x] Проверить новый storage suite и ingestion baseline; статус «native async migration завершена» на этом шаге не ставить.

**Готовность:** PERF-001 устранён уже на уровне metadata; port готов к замене backend. Коммит: `refactor: make storage metadata operations awaitable`.

### Задача 3. Реализовать native async S3Storage и streaming downloads

**Файлы:** `file_storage.py`, config/env example, pyproject/lock; новые `server/tests/test_s3_storage_async.py`, `test_s3_storage_download.py`.

**Consumes:** async FileStorage. **Produces:** `S3Storage.ensure_client()` и `aclose()`; низкоуровневые async S3 операции; ограниченные соединения и streaming temp download. LIST/HEAD больше не выполняют boto3 через to_thread.

- [x] Добавить тесты lazy initialization: 10 одновременных первых операций открывают один client; init failure закрывает частичный context и допускает повторный вызов.
- [x] Добавить параметры операций для LIST/HEAD/PUT/COPY/DELETE: Bucket/Key/Prefix, extended keys с пробелами и Unicode, supported extensions, сортировка и pagination >1000 объектов.
- [x] Добавить download-тесты на несколько блоков, пустой объект, ошибку чтения, ошибку записи, cancelled read и cancelled thread-write. Проверять фактический контент/temp cleanup/response close.
- [x] Добавить тест `S3_MAX_CONCURRENT=1`: второй запрос ждёт, slot освобождается при ошибке/отмене; streaming GET удерживает slot до body close; rename не зависает.
- [x] Добавить aiobotocore и прямой botocore dependency, выполнить `task lock`, просмотреть ограниченный diff lock. Boto3 пока сохранить.
- [x] Реализовать async client acquisition через AsyncExitStack или сохранённый context manager; module-level imports. Сохранить сетевые timeouts/retries и snapshot config.
- [x] Реализовать low-level await calls и async paginator; streaming writes/cleanup — по разделу 3.5. Перенести boto3 reader из общего adapter в переходный модуль задачи 4 в том же поставляемом PR, чтобы текущий BM25 не отключился.
- [x] Проверить no-retry-after-copy-failure, missing/error mapping и повторное `aclose()`. После закрытия новый вызов даёт явную lifecycle-ошибку, не открывает новый session.

**Готовность:** все document storage network операции используют aiobotocore; локальный I/O offloaded; память download ограничена блоком. Коммит: `feat: add native async S3 storage` вместе с переходным BM25 boundary задачи 4.

### Задача 4. Связать lifecycle и изолировать переходный BM25-reader

**Файлы:** composition, config subscribers, CLI, cron, storage exports; новый `s3_snapshot_reader.py`; BM25 factories/registry; новые `server/tests/test_storage_lifecycle.py`, `test_s3_snapshot_reader.py`; существующие CLI/worker/review lifecycle tests.

**Consumes:** native S3Storage; прежняя sync BM25 loading semantics. **Produces:** container-owned async storage и временный reader с sync `download_bytes(key: str) -> bytes`, `close() -> None`. Reader принадлежит registry/container, а не глобальному cached factory.

- [x] Добавить тесты retirement/re-init/shutdown: смена bucket/endpoint/credentials, незавершённый GET, shutdown до первого запроса, двойной dispose, ошибка одного close не пропускает другие.
- [x] Ввести `create_storage()` без lru_cache; убрать cache_clear на async backend. Config subscriber вызывает invalidate owned storage; после смены S3-config инвалидируется также BM25 generation.
- [x] Перевести cron на `ctx["container"].infrastructure.ml.file_storage`; CLI upload использовать `run_with_infrastructure()`, чтобы storage закрывался до завершения asyncio.run.
- [x] Для standalone benchmark предусмотреть отдельного владельца storage: composition factory создаёт его при standalone запуске и передаёт generator/registry; generator.close() закрывает только owned ресурсы, не shared container storage.
- [x] Переходный sync reader создаётся и вызывается вне loop, сохраняет собственный config snapshot; reader повторно используется, retire/close включён в ML registry teardown. Factory загрузки получает reader явно вместо get_storage().
- [x] Сохранить BM25 locks/generation и sync loading в потоке на этапе A; никаких coroutine/client async calls в reader.
- [x] Проверить CLI success/error, worker startup/shutdown и standalone close. Переходный reader имеет единственный смысл — загрузка BM25; document ingestion его не вызывает.
- [x] Уточнить teardown order: прекратить listener/admission, завершить принадлежащие registry/generator загрузки и CPU jobs, затем закрывать storage. Текущий порядок `storage.aclose()` до `ml_clients.close()` необходимо пересмотреть для injected async loader; late load не должен использовать закрытый клиент. Ошибка одного этапа не пропускает последующий cleanup.

**Решение исполнения:** standalone benchmark на этапе A владеет sync-reader через собственный MLClientRegistry. Неиспользуемый async FileStorage не создаётся; composition-владение async snapshot storage добавляется на этапе B. Узкие BM25LoadError fallback границы из задачи 7 введены уже здесь, чтобы сохранить dense-only деградацию при ошибке reader.

**Готовность этапа A:** задачи 2–4 интегрированы и протестированы совместно. На промежуточной ветке нет awaitable-as-sync вызовов, недоступного download_bytes или отключённого hybrid search. Коммит: `refactor: own async storage lifecycle and isolate BM25 sync reader`.

### Задача 5. Подтвердить этап A на MinIO и задержанном S3

**Файлы:** новый `server/tests/integration/test_s3_storage_minio.py`; описание запуска в `docs/operations/async-s3-migration.md`.

**Consumes:** этап A. **Produces:** подтверждённая совместимость с закреплённым MinIO и проверка отзывчивости loop.

- [x] Создать opt-in integration fixture с `@pytest.mark.integration`, отдельным тестовым prefix и credentials из окружения. Не читать/выводить секреты и не очищать рабочий bucket.
- [x] Сохранить полный marker/path invocation; default unit pipeline не должен случайно обратиться к MinIO. Проверка требует прямого запроса на запуск.
- [x] Выполнить PUT → HEAD → LIST → GET → COPY → DELETE; проверить SHA-256 скачанного контента, пустой объект, Unicode keys, missing key и pagination на 1001 маленьком объекте.
- [x] Cleanup удаляет только объекты, созданные fixture под её уникальным prefix, в том числе после failed test; delete bucket/production data запрещены.
- [x] Через управляемый delayed fake/локальный тестовый HTTP endpoint задержать LIST/HEAD/body-read на 1 секунду; параллельная корутина должна выполнить сигнал до снятия задержки. Дополнительно собрать loop-lag при heartbeat каждые 50 ms: целевой max lag <200 ms на ненагруженном стенде; это критерий проверки, не обещание production SLA.
- [x] Сравнить с baseline: количество S3 HEAD пока не уменьшается, throughput не обязан вырасти; ожидаемое улучшение — другие корутины продолжают работать во время сети.
- [x] Зафиксировать RSS/количество открытых соединений до/после повторных downloads и config rotations; отсутствие Unclosed client/session warnings обязательно.

**Готовность:** MinIO smoke, content equality, metadata formats, heartbeat responsiveness и cleanup подтверждены; при checksum несовместимости этап не выпускается без отдельного решения.

### Задача 6. Перевести BM25 loading и mutations на async orchestration

**Файлы:** BM25 persistence, client registry, factories, bm25_updater, ports, composition; `chunk_mutation_service.py`, `document_delete_command.py`; BM25 lifecycle/incremental, chunk/document/characterization tests и fakes.

**Consumes:** container-owned FileStorage этапа A. **Produces:** async `ensure_bm25_loaded()` без сетевого I/O в потоке, async BM25IndexPort mutations; sync state access/mutations только в памяти.

Контракты:

- `load_bm25_index_from_s3(storage: FileStorage) -> BM25Index | None` остаётся async; download await, read/JSON/from_dict offload вместе, unlink offload. Missing snapshot → None; остальные ошибки → `BM25LoadError`, определённый в persistence, и не кэшируются как missing.
- `MLClientRegistry` получает `bm25_loader: Callable[[], Awaitable[BM25Index | None]] | None` через constructor. Composition всегда предоставляет configured loader; отсутствие loader допустимо для тестов и означает явно не настроенную capability, а не поиск через глобальную фабрику.
- `ensure_bm25_loaded()` обеспечивает один успешный load на generation через asyncio.Lock; `_bm25_generation` и state synchronization сохраняются. Network await выполняется без threading.Lock.
- `bm25_index()` становится cached-only accessor: не загружает S3. При invalid/unloaded state поднимает инфраструктурный `BM25SnapshotNotReady`, определённый на уровне модуля client_registry. Загруженное отсутствие snapshot по-прежнему даёт None.
- `bm25_update()` остаётся синхронным context manager для CPU-потоков: атомарно проверяет loaded state и выдаёт snapshot под существующим state lock. Не вызывает loader. Для отсутствующего, но успешно проверенного snapshot создаёт пустой индекс как сейчас.
- `BM25IndexPort.add/remove/replace` получают `async def` с прежними аргументами. Adapter сначала await ensure, затем выполняет sync mutation в потоке. Если invalidation произошла между этими действиями, `BM25SnapshotNotReady` вызывает повтор ensure/mutation. Это retry state race, а не retry сетевых ошибок. `BM25LoadError` логируется в adapter и завершает текущую best-effort mutation без бесконечного retry; CancelledError распространяется.
- В `bm25_add/replace/remove` исключение `BM25SnapshotNotReady` распространяется до adapter, а не поглощается существующим broad logging handler. Поведение остальных mutation errors отдельно закрепить текущим characterization.

Алгоритм загрузки/публикации:

1. Прочитать loaded/index/generation согласованно; если ready, вернуть snapshot.
2. Взять async load lock; повторить проверку ready.
3. Запомнить generation и await injected loader.
4. Опубликовать snapshot только если generation прежняя. При изменении отбросить результат и повторить для новой generation.
5. Failed/cancelled load не меняет state на loaded. Ожидающий caller может повторить загрузку.

Захват threading state lock, который может удерживаться CPU mutation, не выполнять блокирующим ожиданием на event loop: чтение/публикация snapshot выполняются короткими sync helpers через to_thread. CPU mutation и parsing никогда не выполняются под async load lock на event loop. Invalidation не содержит await или сетевых запросов; сохранённая сериализация state проверяется тестами.

- [ ] Сначала расширить lifecycle tests на 10 одновременных async callers: один download, один JSON parse, один опубликованный snapshot.
- [ ] Добавить invalidation during download; invalidation между ensure и mutation; mutation после missing snapshot; cancellation первого loader при ожидающем втором; failed load затем успешный retry. Отдельный тест: при постоянном storage failure один вызов mutation не зацикливается, не добавляет текст в неподтверждённый пустой snapshot, следующий вызов повторяет load.
- [ ] Переадаптировать существующий тест «sync update + async load share one snapshot»: async adapter + CPU worker по-прежнему дают один snapshot и сохраняют добавленный текст/ACL. Проверки не удалять из-за изменения API.
- [ ] Добавить assert, что cached-only accessor и sync updater не вызывают storage ни при ready, ни при unloaded state.
- [ ] Реализовать async loader/state алгоритм; инжектировать в container registry; оставить переходный reader только у ещё не переведённых потребителей задачи 7. На этом переходном шаге их старый путь должен по-прежнему работать, а не обращаться к уже cached-only accessor без подготовки snapshot: при необходимости сразу включить соответствующий caller из задачи 7 в этот PR.
- [ ] Сделать mutation port/adapter async и заменить application `await to_thread(self._bm25_index.method, ...)` на `await self._bm25_index.method(...)`. Не перемещать вызовы относительно UoW/commit.
- [ ] Обновить фейки/AsyncMock и assertions в chunk/document тестах; подтвердить сохранение hash/visibility/owner/group и поведения rollback.
- [ ] Выполнить BM25 lifecycle/incremental, transaction characterization и обязательные ACL tests.

**Готовность:** cold BM25 load использует async storage; CPU offloaded; concurrent update не теряется; cancellation не отравляет loaded state. Коммит: `refactor: load BM25 snapshots asynchronously`.

### Задача 7. Перевести остальные потребители и удалить boto3

**Файлы:** metrics, benchmark answer_generators/retrieval/sweep_scoring, их exports/tests; composition/config subscribers, BM25 persistence/factories; переходный reader; pyproject/lock.

**Consumes:** async BM25 loader и cached-only state задачи 6. **Produces:** отсутствие boto3 и sync S3 I/O во всех путях.

- [ ] RAG steps/retrieval и benchmark ловят только `BM25LoadError` вокруг подготовки snapshot, логируют ошибку и возвращаются к прежнему dense-only fallback. Добавить regression на ошибку load, повторный успешный load и сохранение Qdrant ACL; остальные exceptions/CancelledError не поглощать.
- [ ] Metrics с registry вызывает `await ensure_bm25_loaded()`; fallback без registry получает явно переданный loader/storage с определённым владельцем. Не создавать глобальный registry/client ради каждой метрики.
- [ ] StandaloneBenchmarkGenerator.generate() await загружает BM25 до `run_owned_sync(retrieve_with_scores_hybrid, ...)`, затем передаёт snapshot явно.
- [ ] В `retrieve_with_scores_hybrid` и `search_sparse` добавить keyword-only `bm25_index: BM25Index | None`; CPU scorer использует переданный snapshot и не обращается к registry.bm25_index(). None означает подтверждённую допустимую работу без sparse, а не необходимость скрытого load.
- [ ] Sweep с registry await ensure; fallback без registry получает async loader через composition/owned standalone lifecycle. Удалить прямой sync factory load.
- [ ] Обновить benchmark exports/callers/fakes и существующие lifecycle tests; отмена standalone benchmark ожидает завершение owned CPU jobs до закрытия его storage.
- [ ] Удалить `s3_snapshot_reader.py`, sync S3 persistence, `factories.load_bm25_index()` и их global cache/dispose ветки. Отдельный sync cached-only accessor при необходимости остаётся — он работает только с памятью.
- [ ] Удалить boto3 из pyproject и выполнить `task lock`. Botocore остаётся прямой зависимостью из-за Config. Подтвердить отсутствие boto3/s3transfer в resolved tree либо объяснить реального нового транзитивного потребителя, если он появился с момента baseline.
- [ ] Найти все старые `get_storage`, `download_bytes`, `load_bm25_index_from_s3_sync`, imports boto3 и `to_thread` вокруг async методов; каждый остаток устранить или доказать, что он относится только к локальному CPU/I/O.

**Готовность этапа B:** весь S3 network I/O async, sync CPU consumers получают готовый snapshot, boto3 не импортируется/не требуется. Коммит: `refactor: remove synchronous S3 client dependencies`.

### Задача 8. Финальная проверка, выпуск и откат

**Файлы:** operation runbook; этот checklist; release/PR description. Application изменения сверх задач 1–7 не требуются.

- [ ] Прогнать выбранный regression набор, lint/mypy и layer checks из раздела 7; записать фактические результаты, не только список команд.
- [ ] Повторить MinIO smoke и responsiveness для финального кода; проверить cold/warm BM25, Redis invalidation и изменение S3-config между запросами.
- [ ] Проверить server, worker, CLI upload/ingest/list и standalone benchmark lifecycle. Не запускать полный LLM benchmark ради проверки storage, если достаточно controlled fixture/mock.
- [ ] Проверить «S3 недоступен при старте»: lazy startup не открывает S3; первая операция завершается явной ошибкой/описанной деградацией, остальные корутины worker живы.
- [ ] Подготовить один server image tag для server и worker; записать предыдущий tag. Не обновлять MinIO, Qdrant или модели вместе с этой миграцией.
- [ ] Перед deploy остановить приём новых заданий/дождаться in-flight storage и BM25 jobs; корректно завершить процессы. Запускаемый stack изменять только по прямому запросу.
- [ ] Выпускать этап A и этап B отдельно; наблюдать реальные операции и heartbeat между ними. Этап B не требуется для быстрого закрытия PERF-001, но требуется для завершения полного перехода.

**Откат:** вернуть предыдущий server image одновременно для server и worker; завершить текущие процессы с cleanup и запустить предыдущую версию с прежними settings. Никаких downgrade БД, очистки bucket или переиндексации не предусмотрено. Незавершённый ingestion не должен иметь successful registry row и повторяется штатным способом. В dev-ветке откатывать только собственные migration commits, не чужие изменения.

**Триггеры отката:** missing sparse при существующем snapshot; потерянные BM25 mutations/ACL; ошибки целостности скачанного содержимого; незакрытые sessions/рост дескрипторов; необъяснённые heartbeat stalls; MinIO compatibility failures.

## 7. Команды и матрица верификации

Команды выполняются при реализации после соответствующего прямого запроса. Новые test-файлы должны существовать к моменту запуска.

### Unit/contract checks

Из `server/`:

```bash
uv run pytest tests/test_file_storage_async.py tests/test_s3_storage_async.py tests/test_s3_storage_download.py tests/test_storage_lifecycle.py -q
uv run pytest tests/test_ingestion_workflows.py tests/test_ingestion_service_pipeline.py tests/test_reingestion_consistency.py tests/test_ingestion_acl_consistency.py tests/test_ingestion_domain_validation.py -q
uv run pytest tests/test_bm25_lifecycle.py tests/test_bm25_incremental.py tests/test_chunk_service.py tests/test_document_service.py tests/test_document_commands_characterization.py tests/test_review_contracts.py -q
uv run pytest tests/test_cli_domain_bootstrap.py tests/test_cli_versioning.py tests/test_worker_resource_lifecycle.py tests/test_review_bug_regressions.py tests/test_ml_clients_port.py -q
uv run pytest tests/test_acl.py tests/test_bm25_acl.py tests/test_answer_cache.py tests/test_curator_role.py tests/test_retrieval_acl.py tests/test_rag_pipeline.py tests/test_rag_retrieval.py -q
uv run pytest tests/test_benchmark_characterization.py tests/test_worker_resource_lifecycle.py tests/test_solid_characterization.py -q
```

Для задачи 4 отдельно запустить переходный `tests/test_s3_snapshot_reader.py`; после задачи 7 файл удалить вместе с reader. Регрессионные тесты на final async loading остаются.

Из корня:

```bash
task lint
```

Task включает ruff и mypy. Формат проверять без автоматической правки unrelated files:

```bash
cd server
uv run ruff format --check app tests
```

### Architecture и остатки sync API

Из корня выполнить обязательные команды AGENTS.md:

```bash
grep -rn "^from \(application\|infrastructure\|presentation\)\|^import \(application\|infrastructure\|presentation\)" server/app/domain/
grep -rln "^from \(infrastructure\|presentation\)\|^import \(infrastructure\|presentation\)" server/app/application/
```

Ожидается отсутствие совпадений; exit code 1 для пустого grep нормален. Если есть baseline нарушения, задокументировать отдельно, не добавлять новые.

```bash
rg -n 'boto3|download_bytes|load_bm25_index_from_s3_sync|get_storage' server/app server/pyproject.toml
rg -n 'list_files\(|get_file_info\(|collect_entries\(|bm25_index\(|bm25_update\(' server/app
```

Первый поиск после этапа B пустой. Во втором все metadata calls awaited; cached-only BM25 доступ используется только по новому контракту, без сетевого I/O. Проверить также qualified imports и exports, которые простой текстовый поиск может не связать с вызовом.

```bash
cd server
uv tree --locked
```

Просмотреть aiobotocore/botocore/aiohttp versions и отсутствие boto3/s3transfer после этапа B; не утверждать, что resolve равен runtime smoke.

### Opt-in MinIO smoke

После подготовки тестовой конфигурации и прямого запроса пользователя, из `server/`:

```bash
uv run pytest tests/integration/test_s3_storage_minio.py -m integration -q
```

Credentials передаются штатным способом окружения/secret config; команды и test output не содержат их. Existing production bucket целиком не очищается.

## 8. Критерии завершения полного перехода

- [ ] S3 LIST/HEAD/GET/PUT/COPY/DELETE выполняются через aiobotocore, без boto3 I/O в loop или worker thread.
- [ ] Pool/semaphore reuse, response-body cleanup, cancelled thread-write и idempotent teardown покрыты тестами.
- [ ] CLI, cron, server, worker и standalone benchmark имеют явных владельцев storage; ни один client не переиспользуется между event loop.
- [ ] Runtime config rotation сохраняет in-flight операции и закрывает retired backends.
- [ ] BM25 load единый для generation; invalidation не публикует stale snapshot; CPU mutation не теряется; отсутствующий snapshot и failed load различаются.
- [ ] ACL triangle, cache scope и curator scope tests сохранены и зелёные; JSON/hashes/source keys совместимы с предыдущей версией.
- [ ] Registry commit sequence и application transaction behavior сохранены.
- [ ] MinIO compatibility, контент download и responsiveness подтверждены фактическими результатами; необъяснённых session warnings нет.
- [ ] Boto3/s3transfer и переходный sync reader удалены; lock зафиксирован.
- [ ] Runbook содержит команды запуска, предыдущий image tag, известные ограничения и процедуру отката.

## 9. Ограничения и последующие работы

Async S3 не делает parsing/OCR/chunking неблокирующими. Часть этих операций сейчас выполняется синхронно в ingestion; отдельно измерить их вклад после устранения PERF-001. Не обещать, что миграция исправляет весь event-loop lag.

Повторные HEAD в batch сохраняются в этом плане. Их можно убрать отдельным изменением, когда metadata будет сопровождать фактически скачанную версию объекта; нельзя записывать новый hash для старого разобранного контента при конкурентной замене файла.

Streaming GET может иметь другую скорость и retry-семантику по сравнению с boto3 managed transfer. Сравнить типичные и крупные файлы перед выпуском; multipart/range downloads не добавлять без измеренной необходимости.

Retired clients сохраняются до shutdown, поэтому частая смена S3-настроек удерживает несколько pools. Этот предел унаследован от текущей lifecycle-модели; закрытие retired клиента сразу после последней активной операции можно выделить отдельно при реальной необходимости.

## 10. Первичные источники

- [aiobotocore 3.9.2 dependencies](https://github.com/aio-libs/aiobotocore/blob/3.9.2/pyproject.toml) — Python/version bounds.
- [aiobotocore lifecycle и pagination](https://github.com/aio-libs/aiobotocore#context-manager-examples) — async client contexts и AsyncExitStack.
- [S3 async streaming example](https://aiobotocore.aio-libs.org/en/stable/examples/s3/basic_usage.html) — awaited get_object/read и body cleanup.
- [Boto3 managed download_fileobj](https://docs.aws.amazon.com/boto3/latest/reference/services/s3/client/download_fileobj.html) — transfer behavior, которое заменяется при миграции.
- [Botocore Config](https://docs.aws.amazon.com/botocore/latest/reference/config.html) — pool, addressing, timeouts/retries/checksums.
- [AWS integrity settings](https://docs.aws.amazon.com/sdkref/latest/guide/feature-dataintegrity.html) — проверка checksum compatibility.

Перед фактическим исполнением ещё раз проверить доступность выбранных версий и актуальный lock; замена версии требует повторить dependency resolution и MinIO smoke, а не автоматически расширить bounds.
