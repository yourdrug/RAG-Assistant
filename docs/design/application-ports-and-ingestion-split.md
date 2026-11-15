# Design: границы application-портов и расщепление ingestion-оркестратора

- **Статус**: Proposed
- **Дата**: 2026-09-13
- **Скоуп**: границы портов `DocumentParserPort` / sparse-index; расщепление
  `application/services/ingestion_orchestrator.py` (768 строк, 12 внешних импортов)
- **Связанные документы**: `docs/design/curator-scope-and-bm25-acl.md` (далее —
  [D1]; ссылки "D1 §..." указывают на его разделы)

---

## 3. Application ports design — границы DocumentParserPort и BM25-persistence

### 3.1 Инвентаризация текущих абстракций

Сегодня за парсинг/сплиттинг/sparse-индекс отвечают **шесть** разнородных
механизмов:

| # | Абстракция | Где живёт | Кто потребляет | Проблема |
|---|---|---|---|---|
| 1 | `DocumentParser`, `DocumentSplitter` (Protocols) | `domain/services/document_parser.py:17-28` | `DocumentProcessor` (API-путь загрузки) | Возвращает `RawDocument`; сигнатура `split(documents, domain)` слишком узкая для CLI-пути |
| 2 | `ContentExtractorPort`, `PDFQualityAssessorPort`, `TextQualityAssessorPort`, `MetricsCollectorPort` | `application/ports/document_processing.py` | `DocumentProcessor` | placement-несогласованность с #1: та же семья портов в двух слоях |
| 3 | Прямые импорты `PARSERS`, `parse_pdf`, `merge_pdf_pages`, `split_documents`, `split_documents_legal` | `infrastructure/ml/ingestion/` | `IngestionService` (CLI-путь) **напрямую** (ingestion_orchestrator.py:32-38) | Никакой абстракции; API- и CLI-пути расходятся |
| 4 | `BM25IndexPort` (add/remove/replace) | `application/ports/bm25_index.py` | `ChunkService`, `DocumentService` | Хороший порт: только примитивы. Но имя называет технологию |
| 5 | Отсутствует: build/persist/invalidate BM25 | `infrastructure/bm25/persistence.py`, `bm25_invalidation.py` | `IngestionService._build_bm25_index` вызывает `BM25Index(...)`, `load/save_bm25_index_to_s3`, `publish_bm25_invalidation` напрямую (ingestion_orchestrator.py:320-339) | Application конструирует инфраструктурный класс — ядро нарушения слойности |
| 6 | `IngestionRegistryRepository` (Protocol) | `domain/repositories/ingestion_registry_repository.py` | — **никто через порт**; оркестратор инстанцирует `SQLAlchemyIngestionRegistryRepository(uow._session)` (ingestion_orchestrator.py:112-113, 129, 137, 155) | Обход UoW через приватный `_session`; доменный интерфейс мёртв |

Дополнительный факт: через границу парсинга курсируют **две модели документа** —
`RawDocument` (domain, API-путь) и `langchain.schema.Document` (CLI-путь,
ingestion_orchestrator.py:28).

### 3.2 Таксономия leaky abstraction (что именно протекает)

Порт протекает, если выполняется хотя бы одно:

- **L1 — порт зеркалит внутренности реализации.** Например, порт,暴露ющий
  strategy-реестр `PARSERS` (dict по расширению), кортеж-конвенцию
  `(text, extra_meta)` или `merge_pdf_pages` как отдельный шаг. Мена формата
  PDF-парсинга меняет контракт порта.
- **L2 — порт транспортирует конкретный инфраструктурный тип.** Например,
  `save(index: BM25Index)` — приложение вынуждено конструировать объект
  инфраструктуры, т.е. нарушение переезжает из импорта в сигнатуру.
- **L3 — порт замораживает модель реализации.** Например,
  `persist(texts: list[str])` фиксирует "sparse-индекс = плоский список текстов"
  и блокирует: (a) per-doc ACL-атрибуты в индексе — D1 §2, фаза 1; (b) переход на
  native Qdrant sparse — D1 §2, фаза 2.
- **L4 — порт shaped одним потребителем.** `split(documents, domain)` сегодня,
  `split(documents, domain, profile, settings, legal_mode)` завтра — сигнатура
  отражает нужды одного вызова, а не суть операции.
- **L5 — port ceremony без выгоды.** Порт с единственной реализацией, у которого
  нет ни второго адаптера в перспективе, ни фейка для тестов, — это лишний
  косвенный слой.

Правило проектирования: **порт выражает намерение и оперирует только
domain/application типами; вся механика принадлежит адаптеру.**

### 3.3 DocumentParserPort — рекомендуемая граница

#### Отклонённые формы

- **Зеркало function-registry** (`parse_by_ext(ext, path) -> tuple[str, dict]`,
  `merge_pages(docs)`, `split(docs, domain, profile, settings)`): L1 + L4 —
  порт повторяет сегодняшний call-graph оркестратора; смена парсера или
  сплиттера ломает контракт.
- **Отказ от порта вообще** (оба пути зовут concrete-функции): легально только
  если CLI-оркестратор переезжает в infrastructure (решение §4). Но API-путь
  (`DocumentProcessor`) уже зависит от протоколов — их придётся либо сохранить,
  либо сломать; плюс фейки в тестах DocumentProcessor строятся именно на
  протоколах. Отклоняем.

#### Рекомендуемая форма

```python
# application/ports/document_parser.py  (консолидация: сюда переезжают
# DocumentParser/DocumentSplitter из domain/services/document_parser.py)

@dataclass(frozen=True)
class FileMeta:
    """Транспортные метаданные файла, известные вызывающей стороне."""
    source_key: str        # напр. "s3://bucket/docs/x.pdf" — генерирует caller
    filename: str
    extension: str         # нормализованный lowercase, с точкой
    size_bytes: int


@runtime_checkable
class DocumentParserPort(Protocol):
    def supports(self, extension: str) -> bool: ...
    def parse(self, path: Path, meta: FileMeta) -> list[RawDocument]: ...


@dataclass(frozen=True)
class SplitContext:
    """Домен-зависимые параметры сплиттинга (domain-типы в порте — допустимо)."""
    domain: str
    profile: DomainProfile | None = None


@runtime_checkable
class DocumentSplitterPort(Protocol):
    def split(self, documents: list[RawDocument], context: SplitContext) -> list[RawDocument]: ...
```

Что intentionally скрыто в адаптере (L1-изоляция):

- выбор стратегии по расширению (`PARSERS`-dict), включая PDF-спецветку;
- `merge_pdf_pages` — деталь PDF-парсинга, не шаг пайплайна;
- кортеж-конвенция `(text, extra_meta)` — адаптер сам мержит `extra_meta` в
  метаданные `RawDocument`.

Что intentionally **не** скрыто (политика, единая для всех форматов):

- отсечение "слишком мало текста" (< 20 chars, ingestion_orchestrator.py:696) —
  это политика качества, а не механика парсинга; остаётся в вызывающей стороне
  (и становится одинаково тестируемой для всех форматов);
- логирование ошибок парсинга с именем файла — caller знает контекст запроса.

Что даёт консолидация placement'а: семья портов обработки документа собирается в
одном месте (`application/ports/document_parser.py` +
`application/ports/document_processing.py`), из `domain/services/document_parser.py`
остаётся re-export на переходный период. `RawDocument` остаётся domain-сущностью —
порт в application может импортировать domain (направление зависимостей по
AGENTS.md). Побочный эффект: **langchain Document исчезает из контракта**;
`LCDocument` остаётся только внутри infrastructure (retrieval), ingestion-путь
переводится на `RawDocument`.

Единственность реализации — не ceremony (L5), потому что у порта два потребителя
(`DocumentProcessor` — API; pipeline из §4 — CLI) и фейки для обоих тестовых
комплектов.

### 3.4 BM25-persistence — порт с максимальным риском утечки

Потребность application-слоя сегодня: "sparse-индекс должен отражать эти чанки".
Рассмотрим три формы порта.

#### Форма S1 — `save(index) / load() -> BM25Index` (отклоняется, L2)

Application строит `BM25Index(all_texts)` и отдаёт на сохранение — ровно то, что
происходит сегодня в `_build_bm25_index` (ingestion_orchestrator.py:337-339).
Нарушение не устраняется, а легализуется сигнатурой.

#### Форма S2 — `persist(texts, hashes) / load_texts() -> list[str]` (отклоняется, L3)

Убирает L2, но замораживает модель "индекс = плоский список текстов":

- D1 §2 фаза 1 требует хранить в индексе per-doc ACL-атрибуты
  (`visibility`, `owner_id`, `group_id`) —texts-only порт их не пропустит;
- D1 §2 фаза 2 (native Qdrant sparse) меняет носитель с "список текстов в JSON
  на S3" на "sparse-вектора в Qdrant" — порт придётся ломать.

#### Форма S3 — intent-level порт (рекомендуется)

```python
# application/ports/sparse_index.py

@dataclass(frozen=True)
class ChunkRef:
    """Минимальная ссылка на чанк для пересборки sparse-индекса."""
    text: str
    content_hash: str
    # D1 §2 фаза 1 (расширение, не ломающее контракт):
    # visibility: str = "internal_public"
    # owner_id: int | None = None
    # group_id: int | None = None


@runtime_checkable
class SparseIndexAdminPort(Protocol):
    """Пакетное администрирование sparse-индекса из ingestion-пайплайна."""

    async def rebuild(self, chunks: Sequence[ChunkRef]) -> None:
        """Полная пересборка (reset-режим)."""

    async def extend(self, chunks: Sequence[ChunkRef]) -> None:
        """Инкрементальное добавление (append-режим)."""
```

Адаптер (новый, `infrastructure/bm25/sparse_index_admin.py`) инкапсулирует всю
механику today: load-from-S3 (для extend), построение `BM25Index`,
save-to-S3, `publish_bm25_invalidation()` (ingestion_orchestrator.py:320-339
переезжает сюда целиком). Tomorrow (D1 фаза 2) тот же порт реализуется upsert'ом
sparse-векторов в Qdrant — контракт не меняется.

Отношение к существующему `BM25IndexPort`: **оба порта остаются**.

| Порт | Семантика | Потребители |
|---|---|---|
| `BM25IndexPort` (add/remove/replace) | точечные мутации после ручных правок | `ChunkService`, `DocumentService` |
| `SparseIndexAdminPort` (rebuild/extend) | пакетные операции ingestion-пайплайна | ingestion pipeline (§4) |

Один адаптер-класс реализует оба; `BM25IndexAdapter`
(infrastructure/bm25/bm25_updater.py:89) остаётся вторым адаптером той же
мутационной логики поверх `MLClientRegistry` — при желании объединяются в один
файл. Попутное переименование `BM25IndexPort` → `SparseIndexMutationPort`
(имя порта не должно называть технологию — в отличие от имени
infrastructure-класса, где конвенция `<Tech><PortName>` наоборот требует её
называть); низкий приоритет, механическое.

Проверка на L5: у `SparseIndexAdminPort` единственная реализация, но есть
(а) вторая реализация в перспективе (D1 фаза 2), (б) фейк для
characterization-тестов пайплайна (§4.4) без S3/Redis — порт окупается.

### 3.5 Ingestion registry — закрыть обход UoW

Доменный протокол `IngestionRegistryRepository` уже существует, но UoW его не
экспонирует (`infrastructure/uow_factory.py:84-100` — списка репозиториев нет),
и оркестратор лазает в `uow._session` (ingestion_orchestrator.py:113).

Решение: добавить `ingestion_registry=SQLAlchemyIngestionRegistryRepository(session)`
в фабрику UoW и удалить прямую инстанциацию. Это независимая правка, не ждущая
§4 (устраняет доступ к приватному атрибуту из application-слоя).

### 3.6 Итоговая таблица портов

| Порт | Действие |
|---|---|
| `DocumentParserPort` + `DocumentSplitterPort` (консолидированные) | **Добавить** в `application/ports/document_parser.py`; убрать протоколы из `domain/services/document_parser.py` (re-export на переходный период) |
| `ContentExtractorPort`, `PDFQualityAssessorPort`, `TextQualityAssessorPort`, `MetricsCollectorPort` | Оставить как есть |
| `FileStorage` | Оставить (хорошая форма: DTO `FileItem`, только примитивы) |
| `BM25IndexPort` | Оставить; опционально переименовать в `SparseIndexMutationPort` |
| `SparseIndexAdminPort` | **Добавить** (форма S3) |
| `save/load BM25Index` как порт | **Отклонить** (L2) |
| `persist(texts)` как порт | **Отклонить** (L3) |
| Порты `PARSERS`-registry / `merge_pdf_pages` | **Отклонить** (L1) |

---

## 4. Ingestion orchestrator split — application vs infrastructure

### 4.1 Текущее состояние

`application/services/ingestion_orchestrator.py`, класс `IngestionService`:

- **768 строк**, docstring первой строки: *"Infrastructure implementation of the
  S3-only document ingestion pipeline"* — файл сам признаёт, что лежит не в своём слое.
- **Импортирует напрямую**: `infrastructure.bm25.*` (2 модуля),
  `infrastructure.ml.ingestion`, `infrastructure.metrics`,
  `infrastructure.repositories.*`, `infrastructure.storage`,
  `infrastructure.uow_factory`, плюс `config.settings` и `langchain.schema` —
  9 запрещённых для application импортов. Enforced-греп из AGENTS.md возвращает
  непусто (нарушителей в application/ всего три: этот файл,
  `benchmark_orchestrator.py`, `retrieval.py`).
- Потребители:

| Потребитель | Как зависит | Где |
|---|---|---|
| API route `/ingest` | через `IngestionPort` (DI) — корректно | presentation/api/routes/ingest.py:133 |
| API dependencies | **импортирует конкретный класс** | presentation/api/dependencies.py:42 |
| CLI `ingest` | **импортирует конкретный класс** и сам его собирает | presentation/cli/commands/ingest.py:12-27 |
| `IngestAppService` | через `IngestionPort` — корректно | application/services/ingest_service.py:21 |
| composition | собирает | composition/service_providers.py:23-53 |

- Внутри класса смешаны три рода ответственности:
  1. **Оркестрация use case**: порядок шагов, инвариант "registry-строка
     коммитится только после sync-транзакции" (ingestion_orchestrator.py:197-203,
     293-297), reset-семантика, owner_id=None для CLI.
  2. **Инфраструктурные адаптеры**: `_build_bm25_index` (S3 + Redis pub/sub),
     `_parse_file` (concrete-парсеры), `_registry_*` (SQLAlchemy через
     `uow._session`), `_load_documents` (S3 scan).
  3. **Presentation-подобное логирование**: баннеры `"=" * 55`, форматирование
     прогресса `[  3/42] CACHED ...`.
- Уже существуют правильные швы: `IngestionPort` (контракт),
  `IngestAppService` (тонкая application-фаска, 93 строки),
  `document_pipeline.py` (общая логика classify/enrich/process_chunks, 289 строк).

### 4.2 Варианты

#### Вариант A — оставить в application, "запортить" всё

Для соблюдения слойности понадобятся: DocumentParserPort+SplitterPort (§3.3),
SparseIndexAdminPort (§3.4), IngestionRegistry через UoW (§3.5), metrics — порт
уже есть (`MetricsCollectorPort`). Файл остаётся оркестратором, но:

- Половина портов будет иметь единственную реализацию и единственного
  потребителя — риск L5-ceremony.
- Класс всё равно оркестрирует инфраструктурные по сути шаги (S3-scan,
  download-to-temp, BM25 rebuild) — порты лишь переименуют прямые вызовы.
- Сигнатуры портов придётся расширять под нужды ЭТОГО класса (сплиттинг с
  profile/settings) — риск L4.
- Деловая логика, которую хотелось бы защитить слоем (visibility-теги,
  классификация, версионирование), уже живёт в domain/application
  (`document_pipeline`, `ActVersioningService`, `_tag_chunks` — тривиальный
  маппинг). Защищать нечего — оркестратор это механика.

**Отклоняется**: высокая ceremony, нулевая защита бизнес-правил.

#### Вариант B — перенести класс в infrastructure, реализующим IngestionPort

`IngestionPort` уже определён в application/ports и уже потребляется роутом и
`IngestAppService`. Перенос реализации в infrastructure — это ровно направление,
заявленное AGENTS.md ("Infrastructure implements abstractions required by
Application/Domain") и собственной docstring файла.

**Принимается** — с внутренним расщеплением (B+C) и предварительными
characterization-тестами.

#### Вариант C — расщепить на application-usecase + infra-pipeline

В чистом виде избыточен: application-usecase уже существует (`IngestAppService`).
Полезная часть C — внутренняя декомпозиция класса — включается в B.

### 4.3 Рекомендуемая структура

```
application/
  ports/ingestion_port.py          ← контракт, без изменений
  ports/document_parser.py         ← НОВОЕ (§3.3)
  ports/sparse_index.py            ← НОВОЕ (§3.4)
  services/ingest_service.py       ← IngestAppService, без изменений
  services/document_pipeline.py    ← + сюда переезжают _tag_chunks/_tag_domain
                                     (метаданные ACL-тегирования — общая
                                     политика для API и CLI путей)

infrastructure/
  ingestion/
    s3_ingestion_pipeline.py       ← класс IngestionService (переезд, implements
                                     IngestionPort); оркестрирует коллабораторов
    file_source.py                 ← _load_documents, _parse_file, _s3_file_hash,
                                     _s3_source_key, _base_metadata (S3-scan+parse)
    db_sync.py                     ← _ensure_document_rows, _sync_one_document,
                                     _sync_documents_to_db — оба фазы и
                                     transaction-инварианты
    registry_bookkeeping.py        ← _registry_* (через uow.ingestion_registry, §3.5)
  bm25/
    sparse_index_admin.py          ← адаптер SparseIndexAdminPort
                                     (реализация _build_bm25_index)

composition/service_providers.py   ← правка импортов (create_ingestion_service)
presentation/cli/commands/ingest.py ← сборка через composition, не напрямую
presentation/api/dependencies.py    ← тип по IngestionPort, не конкретный класс
```

Правила расщепления:

1. **Класс-оркестратор тонкий**: только порядок шагов и проброс зависимостей;
   каждый коллаборатор получает зависимости конструктором (никаких module-level
   singleton'ов), тестируется отдельно.
2. **Transaction-инварианты переезжают вместе с кодом и документируются в
   docstring `db_sync.py`**: (a) registry-строка только после коммита sync
   (ingestion_orchestrator.py:197-203 — иначе crash между upsert и sync навсегда
   прячет файл как CACHED); (b) двухфазный sync: document-rows (коммит) →
   versioning (собственная tx) + chunks + outbox (ingestion_orchestrator.py:341-357);
   (c) SET_DOCUMENT_ID только через outbox, никогда inline (ingestion_orchestrator.py:504-514).
3. **`config.settings` и `langchain` в pipeline становятся легальными**
   (infrastructure может их импортировать). Проверить, что из 12 импортов
   останутся только нужные: ожидаемо −`langchain` (перевод на `RawDocument`,
   §3.3), −`SQLAlchemyIngestionRegistryRepository` (через UoW), −`bm25.*`
   (через адаптер порта).
4. **Логирование**: баннеры/прогресс остаются в инфраструктурном пайплайне (это
   CLI-UX, не бизнес-событие); никаких `print()` — уже так.
5. **Переименование**: `IngestionService` → `S3IngestionPipeline` (имя по
   конвенции `<Tech><...>`; старое имя оставить как alias на один релиз —
   внешний потребитель только composition + CLI, оба перезаправляются).

### 4.4 План миграции

Предусловие по AGENTS.md (правило god-files: не трогать без characterization-тестов;
файл формально не в таблице бэклога — таблица устарела, `infrastructure/ml/ingestion.py`
уже расщеплён на пакет, — но правило применим по духу):

| # | Шаг | Проверка |
|---|---|---|
| 1 | Characterization-тесты на публичные методы (`run_full_ingestion`, `run_single_file`, `get_registry`, `force_reindex`, `upload_files`) с фейками FileStorage/UoW/SparseIndexAdminPort: reset vs append, порядок registry-after-commit, visibility-теги чанков, owner_id=None, skip-по-хэшу | baseline зелёный ДО переезда |
| 2 | §3.5: `uow.ingestion_registry` в фабрике; удалить `uow._session`-доступ | тесты регресс; grep `_session` в application → пусто |
| 3 | Переезд класса + коллабораторы (§4.3), правка 3 точек импорта (composition, CLI, api/dependencies) | characterization-тесты зелёные без правки ассертов; `python main.py ingest list` из Docker работает |
| 4 | Ввести `SparseIndexAdminPort` + адаптер; pipeline потребляет порт | characterization: rebuild вызывается при reset, extend при append, invalidation публикуется |
| 5 | Консолидация DocumentParserPort (§3.3); CLI-путь переведён на адаптер порта; `langchain.Document` удалён из ingestion-контура | layering-grep из AGENTS.md по application → пусто для ingestion; `retrieval.py`/`benchmark_orchestrator.py` — отдельные follow-up тикеты |
| 6 | Обновить AGENTS.md: god-file таблицу (удалить #3, добавить текущее состояние), зафиксировать расположение ingestion-пайплайна | review |

Шаги 3 и 4 можно объединить в один PR (переезд + порт), если characterization-покрытие
из шага 1 достаточное; шаги 2 и 5 независимы.

### 4.5 Литмус-тест "что осталось в application"

| Ответственность из 768 строк | Слой | Обоснование |
|---|---|---|
| Видимость запроса ingestion (доступ к запуску, статусы) | presentation (routes) | уже там |
| Use case "запустить ingestion / спросить registry" | application: `IngestAppService` | тонкая фаска над портом — уже есть |
| Классификация домена, enrich метаданных, persist чанков | application: `document_pipeline` | общая для API и CLI — уже есть |
| Версионирование (акты) | application: `ActVersioningService` | уже есть |
| ACL-тегирование чанков (`_tag_chunks`/`_tag_domain`) | application: `document_pipeline` (перенос) | политика, общая для обоих путей |
| Порядок шагов пайплайна, reset-семантика | infrastructure: `S3IngestionPipeline` | механика S3/парсеров/BM25; контракт — `IngestionPort` |
| S3 scan/download, парсинг форматов | infrastructure | |
| BM25 build/persist/invalidate | infrastructure (адаптер порта) | |
| Registry CRUD | infrastructure (через UoW + доменный протокол) | |
| Transaction-инварианты sync | infrastructure: `db_sync.py` + characterization-тесты | граница транзакций — механика хранилищ |

### 4.6 Риски

- **CLI-совместимость**: `python main.py ingest ...` (AGENTS.md, Docker-команды)
  не должен сломаться — точка сборки одна (`presentation/cli/commands/ingest.py`),
  переводится на composition; проверяется шагом 3.
- **Worker-задачи**: `create_ingest_app_service` (service_providers.py:96-114)
  уже собирает через composition — не затрагивается, только внутренний импорт.
- **Скрытые импорты по строке "from application.services.ingestion_orchestrator
  import ..."**: полный grep перед шагом 3 (сейчас известны 4 точки —
  composition ×2, CLI, api/dependencies).
- **Тесты, импортирующие класс напрямую**: проверить `server/tests/` на шаге 1
  (characterization-тесты пишутся против `IngestionPort`, не конкретного класса).

---

## 5. Взаимодействие с [D1]

1. **SparseIndexAdminPort — мост к D1 §2**: ChunkRef уже спроектирован под
   расширение ACL-атрибутами (D1 фаза 1), не ломая контракт; адаптер единственная
   точка, где индекс узнаёт о per-doc visibility/owner/group. Переход на native
   Qdrant sparse (D1 фаза 2) = смена адаптера.
2. **ACL-тегирование в document_pipeline** (§4.5): с D1 §1 (CuratorScope) ничего
   не пересекается — ingestion пишет теги, retrieval их читает; но
   characterization-тесты тегирования (шаг 1) фиксируют формат
   `visibility/owner_id/group_id/client_id` в метаданных чанков, на который
   опирается `build_qdrant_filter` (infrastructure/repositories/vector/acl.py:66-77).
3. **Порядок внедрения относительно D1**: независимо; пересечение только в
   формате метаданных чанка (п. 2 выше) — поэтому characterization-тесты
   тегирования полезны обоим документам.
