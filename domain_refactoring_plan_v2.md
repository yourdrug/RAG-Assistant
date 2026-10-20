# План рефакторинга `server/app/domain/`

Сводный план по всем обсуждениям: структура по bounded context, разделение `ingestion` (запись) и `rag` (чтение), без shim-ов и обратной совместимости. В конце список открытых вопросов, на которые нужно ответить в ходе работы.

---

## 1. Проблемы текущей структуры

1. `value_objects/` стала свалкой (~50 файлов). Настоящих VO меньше половины. Остальное: конфиги, DTO, read-модели, regex-паттерны, 15+ файлов по одному enum.
2. Слой группируется по техническому типу, а не по смыслу. Чтобы понять бенчмарки, нужно открыть 4 папки; то же для evidence и regulatory acts.
3. Порты разбросаны: `repositories/`, `services/token_provider.py`, `services/password_hasher.py`, `services/document_parser.py`, `domain_profile/settings_port.py`. Нейминг разный (`ITokenProvider` против `GroupRepository`).
4. `utils.py` — мешок несвязанных функций.
5. Типы лежат внутри файлов репозиториев (`ChunkSearchResult`, `ChunkStats`, `ConversationListItem`, `IngestionRegistryEntry`).
6. Смешение записи и чтения: `DomainProfile` обслуживает и индексацию, и генерацию ответа; `ChunkRepository` и `VectorStoreRepository` совмещают write- и read-методы.
7. Файлы не на своём месте: `ActVersionTimeline` в `entities/`, `raw_document.py` в `entities/`, `Breadth` в `llm_provider.py`, `not_found_patterns` отдельно от `prompt_blocks`, живой `classify_doc_type` рядом с deprecated-кодом.
8. Логика конкретного корпуса (ЭТТН/ЭТН/BLRDLN, «логистический оператор») зашита в `evidence_*` и промпты.
9. Три модели конфигурации: `ConfigDefault`, `DynamicParam`, `ConfigParameter`.

### Найденные дефекты

- В `value_objects/__init__.py` в `__all__` есть `StreamEvent`, но он не импортирован.
- В `repositories/__init__.py` не экспортирован `ChunkSearchPageRepository`.
- В `access_control.py` стоит `if False:  # TYPE_CHECKING`.
- В `LifecycleStatusMixin` докстринг упоминает `is_terminal`, свойства нет; пример использования устарел.
- В `entities/__init__.py` экспортировано 5 из ~15 сущностей.
- В `cost_budget.py` захардкожена строка `'deepinfra'`.
- В `stream_events.py` `UsageReport` используется в аннотации раньше объявления.

---

## 2. Принципы

1. Верхний уровень делится по смыслу. Папок `entities/`, `value_objects/`, `services/`, `repositories/`, `exceptions/`, `events/` и `utils.py` на верхнем уровне нет.
2. Внутри контекста всё плоско. Подпакет создаётся, если в нём 3 и больше связанных файла.
3. Enum лежит рядом с единственным владельцем; общие собираются в `enums.py` контекста.
4. Все `__init__.py` пустые: без `__all__`, без реэкспортов. Импорт только из конкретного модуля.
5. Без shim-файлов, без `DeprecationWarning`, без обратной совместимости. Старые пути удаляются сразу.
6. В `shared` попадает только то, что импортируют минимум два контекста.
7. Порты лежат в `repositories.py` или `ports.py` контекста.
8. `ingestion` отвечает за запись (файл → чанки → индекс), `rag` за чтение (вопрос → retrieval → ответ). Прямых импортов между ними нет.

---

## 3. Целевая структура

```
domain/
├── shared/                          (общее ядро, не зависит от контекстов)
│   ├── exceptions.py                (иерархия доменных ошибок 4xx/5xx с as_dict)
│   ├── enums.py                     (ValidatedEnumMixin, LifecycleStatusMixin)
│   ├── pagination.py                (CursorPage для keyset-пагинации)
│   ├── hashing.py                   (content_hash для дедупликации)
│   ├── dates.py                     (parse_date_guess, русские месяцы)
│   └── stats.py                     (percentile)
│
├── identity/                        (пользователи, роли, доступ)
│   ├── roles.py                     (UserRole, UserKind)
│   ├── capabilities.py              (карта «роль → права»)
│   ├── user_context.py              (UserContext, CuratorScope: снимок прав запроса)
│   ├── user.py                      (агрегат User с инвариантами роли и вида)
│   ├── api_key.py                   (ApiKey, ApiKeyClientInfo: статические ключи клиентов)
│   ├── group.py                     (GroupInfo, GroupMemberInfo)
│   ├── ports.py                     (TokenProvider, PasswordHasher: JWT и хеширование)
│   └── repositories.py              (User-, ApiKey-, Group-, Assignment-репозитории)
│
├── knowledge/                       (база знаний: документы, чанки, ACL, версии актов)
│   ├── enums.py                     (DocDomain, DomainDetectionMode, DocDomainSelection, DocumentStatus,
│   │                                 DOCUMENT_SLOT_STATUSES, DocumentVisibility, SourceType, SearchMode,
│   │                                 PageContentType: тип содержимого для ingestion и RAG)
│   ├── document.py                  (агрегат Document: жизненный цикл и права)
│   ├── chunk.py                     (Chunk, ChunkRecord, ChunkStats, контекст чанка, SENTENCE_UNIT_KIND,
│   │                                 encode/decode_cursor)
│   ├── access_control.py            (правила видимости: единый источник ACL-условий; OwnerMatch)
│   ├── index_sync.py                (VectorOutboxEntry, OutboxOperation/Status, протоколы outbox,
│   │                                 VectorIndexWriter: запись в Qdrant)
│   ├── acts/
│   │   ├── models.py                (RegulatoryAct, ActVersion: акт и его редакция)
│   │   ├── timeline.py              (ActVersionTimeline: интервалы действия редакций)
│   │   └── policy.py                (same_document_scope, document_scope_key)
│   └── repositories/                (порты хранилищ знаний)
│       ├── document.py              (CRUD и maintenance документов)
│       ├── chunk.py                 (ChunkCrudRepository, ChunkVersioningRepository, ChunkCorpusRepository)
│       └── acts.py                  (ActVersion- и RegulatoryAct-репозитории)
│
├── ingestion/                       (запись: разбор файла → чанки → индекс)
│   ├── raw_document.py              (разобранный текст до разбиения)
│   ├── ports.py                     (DocumentParser, DocumentSplitter, DomainSettingsPort)
│   ├── content_splitter.py          (рекурсивное разбиение по структурным границам)
│   ├── doc_type.py                  (эвристика типа документа: приказ, договор и т.п.)
│   ├── references.py                (extract_act_number из ReferenceMatch)
│   ├── versioning.py                (решение version или replace при конфликте)
│   ├── preview.py                   (PreviewUnitKind, DTO dry-run, качество PDF)
│   ├── text_quality.py              (классификация текста: скан, мусор, таблица)
│   ├── repositories.py              (реестр проиндексированных файлов)
│   └── profiles/                    (как разбирать документ данного домена)
│       ├── types.py                 (ConfigDefault, BoundaryLevel, ReferenceMatch,
│       │                             EffectiveDateCandidate, refs_to_metadata)
│       ├── protocol.py              (DocumentProfile)
│       ├── registry.py              (классификация: fingerprint → density-score)
│       ├── base.py                  (общая логика профилей, извлечение даты вступления в силу)
│       ├── general.py               (неструктурированные документы)
│       ├── legal.py                 (законы и договоры: глава/статья/пункт; regex без промптов)
│       └── decree.py                (указы и постановления: пункт/подпункт; regex без промптов)
│
├── rag/                             (чтение: вопрос → retrieval → ответ)
│   ├── breadth.py                   (Breadth narrow/broad и алиасы)
│   ├── context.py                   (RagContext, бывший ChatContext: права для retrieval)
│   ├── settings.py                  (RagSettings: снимок динамического конфига на запрос)
│   ├── result.py                    (RagResult: итог пайплайна для логирования)
│   ├── stream_events.py             (типизированные события стрима ответа)
│   ├── ports.py                     (ChunkSearch/Page/Context, VectorSearch: чтение)
│   ├── classification.py            (ширина вопроса, декомпозиция, out-of-domain, explicit_question_date)
│   ├── answer_policy.py             (AnswerPolicy: промпт-правила и fallback по домену,
│   │                                 versioned_prompt_date_stamp)
│   ├── prompt_blocks.py             (тексты промптов, NOT_FOUND_ANSWER, NOT_FOUND_PATTERNS)
│   ├── prompts.py                   (сборка системного промпта и сообщения с контекстом)
│   ├── retrieval.py                 (лимиты, веса, бюджет контекста, RRF, дедупликация, разнообразие,
│   │                                 compute_reranker_score, ContextBudgetExceededError)
│   ├── guardrails/
│   │   ├── injection.py             (паттерны prompt injection)
│   │   ├── pii.py                   (regex персональных данных RU/BY)
│   │   └── sanitization.py          (экранирование недоверенного текста в промпте)
│   └── evidence/                    (логика конкретного корпуса, изолирована)
│       ├── assessment.py            (статусы и структуры оценки доказательств)
│       ├── queries.py               (дополнительные поисковые запросы по таблицам и режимам)
│       ├── applicability.py         (применимость этапов процедуры и режимов передачи)
│       └── reading.py               (правила чтения таблиц, норм, поправок для промпта)
│
├── chat/                            (диалоги и журнал качества)
│   ├── message.py                   (Message, MessageRole: реплика диалога)
│   ├── conversation.py              (агрегат Conversation, ConversationListItem)
│   ├── chat_log.py                  (ChatLog: лог вопрос-ответ)
│   └── repositories.py              (Conversation-, Message-, ChatLog-репозитории)
│
├── benchmark/                       (оценка качества RAG)
│   ├── question.py                  (BenchmarkQuestion, валидация аннотаций, DEFAULT_DATASET)
│   ├── run.py                       (BenchmarkRun: один прогон с метриками)
│   ├── sweep.py                     (BenchmarkSweep, статус, стратегия, режим оценки)
│   ├── scoring.py                   (шкалы метрик и веса целевой функции)
│   ├── evaluation.py                (метрики retrieval и evidence, source hints)
│   ├── judge.py                     (покрытие и ошибки оценок LLM-судьи)
│   ├── selection.py                 (отбор разнообразных конфигураций для sweep)
│   └── repositories.py              (Question-, Run-, Sweep-репозитории)
│
└── runtime/                         (эксплуатация: конфиги, задачи, стоимость)
    ├── enums.py                     (LLMProvider, FileBackend, AppStage, HealthStatus)
    ├── jobs.py                      (BackgroundJob, JobStatus, JobType: фоновые задачи)
    ├── config.py                    (ConfigParameter, ConfigValueType, ConfigParameterChanged, parse_bool)
    ├── dynamic_params.py            (DYNAMIC_PARAMS: горячо перезагружаемые параметры)
    ├── setting_keys.py              (классификация ключей: secret, static, metadata, target)
    ├── cost.py                      (учёт USD в нанодолларах, тарифы, CostRate)
    └── repositories.py              (BackgroundJob- и ConfigParameter-репозитории)
```

Пакет назван `runtime`, а не `platform`: имя совпадает со stdlib.

---

## 4. Карта переноса (старое → новое)

### shared
| Старое | Новое |
|---|---|
| `exceptions/domain_errors.py` (кроме `ContextBudgetExceededError`) | `shared/exceptions.py` |
| `value_objects/base.py`, `lifecycle_status.py` | `shared/enums.py` |
| `value_objects/cursor_page.py` | `shared/pagination.py` |
| `utils.content_hash` | `shared/hashing.py` |
| `domain_profile/date_parsing.parse_date_guess` + `_RU_MONTHS` | `shared/dates.py` |
| `utils.percentile` | `shared/stats.py` |

### identity
| Старое | Новое |
|---|---|
| `value_objects/roles.py` | `identity/roles.py` |
| `value_objects/capabilities.py` | `identity/capabilities.py` |
| `value_objects/user_context.py`, `curator_scope.py` | `identity/user_context.py` |
| `entities/user.py` | `identity/user.py` |
| `entities/api_key.py`, `ApiKeyClientInfo` | `identity/api_key.py` |
| `GroupInfo`, `GroupMemberInfo` (`query_results.py`) | `identity/group.py` |
| `services/token_provider.py`, `password_hasher.py` | `identity/ports.py` |
| `user_`, `api_key_`, `group_`, `assignment_repository` | `identity/repositories.py` |

### knowledge
| Старое | Новое |
|---|---|
| `doc_domain`, `document_status`, `visibility`, `source_type`, `search_mode`, `page_content_type` | `knowledge/enums.py` |
| `entities/document.py` | `knowledge/document.py` |
| `entities/chunk.py`, `ChunkSearchResult`, `ChunkStats`, `chunk_context.py`, `utils.encode/decode_cursor`, `SENTENCE_UNIT_KIND` | `knowledge/chunk.py` |
| `services/access_control.py`, `owner_match.py` | `knowledge/access_control.py` |
| `entities/vector_outbox_entry.py`, `vector_outbox_repository.py`, write-часть `vector_store_repository.py` | `knowledge/index_sync.py` |
| `entities/regulatory_act.py`, `act_version.py` | `knowledge/acts/models.py` |
| `entities/act_version_timeline.py` | `knowledge/acts/timeline.py` |
| `same_document_scope`, `document_scope_key` | `knowledge/acts/policy.py` |
| `document_repository.py` | `knowledge/repositories/document.py` |
| `ChunkCrudRepository`, `ChunkVersioningRepository`, `ChunkCorpusRepository` | `knowledge/repositories/chunk.py` |
| `act_version_repository.py`, `regulatory_act_repository.py` | `knowledge/repositories/acts.py` |

### ingestion
| Старое | Новое |
|---|---|
| `entities/raw_document.py` | `ingestion/raw_document.py` |
| `services/document_parser.py`, `domain_profile/settings_port.py` | `ingestion/ports.py` |
| `domain_profile/content_splitter.py` | `ingestion/content_splitter.py` |
| `classify_doc_type`, `DOC_TYPE_KEYWORDS`, `DOC_TYPE_SAMPLE_CHARS` | `ingestion/doc_type.py` (выполнено на шаге 1) |
| `extract_act_number` (`act_version_policy.py`) | `ingestion/references.py` |
| `document_versioning.py` | `ingestion/versioning.py` |
| `preview_unit_kind`, `pdf_dto`, `pdf_quality_report` | `ingestion/preview.py` |
| `services/text_quality.py` | `ingestion/text_quality.py` |
| `ingestion_registry_repository.py` | `ingestion/repositories.py` |
| `ConfigDefault`, `BoundaryLevel`, `ReferenceMatch`, `EffectiveDateCandidate`, `refs_to_metadata` | `ingestion/profiles/types.py` |
| индексационные методы `DomainProfile` | `ingestion/profiles/protocol.py` (`DocumentProfile`) |
| `domain_profile/registry.py`, `base.py` | `ingestion/profiles/registry.py`, `base.py` |
| `profiles/general.py`, `legal.py`, `decree.py` (без промптов) | `ingestion/profiles/*.py` |

### rag
| Старое | Новое |
|---|---|
| `Breadth`, `BREADTH_ALIASES` (`llm_provider.py`) | `rag/breadth.py` |
| `chat_context.py` (`ChatContext` → `RagContext`) | `rag/context.py` |
| `rag_settings.py` | `rag/settings.py` |
| `rag_result.py` | `rag/result.py` |
| `stream_events.py` | `rag/stream_events.py` |
| search/context-часть `chunk_repository.py` и read-часть `vector_store_repository.py` | `rag/ports.py` |
| `rag_policy/classification.py`, `explicit_question_date` | `rag/classification.py` |
| `prompt_addendum`, `retrieval_fallback_to_full_corpus`, `_LEGAL_PROMPT_RULES`, `_DECREE_PROMPT_RULES`, `versioned_prompt_date_stamp` | `rag/answer_policy.py` |
| `rag_policy/prompt_blocks.py`, `not_found_patterns.py` | `rag/prompt_blocks.py` |
| `rag_policy/prompts.py` | `rag/prompts.py` |
| `rag_policy/retrieval.py`, `retrieval_diversity.py`, `utils.rrf_merge/deduplicate_docs/compute_reranker_score`, `ContextBudgetExceededError` | `rag/retrieval.py` |
| `injection_patterns.py` | `rag/guardrails/injection.py` |
| `pii_patterns.py` | `rag/guardrails/pii.py` |
| `rag_policy/sanitization.py` | `rag/guardrails/sanitization.py` |
| `evidence_assessment.py`, `evidence_queries.py`, `evidence_applicability.py`, `rag_policy/evidence_reading.py` | `rag/evidence/*.py` |

### chat
| Старое | Новое |
|---|---|
| `entities/message.py`, `MessageRole` | `chat/message.py` |
| `entities/conversation.py`, `ConversationListItem` | `chat/conversation.py` |
| `entities/chat_log.py` | `chat/chat_log.py` |
| `conversation_`, `message_`, `chat_log_repository` | `chat/repositories.py` |

### benchmark
| Старое | Новое |
|---|---|
| `benchmark_question.py`, `benchmark_annotations.py` | `benchmark/question.py` |
| `benchmark_run.py` | `benchmark/run.py` |
| `benchmark_sweep.py`, `sweep_status`, `sweep_evaluation_mode`, `benchmark_strategy` | `benchmark/sweep.py` |
| `benchmark_scoring.py` | `benchmark/scoring.py` |
| `benchmark_evaluation.py`, `benchmark_source_hints.py` | `benchmark/evaluation.py` |
| `benchmark_judge.py` | `benchmark/judge.py` |
| `benchmark_selection.py` | `benchmark/selection.py` |
| `benchmark_*_repository.py` | `benchmark/repositories.py` |

### runtime
| Старое | Новое |
|---|---|
| `LLMProvider`, `FileBackend`, `AppStage`, `HealthStatus` | `runtime/enums.py` |
| `background_job.py`, `job_status.py`, `job_type.py` | `runtime/jobs.py` |
| `config_parameter.py`, `config_value_type.py`, `events/config_events.py`, `utils.parse_bool` | `runtime/config.py` |
| `DynamicParam`, `DYNAMIC_PARAMS`, `DYNAMIC_FIELDS_MAP`, `UPLOAD_*` | `runtime/dynamic_params.py` |
| `*_SETTING_KEYS` из `dynamic_config.py` | `runtime/setting_keys.py` |
| `cost_budget.py` | `runtime/cost.py` |
| `background_job_`, `config_parameter_repository` | `runtime/repositories.py` |

---

## 5. Разделение ingestion и rag

### Профили доменов

| Было | Стало |
|---|---|
| один `DomainProfile` | `ingestion.DocumentProfile` и `rag.AnswerPolicy` |

- **`DocumentProfile`:** `key`, `display_name`, `is_versioned`, `config_defaults`, `structural_fingerprint`, `classify_score`, `content_boundaries`, `extract_references`, `extract_effective_date`.
- **`AnswerPolicy`:** `prompt_addendum(breadth, as_of_date)`, `retrieval_fallback_to_full_corpus()`. Реализации по ключу домена: `general`, `legal`, `decree`.
- Тексты промптов уходят из `legal.py` и `decree.py` в `rag/answer_policy.py`.
- Registry нужен только `DocumentProfile`. Для RAG достаточно `policy_for(domain_key)` в `rag/answer_policy.py`, без второго registry. Связь между ними только ключ `DocDomain` из `knowledge`.

### Порты

| Протокол | Куда |
|---|---|
| `ChunkCrudRepository`, `ChunkVersioningRepository`, `ChunkCorpusRepository` | `knowledge/repositories/chunk.py` |
| `ChunkSearchRepository`, `ChunkSearchPageRepository`, `ChunkContextRepository`, `ChunkRetrievalRepository` | `rag/ports.py` |
| combined `ChunkRepository` для UoW | `application/uow/chunks.py` (пересечение knowledge- и rag-портов) |
| `VectorStoreRepository`: `ensure_collection`, `upload_documents`, `delete_*`, `upsert_point`, `set_document_id_*`, `update_*` | `knowledge/index_sync.py` как `VectorIndexWriter` |
| `VectorStoreRepository`: `generate_embeddings`, `get_point_payload` | `rag/ports.py` как `VectorSearch` |
| outbox (entry и три протокола) | `knowledge/index_sync.py` |

Outbox остаётся в `knowledge`: в него пишут и индексация, и правка чанков, и обновление метаданных редакций. Иначе `knowledge` зависел бы от `ingestion`.

---

## 6. Удаляется

- Весь `document_domain_classifier.py`: legacy-эвристика удаляется, используемый `classify_doc_type` с константами переносится в `ingestion/doc_type.py`.
- Реэкспорт `DomainSettingsPort` в `application.ports.domain_settings`.
- Реэкспорт `_CRITICAL_RULES_BLOCK` из `rag_policy/__init__`.
- `LEGACY_POOL_SETTING_KEYS` (уходит в скрипт офлайн-миграции).
- `BenchmarkDataset` (одно значение). Заменяется константой `DEFAULT_DATASET = "main"`.
- `BenchmarkStrategy.SUCCESSIVE_HALVING` (после миграции данных).
- `HealthStatus.OK` и `HEALTHY` сохранить: это разные значения публичного health-контракта (`ok` для компонентов и `/live`, `healthy` для общей готовности).
- `if False:  # TYPE_CHECKING` в `access_control.py`.
- Дубли в протоколах: `ConversationRepository.get` или `get_by_id`; нестримовые методы `ChunkCorpusRepository`.
- Все barrel-`__init__` с `__all__`.

## 7. Переименования

| Было | Стало | Причина |
|---|---|---|
| `ITokenProvider`, `IPasswordHasher` | `TokenProvider`, `PasswordHasher` | единый стиль портов |
| `ChunkSearchResult` | `ChunkRecord` | это общая строка чанка, не только результат поиска |
| `ChatContext` | `RagContext` | это вход RAG, не чата |
| `pdf_dto` | `preview` | используется для PDF, RTF, DOCX |
| `domain_profile` | `ingestion.profiles` | профили определяют разбор документа |
| `DomainProfile` | `DocumentProfile` + `AnswerPolicy` | разделение записи и чтения |
| `rag_policy` | `rag` | политика и есть содержимое контекста |

## 8. Исправления контрактов при переезде

1. `LifecycleStatusMixin`: добавить `is_terminal`, исправить докстринг (пример с `terminal=`/`active=`).
2. `stream_events.py`: объявить `UsageReport` выше `SourcesEvent`.
3. Ввести `NOT_FOUND_ANSWER` в `prompt_blocks.py`, подставить во все блоки, добавить тест, что фраза матчится `NOT_FOUND_PATTERNS`.
4. `cost.py`: заменить `provider != 'deepinfra'` на поле `input_only: bool = False` в `CostRate`.
5. Хранить enum как enum с приведением в `__post_init__`: `BackgroundJob.status`, `BenchmarkSweep.status/strategy/evaluation_mode`, `UserContext.user_kind/user_role`. Убрать повторные `UserKind(ctx.user_kind)` в `access_control`.
6. `Document`: общий curator-блок из `can_be_deleted_by` и `can_edit_chunks` вынести в `in_curator_scope(...)`.
7. `ConversationListItem.creation_date: object` заменить на `datetime`.
8. `deduplicate_docs` и `prioritize_distinct_provisions` аннотировать через `RawDocument` или `Protocol`.
9. Тест на равенство множеств `TARGET_DYNAMIC_SETTING_KEYS` и ключей `DYNAMIC_PARAMS`, либо довести миграцию и убрать дубль.
10. `ruff format` один раз, чтобы убрать смесь кавычек.
11. Отдельный PR: `ChunkCrudRepository.bulk_insert` с десятью параллельными списками заменить списком dataclass'ов.
12. `runtime/dynamic_params.py`: поле `category` сделать enum'ом.

---

## 9. Правила зависимостей (`import-linter`)

Разрешённые рёбра: `chat → rag, identity`; `rag → knowledge, identity`; `ingestion → knowledge`; `runtime → knowledge`; `knowledge → identity`; `benchmark → shared`. Все контексты могут импортировать `shared`.

```ini
[importlinter:contract:layers]
name = Domain contexts
type = layers
layers =
    domain.chat
    domain.rag | domain.ingestion | domain.runtime | domain.benchmark
    domain.knowledge
    domain.identity
    domain.shared

[importlinter:contract:rag-ingestion-independent]
name = RAG and ingestion do not import each other
type = independence
modules =
    domain.rag
    domain.ingestion

[importlinter:contract:domain-pure]
name = Domain has no outer dependencies
type = forbidden
source_modules = domain
forbidden_modules =
    application
    infrastructure
    presentation
    langchain
    sqlalchemy
    qdrant_client
```

Ребро вне списка считается ошибкой структуры, контракт не ослабляется.

---

## 10. Порядок выполнения

1. **Подготовка (один коммит).** Пункты 8.1–8.3, удаление legacy-модуля и его порта/параметров из Application и DI, `ruff format` только затронутого домена. API и S3 используют общий registry-путь; без registry/settings — `general` с warning. Тесты зелёные.
2. **Закрыть блокирующие вопросы** из раздела 11 (помечены «до переноса»).
3. **Codemod на уже установленном `libcst`.** Таблица `(старый_модуль, символ) → (новый_модуль, символ)` на уровне символов, потому что `utils`, `query_results`, `llm_provider`, `dynamic_config`, `domain_profile/protocol` разъезжаются по разным местам. Переписывает импорты в domain, application, infrastructure, presentation, composition, tests и tools. Проверить алиасы, импорты модулей и обращения к реэкспортам; повторный запуск не меняет результат.
4. **Перемещения через `git mv`**, один коммит на контекст: `shared → identity → knowledge → ingestion → rag → chat → benchmark → runtime`. В каждом коммите проходят целевые тесты и `task lint`; общий pytest — после изменений общих импортов и завершения переноса. Перенос символов, изменение поведения и массовое форматирование не смешиваются.
5. **Разделение `DomainProfile` и портов** выполнять вместе с переносом соответствующих потребителей, чтобы не вводить временные `rag ↔ ingestion` зависимости. До переноса определить размещение combined-протокола `ChunkRepository` из UoW на стороне Application, отдельно связать read/write-порты в DI. Не создавать реестр `AnswerPolicy`, если достаточно функции/значения по домену.
6. **Строковые ссылки.** `grep` по `"domain.`, `mock.patch(`, `importlib`, DI-конфигу, логгерам с `__name__`-фильтрами, Alembic, `pyproject`, `Dockerfile`.
7. **Данные в БД.** Значения enum'ов не меняются. Единственная миграция: `successive_halving → optuna_tpe`.
8. **Контрактные правки** 8.4–8.12 отдельными коммитами.
9. **Дополнить существующий `import-linter`**, уже включённый в `task lint` и CI: существующие контракты Application сохранить. Контракты контекстов добавлять по мере появления пакетов, сразу в режиме error. Один layers-контракт не запрещает все рёбра вне списка раздела 9: нужны дополнительные forbidden-контракты с тестами разрешённых и запрещённых прямых зависимостей. Финальная проверка отсутствия старых путей и обратных импортов.

### Уточнения по результатам проверки 20.10.2026

- Ветка: `refactor/domain-contexts`. Исходная staged-правка удаления `APPLICATION_REFACTORING_PLAN.md` принадлежит пользователю и не входит в коммиты рефакторинга.
- Legacy-классификатор был живым: его передавали `application.documents.classification` и `application.ingestion.loader`. По запросу пользователя модуль удалён на шаге 1. Общий registry-путь сохраняется; без registry/settings возвращается `general` с warning (раньше API/S3 вызывали legacy-эвристику). Добавлены проверки отсутствия каждого из двух компонентов, совпадения API/CLI и короткого/структурированного текста с настоящим registry. Порог `document_domain_marker_threshold` удалён из прикладных портов, конструкторов и DI. Пока остаётся неиспользуемое поле конфигурации и запись в списке static-ключей; их удалить согласованно на этапе конфигурации, проверив сохранённые настройки.
- `ChunkCorpusRepository` используется в `infrastructure.worker.cron` для построения BM25. По условию вопроса 11.A.2 порт следует разместить в `knowledge/repositories/chunk.py`; окончательно проверить зависимости его типов до переноса.
- Правило имён из AGENTS.md распространяется на новые API: вспомогательный метод из 8.6 назвать `in_curator_scope`, а не `_in_curator_scope`.
- Совместимость включает не только строковые значения enum: проверить OpenAPI, health API, frontend, кеши и сохранённые настройки. Удалять алиасы/значения только после проверки потребителей; миграцию стратегии сделать до их удаления.
- После каждого этапа исправлять затронутые вызовы, типы, порты и контрактные тесты в уже отрефакторенном Application, не возвращая старую структуру `application.services`.

### Прогресс

- [x] План проверен; поправки процесса и риски внесены. Создана отдельная ветка.
- [x] Шаг 1: подготовка выполнена. `is_terminal`, документация mixin, порядок `UsageReport`, общий `NOT_FOUND_ANSWER` в промптах и ответах runtime, удаление legacy-модуля и согласование Application/DI. `ruff format app/domain` не потребовал массовых изменений. Baseline: **362 passed**, `task lint` пройден. После удаления legacy: **914 passed** в итоговом наборе (Domain, API/S3, RAG/guardrails, ACL, кеш, Application contracts), `task lint` пройден (Ruff, формат, mypy, 18 import-контрактов, deptry, vulture, blank-lines).
- [x] Шаг 2: блокирующие вопросы 11.1–3, 11.9, 11.14, 11.26 закрыты по коду; решения и ограничения проверки данных зафиксированы ниже. Согласованы PageContentType, профили, combined-порт UoW и сохранение health-контракта.
- Проверки шага 2: **155 passed** (guardrails, domain_profiles, domain_registry, text_quality, preview_strategies, закрытие corpus-stream, answer_cache, act_version_timeline), `git diff --check` пройден. Два более широких запуска прерваны при отсутствии прогресса на benchmark-сценарии `test_solid_characterization.py` и `test_retrieval_acl.py`; эти наборы не считаются пройденными. Исходники не менялись; полный pytest и `task lint` на этом этапе не запускались.
- [x] Шаг 3: подготовлен LibCST-codemod `server/tools/domain_imports.py` и таблица `server/tools/domain_moves.json` (441 соответствие на уровне символов, включая текущие реэкспорты). По умолчанию показывает diff; запись только с `--write`, выбор контекста — `--context`. Проверены split-imports, переименования, алиасы и их scope, модульные обращения, сохранение неиспользуемых экспортов, идемпотентность и отказ от частичной записи при star-import. Разделение `DomainProfile`/`VectorStoreRepository` и удаление legacy-значений остаются ручными изменениями шагов 4–8; неоднозначный векторный порт намеренно не получает автоматическую замену. Прикладные parser/splitter-порты уже находятся в `application/ingestion/ports/parser.py`, отсутствующего Domain-модуля в таблице нет. `MLProvider` переносится вместе с остальными provider-enum в `runtime/enums.py`.
- Проверки шага 3: **14 passed**, все семь команд из `task lint` выполнены напрямую и прошли (сам snap-бинарник `task` не запускается в этой среде). Пробный запуск по `app`, `tests`, `tools` без записи завершён: **449 файлов** меняются, результат каждого разбирается Python AST, повторное преобразование не меняет результат. Переносы исходников ещё не применялись.
- [x] Шаг 4, контекст `shared`: выделены `exceptions`, `enums`, `pagination`, `hashing`, `dates`, `stats`; `__init__.py` пустой. Потребители во всех слоях и тестах переведены на конкретные модули, старые файлы base/lifecycle_status/cursor_page удалены. `ContextBudgetExceededError`, `explicit_question_date` и остальные функции `utils` остаются в своих старых модулях до переноса соответствующих контекстов; реэкспорты общих ошибок удалены. Добавлен error-контракт независимости `shared` от всех остальных текущих Domain-пакетов/модулей и тесты разрешённых/запрещённых зависимостей; отдельный тест требует дополнять контракт при появлении контекстов. Точечные Ruff/mypy/blank-lines исключения перенесены вместе с существующим кодом; зависимости и lock не изменились (`uv lock --offline` выполнен).
- Проверки `shared`: baseline **199 passed**; после переноса **237 passed**, расширенный набор **424 passed**, итоговые архитектурные тесты **39 passed** (наборы пересекаются). Все команды `task lint` выполнены напрямую и прошли: Ruff, формат, mypy, **19 import-контрактов**, deptry, vulture, blank-lines; `git diff --check` пройден. Python AST разбирается во всех app/tests/tools; импортов перенесённых shared-символов по старым путям не осталось. Полный pytest запущен, но остановлен без прогресса на `test_api_asgi.py`; дополнительный широкий запуск остановлен на `test_rate_limit_dependency.py`. Эти два запуска не считаются пройденными; полный набор остаётся обязательной проверкой. Индекс `.git` доступен только для чтения: перенос выполнен файловой системой, коммит контекста не создан, исходная staged-правка пользователя сохранена.
- [x] Шаг 4, контекст `identity`: перенесены роли, capabilities, User, ApiKey и ApiKeyClientInfo, группы, UserContext и CuratorScope; объединены репозитории и порты, `ITokenProvider`/`IPasswordHasher` переименованы в `TokenProvider`/`PasswordHasher`. Потребители и скрипты обновлены, старые модули и реэкспорты удалены, новый `__init__.py` пустой. Добавлен error-контракт `identity → shared`, расширены архитектурные тесты и shared-контракт; существующие lint-исключения перенесены, зависимости и lock не изменились (`uv lock --offline`). AST всех 16 перенесённых классов совпадает с исходным, кроме имён двух портов; контрактные изменения 8.5 оставлены отдельному этапу.
- Проверки `identity`: baseline **260 passed**; после переноса **389 passed** (identity, HTTP/auth, Domain, ACL/BM25, curator scope, answer cache, retrieval/ingestion ACL и архитектурные контракты). Полный pytest: **3533 passed, 80 skipped, 11 warnings**; интеграции с отсутствующими внешними сервисами пропущены. Зависание HTTP-тестов воспроизведено до переноса только в sandbox; вне sandbox целевые и полный набор завершились. Все семь команд из `task lint` выполнены напрямую и прошли, включая mypy и **20 import-контрактов**; `git diff --check` пройден. Старых identity-импортов и строковых ссылок в исходниках/скриптах не осталось (историческая таблица codemod сохранена).
- Pre-commit для `identity`: Ruff-хуки охватывают также `server/scripts`, в отличие от `task lint` (`app/tests/tools`). В трёх затронутых скриптах подтверждены **185 прежних lint-диагностик**, новых после замены/сортировки импортов нет; синтаксис проверен. Массовое автоформатирование скриптов отменено. При коммите пропущены только `ruff-check`/`ruff-format`; соответствующие проверки `app/tests/tools` выполнены отдельно, остальные commit-хуки сохраняются.
- [x] Шаг 4, контекст `knowledge`: перенесены Document, Chunk/ChunkRecord/ChunkStats, контекст чанка и курсоры, общие enum, ACL/OwnerMatch, акты и timeline, репозитории документов/редакций, outbox и `VectorIndexWriter`. Новые `__init__.py` пустые; старые модули и реэкспорты перенесённых символов удалены. Combined `ChunkRepository` находится в `application/uow/chunks.py`; read-порты пока остаются в `domain/repositories/chunk_repository.py` и `vector_store_repository.py` до переноса `rag`, без реэкспортов knowledge-типов. Writer/search-привязки DI возвращают тот же Qdrant-адаптер, ingestion и outbox получают writer. Удалены три неиспользуемых нестримовых corpus-метода из порта, адаптеров и фейка; ACL-stream и закрытие результата сохранены. В ACL заменён `if False` на `TYPE_CHECKING`, вспомогательные функции названы без начального `_`; правила доступа не менялись. Добавлен error-контракт `knowledge → identity, shared`, расширены архитектурные тесты и ограничения shared/identity. Контрактные изменения 8.5–8.8 остаются отдельным этапом.
- Проверки `knowledge`: baseline **242 passed**; после переноса **257 passed**, дополнительный набор ACL/RAG/Application/codemod/corpus **252 passed**. Полный pytest: **3548 passed, 80 skipped, 11 warnings**; интеграции с отсутствующими внешними сервисами пропущены. Повторное зависание retrieval ACL в sandbox устранено запуском вне sandbox. Все семь команд `task lint` выполнены напрямую и прошли: blank-lines, Ruff, формат, mypy, **21 import-контракт**, deptry, vulture. Проверены AST **842** Python-файлов и отсутствие старых импортов перенесённых символов; AST **47** перенесённых классов/функций совпадает с исходным после переименований. `git diff --check` пройден; `uv lock --offline` не изменил lock. Существующие lint-исключения перенесены вместе с кодом. Коммит не создавался; staged-удаление пользовательского плана сохранено.
- [x] Шаг 4, контекст `ingestion`: перенесены RawDocument, content splitter, типы/протокол/registry и реализации профилей, DomainSettingsPort, извлечение номера акта, стратегия version/replace, preview/quality DTO, text quality и реестр ingestion. `DomainProfile` разделён: `DocumentProfile` содержит только классификацию и разбор; промпты и temporal stamp перенесены в `rag/answer_policy.py` с `policy_for(domain_key)`, без второго registry. Composition передаёт в RagService снимок ключей зарегистрированных доменов вместо профилей; сохранены отсутствие addendum без реестра/для неизвестного домена, оба breadth-режима, дата и состав `domain_prompts` в ключе кеша. Обновлены Application, Infrastructure, Composition, тесты и offline replay-скрипт; старые модули и реэкспорты удалены, новые `__init__.py` пустые. `domain_profile/date_parsing.py` с `explicit_question_date` остаётся до этапа `rag`. Parser/splitter-порты сохраняются в существующем `application/ingestion/ports/parser.py`. Добавлены error-контракты `ingestion → knowledge, shared` (прямые зависимости; транзитивный путь knowledge → identity разрешён) и RAG без ingestion, расширены архитектурные тесты. Правила ACL, retrieval fallback и индексации не менялись.
- Проверки `ingestion`: baseline **174 passed**; после переноса **285 passed** (включая проверки точных промптов, даты, регистрации доменов и scope ключа кеша), расширенный набор **271 passed**. Полный pytest: **3572 passed, 80 skipped, 11 warnings**; интеграции без внешних сервисов/явного opt-in пропущены. Запуск расширенного набора в sandbox остановлен без прогресса на обработчике документов; повторный расширенный и полный запуски вне sandbox завершились. Все семь команд `task lint` выполнены напрямую через установленную `.venv` и прошли: blank-lines, Ruff, формат, mypy (**778** файлов), **23 import-контракта**, deptry, vulture. AST **844** Python-файлов app/tests/tools/scripts разбирается; реализация **32** перенесённых классов/функций совпадает с baseline после исключения разделённых prompt-методов, обновления docstring и переименований `DomainProfile`/`get_setting`. Старых импортов перенесённых символов и строковых ссылок в app/tests/scripts не осталось; историческая таблица codemod сохранена. Существующие lint-исключения перенесены вместе с кодом; `uv lock --offline` не изменил lock. `git diff --check` пройден. Коммит не создавался; исходное staged-удаление пользовательского плана сохранено.
- [ ] Остальная часть шагов 4–9: переносы, разделение портов/профилей, контракты и финальные проверки. Следующий контекст — `rag`.

### Решения шага 2 (20.10.2026)

- **11.1 — PII остаётся в `rag/guardrails/pii.py`.** Единственный импорт `ALL_PATTERNS` — `infrastructure/ml/guardrails/guardrails.py`. Через `PIIRedactorAdapter` детектор используется в `application/chat/service.py` (вопрос), `rag_steps.py` (ответ) и `rag_cache.py` (кешированный ответ); пути индексации его не вызывают.
- **11.2 и 11.26 — корпус в `knowledge`, только ACL-stream API.** Единственный рабочий потребитель — `infrastructure/worker/cron.py::cron_bm25_rebuild`, вызывающий `iter_all_contents_with_acl`. Возвращаемый tuple содержит только текст, visibility, owner_id, group_id; зависимости от RAG нет. Три нестримовых метода не имеют потребителей: удалить их из протокола, SQLAlchemy-фасада, `chunk_corpus.py` и фейков при переносе. Сохранить закрытие результата в `finally` и тест `test_solid_characterization.py`.
- **11.3 — `PageContentType` переносится в `knowledge/enums.py`.** Тип сохранён в metadata чанков и читается RAG-модулями `field_evidence`, `linked_context`, `applicable_context`, `rag_postprocess`, `rag_formatting`, а также ingestion и preview. Размещение в `ingestion/preview.py` создало бы запрещённое ребро RAG → ingestion. `PreviewUnitKind`, dry-run DTO, `PDFQualityReport` и `text_quality` остаются в ingestion: их потребители — обработка/предпросмотр документов, не retrieval.
- **11.9 — потребители ingestion-половины:** `application/indexing/classification.py` (`classify`); `application/ingestion/profiles.py` и его вызывающие `loader.py`/`sync.py` (разбор и версия); `application/documents/conflicts.py`/`versioning.py` (`is_versioned`); `infrastructure/ml/langchain_document_parser.py` (границы, ссылки, fingerprint); `infrastructure/ml/preview/rtf_preview_strategy.py` (структура decree); `infrastructure/ml/config/provider_configuration.py` и `infrastructure/initialization.py` (`all/config_defaults`). `documents/processor.py`, `classification.py`, `persistence.py`, `indexing/pipeline.py` передают registry этим потребителям. Composition и `infrastructure/domain_profile/__init__.py` собирают/регистрируют профили; CLI bootstrap использует `infrastructure.initialization`.
- **11.9–11.11 — потребители RAG-половины:** `infrastructure/ml/rag/helpers.py::domain_prompt_addendum` (`get/prompt_addendum`) и `rag_steps.py::step_check_cache` (`all/prompt_addendum`); registry передаётся через `rag_service.py`. Заменить их на `policy_for(domain_key)`; при построении ключа кеша сохранить набор доменов и оба prompt-addendum (narrow/broad), дату и поведение отсутствующего/неизвестного домена. Общего бизнес-потребителя обеих половин не найдено. `versioned_prompt_date_stamp` — чистая RAG-функция; temporal-правила legal/decree заданы в их AnswerPolicy и не требуют чтения `DocumentProfile.is_versioned`. `retrieval_fallback_to_full_corpus` сейчас объявлен профилями, но не вызывается: реальный legal fallback находится в `helpers.py::run_retrieval`; при переносе не менять его поведение.
- **11.12 и 11.29 — разделить типы, сохранить одну реализацию и транзакцию.** `QdrantVectorStoreRepository` продолжает реализовывать оба порта; Composition предоставляет отдельные типизированные привязки writer/search к тому же экземпляру. `application/ingestion/batch.py` получает writer. Combined `ChunkRepository` перенести в `application/uow/chunks.py`: этот пакет уже владеет UoW и может импортировать оба Domain-контекста без изменения существующего контракта Application. `ChunkRetrievalRepository` остаётся read-портом в `rag/ports.py`; `UnitOfWorkRepositories.chunks` сохраняет один transactional facade.
- **11.13 — `SENTENCE_UNIT_KIND` в `knowledge/chunk.py`.** Splitter записывает структурные единицы, а `domain/services/retrieval_diversity.py` интерпретирует `sentence` в section-пути; это общий контракт метаданных, а не внутренняя настройка splitter.
- **11.14 — Python-пути не являются форматом сохраняемых данных в проверенных адаптерах.** Answer cache использует JSON + gzip/base64 (`infrastructure/ml/answer_cache.py`), BM25 — JSON (`bm25/persistence.py`, `BM25Index.to_dict/from_dict`), Qdrant — `page_content/metadata` (`repositories/vector/qdrant_ops.py`), outbox — payload и JSON уведомление (`repositories/config/sqlalchemy_vector_outbox_repository.py`). Поиск по app/tools/scripts не выявил pickle или сериализации доменных классов по Python-пути. Alembic динамически импортирует только `infrastructure.database.models`; logging использует именованные логгеры `default/detailed/actions`, без фильтра по старым Domain-пакетам. Обновить mypy-строки модулей в `server/pyproject.toml` и два строковых patch-target в `tests/test_retrieval_acl.py` (реэкспорты `rag_policy`). Миграция `l2c3d4e5f6g7_add_sweep_evaluation_mode.py` содержит обычные Domain-импорты: включить migrations в codemod. В Dockerfile и alembic.ini старых Domain-путей нет. Содержимое работающих БД/Redis/Qdrant не проверялось; вывод относится к формату записи в коде.
- **11.15–11.18 — ограничения совместимости.** Наличие `successive_halving` в работающей БД не проверялось: перед удалением enum нужна безусловная data migration, безопасная при отсутствии таких строк; учесть также публичную схему `SweepCreateRequest` и strategy factory. `HealthStatus.OK="ok"` и `HEALTHY="healthy"` не дубли: сохранить оба и проверку `/ready` в `presentation/api/routes/health.py`. `BenchmarkDataset` задаёт только default `main`, тогда как API принимает `dataset: str`; замена константой не должна запрещать другие наборы. `ChunkSearchResult` и `ChatContext` не используются непосредственно как поля Pydantic-схем в presentation; переименование проверить тестами API после переноса.

---

## 11. Открытые вопросы

Пометка **[до переноса]** означает, что ответ нужен до начала соответствующего шага, иначе придётся переделывать.
Ответы на 1–3, 9–14, 26, 29 и ограничения для 15–18 зафиксированы в «Решениях шага 2» выше; исходные вопросы ниже сохранены для трассировки. По 36: `register_all_profiles` остаётся инфраструктурной регистрацией только DocumentProfile; отдельного registry AnswerPolicy не будет.

### A. Границы контекстов

1. **[до переноса] Где применяется `pii.py`?** Если редактирование ПДн выполняется только над запросом и ответом, файл остаётся в `rag/guardrails`. Если ещё и при индексации, он переезжает в `shared`. Проверить: `grep -r "pii_patterns\|ALL_PATTERNS"`.
2. **[до переноса] Кто строит BM25-индекс?** Если `ChunkCorpusRepository` вызывает ingestion-воркер или планировщик, порт уходит в `knowledge`, а не в `rag/ports.py`.
3. **[до переноса] Где используется `text_quality` и `preview`?** Предполагается, что только в ingestion (dry-run предпросмотр). Если их читает ещё и rag или presentation через rag, нужно пересмотреть владельца.
4. **Нужен ли `knowledge → identity`?** `access_control` импортирует `UserContext` и роли. Если `UserContext` логичнее считать частью `knowledge`, ребро исчезнет, но `identity` потеряет снимок прав. Решить один раз и зафиксировать.
5. **Откуда `chat → rag`?** После переноса `RagContext` в `rag` непонятно, зачем `chat` нужен импорт `rag`. Если импорта нет, убрать ребро из контракта.
6. **`runtime → knowledge`.** Единственная причина: `DocDomain` в `DynamicParam.domain_key`. Допустимо ли заменить строковым литералом и убрать ребро?
7. **Где лежит `ContextBudgetExceededError`?** В плане он в `rag/retrieval.py`. Проверить, не ловят ли его application или presentation вне rag-пути; если да, решить, не оставить ли его в `shared/exceptions.py`.
8. **Правильное ли место у `decide_resolution_strategy`?** Предполагается, что решение принимает только ingestion. Если `DocumentService` в application вызывает его вне пайплайна загрузки, это не влияет на структуру, но важно для границ.

### B. Разделение профилей и портов

9. **[до переноса] Кто потребляет `DomainProfileRegistry`?** Перечислить все места, где из реестра берётся профиль, и для каждого определить, какая половина (`DocumentProfile` или `AnswerPolicy`) там нужна. Если есть потребитель обеих половин одновременно, это сигнал о смешении в application.
10. **Нужны ли `AnswerPolicy` отдельные экземпляры на каждый домен или достаточно функции `policy_for(domain_key)`?** Протокол оправдан, только если реализации будут расти. Сейчас у `general` оба метода возвращают `None`/`False`.
11. **`versioned_prompt_date_stamp` и `is_versioned`.** `is_versioned` нужен ingestion (version/replace), а rag использует его косвенно для даты в промпте. Откуда `AnswerPolicy` узнаёт, что домен версионируемый: из собственной константы, из `knowledge`, из `DocumentProfile`? Последнее нарушит независимость.
12. **Как делится `VectorStoreRepository` в инфраструктуре?** Одна реализация может реализовывать оба протокола (`VectorIndexWriter` и `VectorSearch`), но DI должен связывать их по отдельности. Проверить, что тестовые фейки не ломаются.
13. **`SENTENCE_UNIT_KIND`.** Он нужен и `content_splitter`, и `retrieval_diversity`. Убедиться, что в `knowledge/chunk.py` он действительно смысловая часть метаданных чанка, а не техническая деталь разбиения.

### C. Данные и совместимость

14. **[до переноса] Сохраняются ли где-то строковые пути модулей?** Pickle, JSON-метаданные чанков, записи в БД, кэш Redis, `config_parameters`, Alembic-миграции, логи с `__name__`-фильтрами. Любое такое место сломается при переносе без шима.
15. **Есть ли в БД значения `successive_halving`?** Если да, нужна миграция `UPDATE … SET strategy='optuna_tpe'` до удаления значения. Если нет, удалять сразу.
16. **`HealthStatus.OK` и `HEALTHY`:** кто из потребителей (API, фронтенд, мониторинг) завязан на какое значение? Что оставить?
17. **Что значит `BenchmarkDataset` для API и БД?** Колонка `dataset` хранит строку; фронтенд может отправлять другие значения. Проверить, действительно ли поддерживается только `"main"`.
18. **Совместимость API-схем.** Переименования `ChunkSearchResult → ChunkRecord`, `ChatContext → RagContext` затрагивают Pydantic-схемы и OpenAPI? Если схемы импортируют доменные типы напрямую, имена в OpenAPI могут измениться.

### D. Корпус-специфичная логика

19. **Остаётся ли один корпус (ЭТТН/ЭТН)?** Пока да: `rag/evidence` изолирован и помечен «corpus-specific». Если планируются другие корпуса, нужен механизм подключения (профиль или плагин), а не файл в `rag`. Решение нужно сейчас, чтобы не городить абстракцию заранее.
20. **Промпты тоже заточены под корпус.** `_CRITICAL_RULES_BLOCK` и `_CONDITIONAL_RULES_BLOCK` содержат примеры ЭТТН/ЭТН/ИМН, «золотые изделия». Выносить ли их в отдельный блок, включаемый по конфигу, или принять как есть?
21. **`evidence_queries.TABLE_MESSAGE_ALIASES`** (`BLRDLN`, `BLRWBL`): данные в коде. Должны ли они жить в конфиге или остаться константой?

### E. Конфигурация

22. **Три модели конфигурации.** `ConfigDefault` (профиль), `DynamicParam` (реестр), `ConfigParameter` (сущность в БД). Какая из них источник правды? Предложение: `DynamicParam` и `ConfigDefault` приводятся к одному типу декларации, `ConfigParameter` остаётся сущностью хранения.
23. **Миграция `DYNAMIC_PARAMS` → `TARGET_DYNAMIC_SETTING_KEYS`.** Довести до конца (удалить `DYNAMIC_PARAMS`) или оставить две структуры с тестом на равенство? В комментарии «legacy registry until each migration stage is ready»; нужен план и срок.
24. **`category` как enum.** Какие значения: `corpus`, `cost`, `ingestion`, `jobs`, `llm`, `openrouter`, `ml`, `rag`, `hybrid`, `reranker`, `toggles`, `ocr`, `rate_limit`? Подтвердить или объединить.
25. **`parse_bool` в `runtime/config.py`.** Используется ли он вне config (например, в presentation при разборе параметров)? Если да, он переезжает в `shared`.

### F. Репозитории и контракты

26. **[до переноса] Какой API `ChunkCorpusRepository` оставить?** Сейчас четыре метода (`iter_all_contents_with_acl`, `get_all_contents_batches_with_acl`, `get_all_contents`, `get_all_contents_batches`). Предложение: оставить только стримовый `iter_all_contents_with_acl`. Проверить потребителей и тесты.
27. **`ConversationRepository.get` или `get_by_id`:** какой оставить? Отличается ли их поведение (например, загрузка сообщений)?
28. **`bulk_insert` с десятью параллельными списками.** Заменить на список dataclass'ов в этом рефакторинге или отдельным PR после? Предложение: отдельным, потому что меняется схема вызовов в ingestion.
29. **Нужен ли `ChunkRepository` как combined-facade?** `UnitOfWork` использует его как единый атрибут. После разделения на write (`knowledge`) и read (`rag`) UoW должен либо экспортировать оба протокола, либо тип атрибута станет пересечением на стороне infrastructure.
30. **`IngestionRegistryEntry`:** остаётся рядом с репозиторием (`ingestion/repositories.py`) или выносится в отдельный модуль? Предложение: рядом, пока он один.

### G. Типизация и стиль

31. **`UserContext.user_kind: str` → `UserKind`.** Затрагивает все места сборки `UserContext` (presentation, infrastructure) и сериализацию (кэш-ключи, логи). Проверить, что значения не передаются как строки в Qdrant-фильтры.
32. **`deduplicate_docs` и `prioritize_distinct_provisions`:** типизировать через `RawDocument` или `Protocol` с `page_content` и `metadata`? Работают они с LangChain-документами и с `(doc, score)` кортежами; `Protocol` точнее.
33. **`BackgroundJob.status` как `BackgroundJobStatus`.** Репозиторий сейчас возвращает строки; проверить маппинг ORM → entity.
34. **Стиль кавычек.** Принять двойные кавычки для всего домена и закрепить в конфиге `ruff`? Подтвердить, что нет файлов, где одинарные принципиальны.

### H. Инструменты и процесс

35. **Достаточно ли покрытие тестами для безопасного переноса?** Если у части модулей (`access_control`, `act_version_timeline`, `content_splitter`) нет тестов, добавить их до codemod'а, иначе поломки импортов найдутся только в рантайме.
36. **Где остаётся `register_all_profiles`?** Предполагается в `infrastructure.domain_profile`. После разделения он регистрирует `DocumentProfile` в registry ingestion и `AnswerPolicy` в registry rag; переименовать ли модуль?
37. **Один большой PR или несколько?** Предложение: один PR, коммит на контекст, чтобы ловить падения по месту. Согласовать с процессом ревью и заморозкой фич-веток.
38. **Что делать с открытыми ветками?** Любой перенос файлов вызывает конфликты. Нужна дата заморозки или сначала слияние висящих PR.
39. **Нужны ли `hashing.py` и `stats.py` в `shared`?** Если после `grep` окажется, что `content_hash` импортируют только `knowledge`/`rag`, а `percentile` только `benchmark`, перенести их к единственному потребителю.
40. **`import-linter` в режиме `warn` на время миграции или сразу `error`?** Предложение: `error` с первого коммита, который вводит конфиг, чтобы не накапливать нарушения.
