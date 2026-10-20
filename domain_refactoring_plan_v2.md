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
│   │                                 DOCUMENT_SLOT_STATUSES, DocumentVisibility, SourceType, SearchMode)
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
│       ├── chunk.py                 (ChunkCrudRepository, ChunkVersioningRepository)
│       └── acts.py                  (ActVersion- и RegulatoryAct-репозитории)
│
├── ingestion/                       (запись: разбор файла → чанки → индекс)
│   ├── raw_document.py              (разобранный текст до разбиения)
│   ├── ports.py                     (DocumentParser, DocumentSplitter, DomainSettingsPort)
│   ├── content_splitter.py          (рекурсивное разбиение по структурным границам)
│   ├── doc_type.py                  (эвристика типа документа: приказ, договор и т.п.)
│   ├── references.py                (extract_act_number из ReferenceMatch)
│   ├── versioning.py                (решение version или replace при конфликте)
│   ├── preview.py                   (типы страниц, DTO dry-run, качество PDF)
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
│   ├── ports.py                     (ChunkSearch/Page/Context/Corpus, VectorSearch: чтение)
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
| `doc_domain`, `document_status`, `visibility`, `source_type`, `search_mode` | `knowledge/enums.py` |
| `entities/document.py` | `knowledge/document.py` |
| `entities/chunk.py`, `ChunkSearchResult`, `ChunkStats`, `chunk_context.py`, `utils.encode/decode_cursor`, `SENTENCE_UNIT_KIND` | `knowledge/chunk.py` |
| `services/access_control.py`, `owner_match.py` | `knowledge/access_control.py` |
| `entities/vector_outbox_entry.py`, `vector_outbox_repository.py`, write-часть `vector_store_repository.py` | `knowledge/index_sync.py` |
| `entities/regulatory_act.py`, `act_version.py` | `knowledge/acts/models.py` |
| `entities/act_version_timeline.py` | `knowledge/acts/timeline.py` |
| `same_document_scope`, `document_scope_key` | `knowledge/acts/policy.py` |
| `document_repository.py` | `knowledge/repositories/document.py` |
| `ChunkCrudRepository`, `ChunkVersioningRepository` | `knowledge/repositories/chunk.py` |
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
| `page_content_type`, `preview_unit_kind`, `pdf_dto`, `pdf_quality_report` | `ingestion/preview.py` |
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
| read-часть `chunk_repository.py` и `vector_store_repository.py` | `rag/ports.py` |
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
- У каждой стороны свой registry. Связь между ними только строковый ключ `DocDomain` из `knowledge`.

### Порты

| Протокол | Куда |
|---|---|
| `ChunkCrudRepository`, `ChunkVersioningRepository` | `knowledge/repositories/chunk.py` |
| `ChunkSearchRepository`, `ChunkSearchPageRepository`, `ChunkContextRepository`, `ChunkCorpusRepository` | `rag/ports.py` |
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
- Дубль `HealthStatus.OK` и `HEALTHY`: оставить один.
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
6. `Document`: общий curator-блок из `can_be_deleted_by` и `can_edit_chunks` вынести в `_in_curator_scope(...)`.
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
- [ ] Шаг 2: закрыть блокирующие вопросы перед переносом.
- [ ] Шаги 3–9: codemod, переносы, контракты и финальные проверки.

---

## 11. Открытые вопросы

Пометка **[до переноса]** означает, что ответ нужен до начала соответствующего шага, иначе придётся переделывать.

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
