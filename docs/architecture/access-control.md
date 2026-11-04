# Модель доступа к документам

Документ описывает действующие правила чтения документов и их применение к RAG.
Каноническая политика находится в
[`domain/knowledge/access_control.py`](../../server/app/domain/knowledge/access_control.py);
адаптеры должны переводить её в формат хранилища, а не повторять правила самостоятельно.

## Контекст доступа

[`UserContext`](../../server/app/domain/identity/user_context.py) — неизменяемый снимок
пользователя на время запроса: `user_id`, `user_kind`, `user_role`, `group_ids` и
назначения куратора (`managed_client_ids`, `managed_internal_ids`, `managed_group_ids`).
`ChatService.prepare_chat()` собирает его через `UserContextFactory` и помещает в
[`RagContext`](../../server/app/domain/rag/context.py). Kind и role отвечают на разные
вопросы: kind разделяет внутренний и клиентский контуры, role добавляет полномочия
USER, CURATOR или ADMIN.

## Политика видимости

Каждый элемент [`VisibilityCondition`](../../server/app/domain/knowledge/access_control.py)
описывает AND-условие: видимость и, при необходимости, владелец или группа. Список
условий образует OR. `get_visibility_conditions()` строит это представление, которое
используют `can_view_document()` для просмотра и `is_in_search_scope()` для RAG.

| Kind и роль | `INTERNAL_PUBLIC` | `INTERNAL_GROUP` | `INTERNAL_PRIVATE` | `CLIENT_PRIVATE` |
|---|---|---|---|---|
| CLIENT, любая роль | — | — | — | Только документы своего владельца |
| INTERNAL / USER | Все | Группы пользователя | Свои | Нет |
| INTERNAL / CURATOR | Все | Группы пользователя и назначенные куратору | Свои и назначенных внутренних пользователей | Назначенных клиентов |
| INTERNAL / ADMIN | Все | Все внутренние группы | Все внутренние документы | Все при просмотре/списке; не включаются в RAG-поиск |

Для `INTERNAL_GROUP` куратору доступны собственные группы и управляемые группы. Его
доступ к private-документам ограничен владельцами из назначений. Для `CLIENT` политика
сразу возвращает только условие «`CLIENT_PRIVATE` и владелец — текущий пользователь»;
роль не добавляет клиенту внутренние документы.

### Список и поиск — разные области

`can_view_document()` строит условия с `for_list=True`; `is_in_search_scope()` и
Qdrant-фильтр — с `for_list=False`. Поэтому ADMIN может просматривать
`CLIENT_PRIVATE` документы, включая список, но они не становятся контекстом ответа
только из-за роли администратора. Проверки изменения также отдельны: доступность для
чтения не заменяет проверки capability, владельца и полномочий для операции.

При назначении видимости `validate_document_visibility()` проверяет допустимость
видимости для kind и capabilities. Например, `INTERNAL_PUBLIC` публикация ограничена
ADMIN, а публикация в группу требует `group_id` и доступа к этой группе.

## Сквозное применение

1. API аутентифицирует пользователя. `ChatService` загружает группы и назначения,
   формирует `UserContext` и передаёт его через `RagContext`.
2. `RagOperations.retrieval_work()` вычисляет условия поиска через
   `get_visibility_conditions(for_list=False)`, строит Qdrant ACL-фильтр и добавляет
   временной фильтр редакций для `as_of_date`.
3. Dense и sparse поиск Qdrant используют один обязательный фильтр. Разрешение sparse
   хешей тоже требует ACL-фильтр. Отсутствующий фильтр приводит к ошибке, а пустой набор
   условий — к пустому результату.
4. Точный поиск чанков получает `UserContext` и дату через
   [`ChunkRetrievalRepository`](../../server/app/domain/rag/ports.py). После сбора
   кандидатов `DocumentAccessService` повторно проверяет ID документов по основной БД:
   только записи, для которых `is_in_search_scope()` истинно, идут дальше. Та же
   авторитетная проверка применяется после reranking и к соседним фрагментам.
5. Извлечённые, связанные и scope-фрагменты добавляются в контекст через методы, которым
   передаётся тот же пользователь; результаты повторно проходят проверку доступа.
   Источники ответа строятся только из выбранного контекста или уже проверенных
   retrieval-документов.
6. Кэш включает хеш области видимости с kind, role, ID пользователя, группами и scope
   куратора. При чтении кэш дополнительно требует полный список зависимых документов и
   повторно проверяет их ID по основной БД. Старые или неполные записи кэша не проходят
   эту проверку.

### Адаптеры и область их ответственности

| Путь | Как применяется политика |
|---|---|
| Domain | `get_visibility_conditions()`, `can_view_document()`, `is_in_search_scope()` |
| Qdrant | [`build_qdrant_filter()`](../../server/app/infrastructure/repositories/vector/acl.py) переводит условия Domain в `should`-ветви; `build_search_filter()` добавляет временной диапазон |
| Основная БД | [`DocumentAccessService`](../../server/app/application/documents/access.py) читает документы с master-соединения и применяет `is_in_search_scope()` |
| Локальный BM25 | [`BM25Index.search_with_hashes()`](../../server/app/infrastructure/bm25/bm25_index.py) фильтрует кандидатов по видимости, владельцу и группе до вычисления score |
| Answer cache | [`CacheKeyBuilder`](../../server/app/infrastructure/ml/cache/keys.py) разделяет ключи по scope; [`cache_is_accessible()`](../../server/app/infrastructure/ml/rag/document_access.py) fail-closed проверяет зависимости ответа |

`BM25Index` — отдельный snapshot-индекс. В основном текущем RAG retrieval-пути
`RetrievalPipeline` запускает dense и sparse поиск через Qdrant; при этом его фильтр
ACL обязателен для обоих механизмов. Не следует считать наличие `BM25Index` заменой
ACL-фильтра Qdrant.

## Согласованность при индексации

В момент подготовки чанков [`enrich_chunks_metadata()`](../../server/app/application/indexing/pipeline.py)
добавляет видимость, владельца, группу и domain metadata. Сохранение документа,
чанков и операций vector outbox выполняется в одной транзакции через
[`persist_document_result()`](../../server/app/application/documents/persistence.py).
Внешняя индексация применяет outbox после commit. При изменении модели доступа нужно
сохранять одинаковые ACL-метаданные в основной записи, Qdrant payload и snapshot BM25;
неполный ACL не должен превращаться в разрешение.

## Проверки для изменений ACL

Сохраняйте эквивалентность `is_in_search_scope()` с ACL-частью Qdrant-фильтра и
`BM25Index.doc_matches_acl()`. Проверяйте разрешённый и запрещённый сценарии для
USER, ADMIN, CURATOR и CLIENT, включая отличия списка от поиска и повторную проверку
ID источника. Целевые наборы: [`test_acl.py`](../../server/tests/test_acl.py),
[`test_bm25_acl.py`](../../server/tests/test_bm25_acl.py),
[`test_curator_role.py`](../../server/tests/test_curator_role.py),
[`test_retrieval_acl.py`](../../server/tests/test_retrieval_acl.py),
[`test_ingestion_acl_consistency.py`](../../server/tests/test_ingestion_acl_consistency.py)
и [`test_answer_cache.py`](../../server/tests/test_answer_cache.py).
