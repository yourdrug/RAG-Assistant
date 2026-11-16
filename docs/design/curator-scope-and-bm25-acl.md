# Design: Curator scope в ChatContext и стратегия ACL для BM25

- **Статус**: Proposed
- **Дата**: 2026-09-13
- **Скоуп**: ACL-контракт RAG-пайплайна (chat → retrieval → answer cache)
- **Затрагиваемые слои**: `domain/value_objects`, `application/services/chat_service.py`,
  `infrastructure/ml/rag_service.py`, `infrastructure/bm25/**`, `infrastructure/ml/answer_cache.py`

---

## 1. Curator scope — структура `managed_*_ids` в ChatContext

### 1.1 Текущее состояние (проблема)

Путь данных сегодня:

```
get_current_user() → UserContext.build()          [managed_*_ids ЗАГРУЖАЮТСЯ]
  → ChatService._prepare_chat()                   [managed_*_ids ТЕРЯЮТСЯ]
    → ChatContext(user_id, kind, role, user_group_ids, ...)
      → RagService._init_state()
        → build_qdrant_filter(user, ctx.user_group_ids)   [managed_*_ids = None]
```

Факты из кода:

| Факт | Место |
|---|---|
| `UserContext` содержит `managed_client_ids`, `managed_internal_ids`, `managed_group_ids`, загружаемые из БД только для CURATOR | `domain/value_objects/user_context.py:17-19, 32-36` |
| `ChatContext` этих полей **не имеет** — только `user_group_ids` | `domain/value_objects/chat_context.py:23-29` |
| При сборке `ChatContext` managed-scope отбрасывается | `application/services/chat_service.py:90-98` |
| `build_qdrant_filter` уже принимает managed-ids, но вызывается без них | `infrastructure/ml/rag_service.py:128`, `infrastructure/repositories/vector/acl.py:28-33` |

**Следствие — расколотый ACL-контракт**: в document listing / CRUD куратор видит
документы managed-пользователей (`get_visibility_conditions` вызывается с managed-ids
из `UserContext`), а в RAG-поиске (`for_list=False`, тот же
`get_visibility_conditions`, но без managed-ids) — **не видит**. Один и тот же
"single source of truth" даёт разные ответы в зависимости от пути вызова.

**Вторая проблема — answer cache**: `compute_visibility_scope_hash` =
`kind:user_id:sorted(group_ids)` (`infrastructure/ml/answer_cache.py:23-26`). Роль и
managed-scope в хэш не входят. Сценарий утечки:

1. Куратор C (managed_internal_ids=[20,30]) задаёт вопрос → ответ кэшируется с
   цитатами из private-документов пользователей 20 и 30. TTL = 7 дней
   (`CACHE_TTL_SECONDS`, answer_cache.py:19).
2. C снимают с кураторства (role → USER) или убирают assignments.
3. `user_id` и `group_ids` не изменились → `vis_hash` тот же → **cache hit**:
   бывший куратор получает ответ, ссылающийся на документы, к которым у него больше
   нет доступа.

Направление "повышение" (USER → CURATOR) утечки не создаёт (старые записи кэша
вычислены в более узком скоупе — fail-closed), но направление "понижение" —
утечка. Значит **scope hash обязан включать весь ретривал-скоуп**, а не только
identity.

### 1.2 Требования и инварианты

1. **Единый ACL-контракт**: `is_in_search_scope(doc, ctx)` и фильтр, который строит
   `RagService`, должны давать **одно и то же множество документов** для одного
   пользователя. Единственная допустимая разница — задокументированный
   `for_list=False` бонус ADMIN'а по `CLIENT_PRIVATE`.
2. **managed-scope — server-side данные**: берутся из БД (`UserContext.build` →
   `assignment_repository`), никогда из JWT и никогда из тела запроса. Инвалидация
   скоупа при смене assignments — на стороне БД + кэш-ключа, не на стороне токена.
3. **Иммутабельность**: `ChatContext` остаётся frozen; managed-scope фиксируется на
   момент запроса (snapshot-семантика, как у `UserContext`).
4. **Обратная совместимость порта**: `ChatRAGPort.stream(question, history, ctx)`
   (`application/ports/chat_rag_port.py:21-26`) не меняется — меняется только
   структура `ctx`. Fakes и benchmark, конструирующие `ChatContext`
   (`infrastructure/benchmark/runner.py:132`), должны остаться рабочими.
5. **Пустой скоуп ≠ отсутствие скоупа не обязано различаться** — текущий
   `get_visibility_conditions` трактует `None` и `[]` одинаково. Сохраняем.

### 1.3 Варианты структуры

#### Вариант A — три плоских поля (зеркало UserContext)

```python
@dataclass(frozen=True)
class ChatContext:
    ...
    managed_client_ids: list[int] = field(default_factory=list)
    managed_internal_ids: list[int] = field(default_factory=list)
    managed_group_ids: list[int] = field(default_factory=list)
```

- ✅ Минимальный diff; `build_qdrant_filter` уже принимает такую сигнатуру.
- ❌ Каждый некураторский `ChatContext` тащит три пустых списка.
- ❌ При расширении кураторских полномочий (например, `managed_domains`) — снова
  размазываем поля по сигнатурам: `rag_service.py:128` превращается в
  `build_qdrant_filter(user, ctx.user_group_ids, ctx.managed_client_ids, ...)`,
  и каждое новое поле требует правок во всех точках проброса. Именно так
  `get_visibility_conditions` уже обросла тремя positional-параметрами
  (access_control.py:146-148) — повторять паттерн не хотим.

#### Вариант B — вложенный value object `CuratorScope` (рекомендуется)

```python
# domain/value_objects/curator_scope.py
@dataclass(frozen=True)
class CuratorScope:
    """Snapshot полномочий куратора на момент запроса.

    Загружается из assignment-репозитория в UserContext.build(),
    фиксируется в ChatContext и не мутирует на протяжении запроса.
    """
    managed_client_ids: tuple[int, ...] = ()
    managed_internal_ids: tuple[int, ...] = ()
    managed_group_ids: tuple[int, ...] = ()

    def is_empty(self) -> bool:
        return not (self.managed_client_ids or self.managed_internal_ids or self.managed_group_ids)


@dataclass(frozen=True)
class ChatContext:
    user_id: int
    user_kind: str
    user_role: str = "user"
    user_group_ids: list[int] = field(default_factory=list)
    curator_scope: CuratorScope | None = None   # None = не куратор
    depth: str | None = None
    summary: str | None = None
    as_of_date: date | None = None
```

- ✅ `None` явно кодирует "роль не куратор" — семантика, а не конвенция пустых списков.
- ✅ Один параметр пробрасывается через сигнатуры: `build_qdrant_filter(user,
  group_ids, curator_scope)` вместо трёх; новые поля скоупа не ломают сигнатуры.
- ✅ `tuple` вместо `list` — hashable, безопасно для frozen VO и для включения в
  кэш-ключ.
- ✅ `UserContext` остаётся источником загрузки; `CuratorScope` — его
  проекция в RAG-контекст. Дублирование полей между VO осознанное: `UserContext`
  обслуживает CRUD/листинг, `ChatContext` — retrieval; сливасть их в один VO
  нельзя (у них разный lifecycle и разный набор потребителей порта).

#### Вариант C — композиция: `ChatContext.user_context: UserContext`

- ❌ `UserContext.build()` делает async I/O к UoW внутри доменного VO — тащить этот
  контракт в RAG-пайплайн значит расширить зависимость retrieval от
  assignment-репозитория.
- ❌ `UserContext` эволюционирует под нужды CRUD (поля типа PII-настроек), что будет
  невольно менять ACL-контракт чата.

**Решение: вариант B.**

### 1.4 Рекомендуемое решение — контракт

1. **Новый VO** `domain/value_objects/curator_scope.py` (код выше). Без импортов
   инфраструктуры — проходит layering-правила AGENTS.md.

2. **`UserContext.build`** возвращает уже `CuratorScope` (или оставляет поля как
   есть и конвертирует в точке сборки `ChatContext` — второе меньше трогает
   existing-тесты; конверсия в `ChatService._prepare_chat`):

   ```python
   # chat_service.py, _prepare_chat
   curator_scope = (
       CuratorScope(
           managed_client_ids=tuple(user_ctx.managed_client_ids),
           managed_internal_ids=tuple(user_ctx.managed_internal_ids),
           managed_group_ids=tuple(user_ctx.managed_group_ids),
       )
       if user_ctx.is_curator
       else None
   )
   ctx = ChatContext(..., curator_scope=curator_scope, ...)
   ```

3. **`RagService._init_state`** (rag_service.py:127-128):

   ```python
   scope = ctx.curator_scope
   access_filter = build_qdrant_filter(
       user, ctx.user_group_ids,
       managed_client_ids=list(scope.managed_client_ids) if scope else None,
       managed_internal_ids=list(scope.managed_internal_ids) if scope else None,
       managed_group_ids=list(scope.managed_group_ids) if scope else None,
   )
   ```

   `get_visibility_conditions` и `_curator_conditions` (access_control.py:86-126)
   **не меняются** — single source of truth сохраняется, меняется только доставка
   данных.

4. **Answer cache — обязательное сопутствующее изменение**:

   ```python
   def compute_visibility_scope_hash(user_kind, user_id, group_ids, user_role, curator_scope):
       scope_repr = (
           f"curator:{sorted(curator_scope.managed_client_ids)}:"
           f"{sorted(curator_scope.managed_internal_ids)}:"
           f"{sorted(curator_scope.managed_group_ids)}"
           if curator_scope else ""
       )
       return content_hash(f"{user_kind}:{user_role}:{user_id}:{sorted(group_ids)}:{scope_repr}")
   ```

   - Включение `user_role` закрывает сценарий понижения роли (см. 1.1).
   - Включение managed-ids закрывает сценарий смены assignments в рамках роли.
   - `CACHE_PREFIX` поднять `rag:cache:v2:` → `rag:cache:v3:` (answer_cache.py:20),
     чтобы старые записи не конкурировали с новой схемой ключа.
   - Побочный эффект: смена assignments куратора инвалидирует его кэш целиком —
     это **корректное** поведение (fail-closed), документируем как trade-off.

5. **Benchmark-контекст** (`infrastructure/benchmark/runner.py:132`): конструирует
   `ChatContext(user_id=0, user_kind="api_key", user_role="admin")` — `curator_scope`
   по умолчанию `None`, ничего менять не нужно. Fakes в `tests/fakes.py` —
   аналогично.

### 1.5 Лимит размера managed-списков (operational guard)

`build_qdrant_filter` превращает managed-ids в `MatchAny(any=owner_ids)`
(acl.py:72-73). Для куратора с тысячами managed-пользователей:

- REST/gRPC-запрос к Qdrant раздувается; фильтрация по длинному `any` на каждой
  точке — деградация латентности dense-поиска.
- В Qdrant длинные `MatchAny` на неиндексированном поле дают full-scan по payload.

**Решение**: числовой порог (предлагаю `CURATOR_SCOPE_MAX_IDS = 1000`,
конфигурируемо в `config.py`). При превышении — не молча обрезать (обрезка =
тихое изменение ACL), а:

1. Логировать WARNING с `user_id` и размером.
2. Явно отказывать запросу с `BusinessRuleViolation("curator scope exceeds
   retrieval limit; use group-based assignment")` — это видимая деградация, а не
   скрытая утечка/потеря полноты.

Долгосрочное исправление — модель assignments через группы
(`managed_group_ids`), а не персональные списки; порог создаёт стимул для
миграции.

### 1.6 План миграции

| # | Шаг | Тесты |
|---|---|---|
| 1 | Новый VO `CuratorScope` + поле в `ChatContext` (default `None`) | unit: frozen/hashable, `None`-семантика |
| 2 | `ChatService._prepare_chat` пробрасывает скоуп | unit: мок UoW, куратор/не-куратор |
| 3 | `rag_service._init_state` + `build_qdrant_filter` принимают скоуп | characterization: фильтр куратора содержит `ASSIGNED`-условия (уже покрыто `test_curator_role.py:313-323` — расширить на путь `RagService`) |
| 4 | `compute_visibility_scope_hash` v3 | unit: разные скоупы → разные хэши; понижение роли → другой хэш |
| 5 | Порог `CURATOR_SCOPE_MAX_IDS` | unit: превышение → исключение |

Инвариатный тест (главный): для фиксированного набора документов
`is_in_search_scope(doc, user_ctx)` ⇔ документ проходит `access_filter`,
построенный из `ChatContext` того же пользователя, — property-based по ролям
USER/CURATOR/ADMIN × visibility × owner/group-комбинациям. Этот тест фиксирует
пункт 1.2.1 навсегда.

---

## 2. BM25 ACL strategy

### 2.1 Текущее состояние

```
run_hybrid_search (infrastructure/ml/rag/rag_retrieval.py:106-162)
  ├─ dense:  qdrant_dense_search(query, fetch_k, access_filter)   ← ACL ПРИМЕНЯЕТСЯ
  ├─ sparse: bm25_index.search_with_hashes(query, fetch_k)        ← БЕЗ ACL (глобальный индекс)
  ├─ RRF-merge по хэшам
  └─ resolve_hashes_batch(missing, access_filter)                 ← ACL ПРИМЕНЯЕТСЯ ПОСТ-ФАКТТОМ
```

- `BM25Index` (`infrastructure/bm25/bm25_index.py`) — один глобальный индекс по
  **всем** чанкам всех пользователей; о ACL не знает.
- `bm25_fetch_k = 25` (`config.py:139`).
- Единственное, что спасает от утечки: sparse-кандидаты — это **только хэши**;
  контент и метаданные выдаёт Qdrant строго под `access_filter`
  (`resolve_hashes_batch`, rag_retrieval.py:30-72).

Два следствия текущей схемы:

1. **Recall-деградация для ограниченных скоупов** (включая кураторов после
   внедрения решения 1): top-25 BM25 по глобальному корпусу может целиком
   состоять из чужих документов → после резолва sparse-ветка даёт 0 кандидатов →
  гибрид молча деградирует до dense-only именно для тех пользователей, кому
   точный лексический поиск нужнее всего (клиенты со своими документами, точные
   reference-запросы — ср. `exact_ref_sparse_boost`, config.py:124).
2. **Хрупкий инвариант**: безопасность держится на соглашении "любой потребитель
   `search_with_hashes` обязан резолвить хэши через Qdrant с ACL-фильтром".
   Сегодня это `run_hybrid_search` и benchmark (`infrastructure/benchmark/
   retrieval.py:53-57` — там admin-контекст, безопасно). Новый потребитель без
   резолва = утечка. Соглашение не проверяется ни типами, ни тестом.

### 2.2 Варианты

#### Вариант 1 — Post-filter (status quo)

Глобальный BM25, фильтрация при резолве в Qdrant.

- **Memory**: O(corpus), один индекс, без дублирования. ✅
- **Performance**: поиск быстрый, но (a) кандидаты впустую, (b) recall-потери для
  узких скоупов, (c) лишний round-trip `resolve_hashes_batch`. ⚠️
- **Security**: корректна ровно до первого потребителя, забывшего про резолв. ⚠️
- **Ops**: ноль изменений. ✅

#### Вариант 2 — Per-user / per-scope индексы

Отдельный `BM25Index` на пользователя (или на scope: user + группы + public).

- **Memory**: `INTERNAL_PUBLIC` и групповые документы дублируются в индексе каждого
  пользователя → O(users × visible_corpus); при 50+ пользователей (loadtest
  создаёт 50) — кратный рост RAM. ❌
- **Performance**: точный top-k без пост-фильтрации. ✅
- **Security**: изоляция по построению. ✅
- **Ops**: катастрофично: любое изменение ACL (загрузка документа, смена состава
  группы, смена assignments куратора, удаление документа) требует перестройки
  затронутых индексов; текущая инвалидация — глобальный Redis pub/sub + полная
  перезагрузка из S3 (`bm25_invalidation.py`, `bm25_updater.py`,
  `main.py:122-127`) — механизм придётся переписывать на инкрементальный. ❌

**Отклонён.**

#### Вариант 3 — Pre-filter в памяти (ACL-aware BM25)

Хранить в `BM25Index` по каждому документу его ACL-атрибуты
(`visibility`, `owner_id`, `group_id`), а в `search_with_hashes` принимать
`VisibilityCondition[]` (уже вычисленные `get_visibility_conditions`) и
отсекать кандидатов **до скоринга**:

```python
# bm25_index.py (эскиз)
def search_with_hashes(self, query, k, visibility_conditions=None) -> list[tuple[str, float]]:
    q_tokens = tokenize(query)
    candidates = self._candidates(q_tokens)             # из inverted index
    if visibility_conditions is not None:
        candidates = (i for i in candidates
                      if self._doc_matches(i, visibility_conditions))
    ...
```

`_doc_matches` — прямая трансляция тех же `VisibilityCondition` (visibility ==
payload.visibility AND owner match / group match), т.е. третье потребление того же
canonical IR. Дополнительно кэшировать bitmap множества видимых индексов на
(request-hash → set) с коротким TTL, чтобы не пересчитывать предикат на каждый
кандидат.

- **Memory**: +3 поля на документ; bitmap-кэш O(users in flight × docs). ✅
- **Performance**: предикат дешёвый (сравнения по целым), кандидаты из inverted
  index уже ограничены; устраняет и пустые кандидаты, и лишние резолвы. ✅
- **Security**: фильтрация до того, как хэши покидают индекс; но резолв через
  Qdrant-фильтр **сохраняем как второй рубеж** (defense in depth). ✅✅
- **Ops**: индекс перестраивается как сегодня (по content_hash), при инкрементальном
  `add_text/replace_text` просто добавляются ACL-поля. ✅
- **Риск**: рассинхрон ACL-атрибутов в BM25-индексе с payload Qdrant (документ
  сменил visibility). Митигируется: BM25-версия пересобирается тем же worker'ом,
  что пишет Qdrant; рассинхрон безопасен по направлению — **fail-closed**
  гарантируется вторым рубежом в Qdrant (резолв с фильтром), а рассинхрон в
  сторону "BM25 не отдаст лишнего" даёт лишь recall-потерю, не утечку.

#### Вариант 4 — Native Qdrant sparse (Query API hybrid)

Перенести sparse-ветку в Qdrant: sparse-вектора в той же коллекции + Query API с
двумя prefetch и server-side RRF:

```python
client.query_points(
    collection_name=...,
    prefetch=[
        Prefetch(query=dense_vector, using="dense",  filter=retrieval_filter, limit=fetch_k),
        Prefetch(query=sparse_vector, using="bm25",  filter=retrieval_filter, limit=fetch_k),
    ],
    query=FusionQuery(fusion=Fusion.RRF),
    limit=top_k,
)
```

- **Memory**: sparse-вектор на чанк живёт в Qdrant (одна копия на документ, не на
  пользователя). ✅
- **Performance**: server-side fusion, один round-trip вместо
  dense+BM25+batch-resolve; фильтр применяется движком к обеим веткам. ✅
- **Security**: **единая точка принуждения** — тот же `query_filter`/`Filter`,
  что и в dense; инвариант "sparse без ACL невозможен" становится структурным, а
  не дисциплинарным. ✅✅✅
- **Риски/затраты**:
  - Нужен генератор sparse-векторов при ingestion. BM25-скоринг (Okapi:
    query-side IDF, doc-length normalization, k1/b) не сводится к статическому
    скалярному произведению — прийдётся либо принять приближение
    (SPLADE-подобные эмбеддинги / fastembed BM25), либо хранить tf-веса и
    нормировать на стороне запроса. **Ранжирование изменится** → обязательный
    A/B на `benchmark run` до переключения трафика.
  - Перевесь всех чанков (новое named vector + payload) — полный reindex.
  - Внешняя зависимость от версии Qdrant (Query API / sparse support).
  - In-memory BM25 остаётся нужным для benchmark-sweep
    (`infrastructure/benchmark/_sweep_scoring.py:224-228` кэширует sparse-результаты
    по вопросам) — т.е. это не замена, а дублирование пути на переходный период.

### 2.3 Сводная матрица trade-offs

| Критерий | 1. Post-filter | 2. Per-user | 3. In-memory pre-filter | 4. Native Qdrant |
|---|---|---|---|---|
| RAM | ✅ 1×корпус | ❌ O(users×корпус) | ✅ 1×корпус + метаданные | ✅ sparse в Qdrant, 1× |
| Recall при узком скоупе | ❌ теряет sparse-ветку | ✅ | ✅ | ✅ |
| Латентность запроса | ⚠️ лишний batch-resolve | ✅ | ✅ | ✅✅ 1 round-trip, server RRF |
| Security-инвариант | ⚠️ дисциплина (резолв с фильтром) | ✅ изоляция по построению | ✅ двойной рубеж | ✅✅ структурно (один filter на обе ветки) |
| Инвалидация при смене ACL | ✅ не нужна | ❌ перестройка индексов | ✅ не нужна (условия per-request) | ✅ не нужна |
| Стоимость миграции | — | высокая | низкая | высокая (reindex + смена ранжирования) |
| Риск регрессии качества | — | низкий | низкий (тот же Okapi) | ⚠️ новый скоринг, нужен A/B |

### 2.4 Рекомендация: поэтапный план

- **Фаза 0 (сейчас, до любого кода)** — закрепить статус-кво тестом-инвариантом:
  "выход `bm25_index.search_with_hashes` не содержит контента/метаданных, только
  хэши; единственный путь к контенту — `resolve_hashes_batch` с непустым
  `access_filter`". Плюс логирование доли sparse-кандидатов, выживших после
  резолва (метрика `sparse_survival_ratio`) — она даст данные о масштабе
  recall-проблемы на реальном трафике.
- **Фаза 1 — вариант 3 (in-memory pre-filter)**: дёшево, сохраняет Okapi-семантику
  и текущую инфраструктуру S3/invalidation, чинит recall и закрывает дисциплинарный
  инвариант первым рубежом. Реализуется независимо от решения 1, но **после**
  решения 1 — кураторы получат от него максимальную пользу (их скоуп без
  managed-ids сегодня почти не пересекается с их видимым множеством).
- **Фаза 2 (цель) — вариант 4 (native Qdrant hybrid)**: миграция при следующих
  триггерах: (a) корпус > ~10⁵ чанков и in-memory BM25 перестал влезать в RAM
  worker'а, или (b) нужна строгая аудитируемость единой точки фильтрации. Обязательные
  условия перехода: parity-бенчмарк ранжирования через `benchmark run` +
  `task loadtest:sse` на латентность, полный reindex с версионируемым именем
  коллекции.

Вариант 2 отклонён без фаз — экономия на фильтрации не оправдывает ни memory, ни
инвалидационный шторм.

### 2.5 Security-инварианты (общие для всех фаз)

1. BM25-индекс **никогда** не возвращает `page_content`/payload — только
   `(content_hash, score)`. Контент выдаёт исключительно Qdrant под ACL-фильтром.
   (Сегодня так — сохранить как явный контракт + тест.)
2. Какой бы механизм ни применялся в sparse-ветке, **резолв через Qdrant с
   `access_filter` не убирается никогда** — это второй рубеж на случай
   рассинхрона/бага первого.
3. Логи гибридного поиска (`"Hybrid: dense=%d, sparse=%d, merged=%d"`,
   rag_retrieval.py:150-155) не содержат контента — сохранить; метрику
   `sparse_survival_ratio` добавить без PII.
4. Метрики/логи не должны позволять пользователю A измерить размер чужого
   корпуса: `sparse_survival_ratio` агрегируется по роли, не по `user_id`, в
   экспортируемых дашбордах.

---

## 3. Взаимодействие решений

1. **Порядок внедрения: сначала решение 1, затем фаза 1 решения 2.** Без
   managed-ids в `ChatContext` pre-filter BM25 для кураторов фильтрует по
   неполному скоупу (второй рубеж в Qdrant скорректирует, но recall-выигрыш
   потерян).
2. **Cache**: обе фазы не меняют `compute_visibility_scope_hash`, но фаза 1 делает
   recall зависимым от скоупа → кэшированные ответы, вычисленные до смены
   assignments, могут быть неполными. Это закрыто v3-ключом из п. 1.4.4
   (assignment-смена → новый хэш → miss).
3. **Порог `CURATOR_SCOPE_MAX_IDS`** (п. 1.5) применяется в обеих ветках: и в
   `MatchAny`-фильтре dense, и в предикате варианта 3, и в prefetch-фильтре
   варианта 4 — порог проверяется один раз, при сборке `CuratorScope`/фильтра, а
   не в каждой подсистеме.
4. **Property-тест** из п. 1.6 остаётся главным инвариантом при любой фазе:
   `is_in_search_scope` ⇔ проходит фильтр dense-ветки ⇔ проходит предикат
   sparse-ветки (с фазы 1).
