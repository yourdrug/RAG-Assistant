---
name: multi-tenant-rag-security
description: >
  Чек-лист безопасности и изоляции данных для RAG-системы с
  visibility-based ACL: роли (ADMIN/CURATOR/USER), видимость документов
  (INTERNAL_PUBLIC/INTERNAL_GROUP/INTERNAL_PRIVATE/CLIENT_PRIVATE),
  группы, UserContext/ChatContext, ACL-фильтры в Qdrant через
  build_qdrant_filter(), кэш ответов с visibility scope hash. Использовать
  для аудита изоляции данных между пользователями.
license: MIT
---

# Модель доступа проекта

Проект **не использует** классическую multi-tenancy (нет `tenant_id`, нет
`X-Tenant` заголовка). Вместо этого — **visibility-based ACL** с ролями
и группами.

## Ключевые термины (для grep)

| Термин | Файл | Описание |
|---|---|---|
| `UserContext` | `domain/value_objects/user_context.py:12` | Frozen dataclass: `user_id`, `user_kind`, `user_role`, `group_ids`, `managed_*_ids` |
| `ChatContext` | `domain/value_objects/chat_context.py:15` | Frozen dataclass: user identity + group_ids + depth + as_of_date |
| `UserRole` | `domain/value_objects/roles.py:10` | Enum: `ADMIN`, `CURATOR`, `USER` |
| `UserKind` | `domain/value_objects/roles.py:16` | Enum: `INTERNAL`, `CLIENT` |
| `DocumentVisibility` | `domain/value_objects/visibility.py:10` | Enum: `INTERNAL_PUBLIC`, `INTERNAL_GROUP`, `INTERNAL_PRIVATE`, `CLIENT_PRIVATE` |
| `VisibilityCondition` | `domain/services/access_control.py:33` | Canonical AND-clause: visibility + owner_match + group_match |
| `get_visibility_conditions()` | `domain/services/access_control.py:140` | **Единый источник ACL-логики** |
| `build_qdrant_filter()` | `infrastructure/vector/acl.py:28` | Транслирует `VisibilityCondition` → Qdrant `Filter` |
| `answer_cache` | `infrastructure/ml/answer_cache.py:4` | Visibility scope hash для предотвращения cross-user утечек |
| `get_current_user()` | `presentation/api/auth_dependencies.py` | JWT → `UserContext` |

## Матрица доступа

| Роль | INTERNAL_PUBLIC | INTERNAL_GROUP | INTERNAL_PRIVATE | CLIENT_PRIVATE |
|---|---|---|---|---|
| ADMIN (INTERNAL) | ✅ | ✅ (свои группы) | ✅ (все) | ✅ (только listing) |
| CURATOR (INTERNAL) | ✅ | ✅ (свои группы) | ✅ (все + managed users) | ❌ |
| USER (INTERNAL) | ✅ | ✅ (свои группы) | ✅ (только свои) | ❌ |
| CLIENT | ❌ | ❌ | ❌ | ✅ (только свои) |

## Правила

- **CLIENT** видит только свои `CLIENT_PRIVATE` документы.
- **INTERNAL/USER** видит `INTERNAL_PUBLIC` + свои `INTERNAL_PRIVATE` + групповые `INTERNAL_GROUP`.
- **INTERNAL/CURATOR** видит всё что USER, плюс документы managed users.
- **INTERNAL/ADMIN** видит все internal + (только listing) все `CLIENT_PRIVATE`.
- **Группы** — механизм для `INTERNAL_GROUP` sharing.

# Главный вопрос ревью

> Может ли пользователь A получить доступ (напрямую или через ответ LLM)
> к документу, диалогу или метаданным, принадлежащим пользователю B?

Ищи конкретный путь эксплуатации для каждой точки данных: HTTP API,
retrieval из векторной БД, кэш, логи, фоновые задачи.

# 1. Источник identity пользователя

- [ ] `user_id`, `user_role`, `user_kind`, `group_ids`, `managed_*_ids`
      **всегда** берутся из JWT (через `get_current_user()` →
      `UserContext`), а не из тела запроса/query-параметра.
- Code smell (CRITICAL): эндпоинт, принимающий `user_id` как параметр
  без сверки с аутентифицированным пользователем (IDOR).
- [ ] Смена роли/видаимости в рамках сессии (если поддерживается)
      явно валидируется на каждый запрос.

# 2. Сквозное распространение ACL-контекста

Проследи путь:

```
get_current_user() → UserContext
  → ChatContext (для RAG)
    → application service
      → repository / Qdrant query
```

- [ ] **Нет разрывов:** каждый вызов репозитория/поиска получает ACL
      контекст, а не работает "вслепую".
- [ ] **Ищи методы без ACL-параметра:** `get_all_documents()` без
      `user_context`/`visibility_conditions` — потенциальная утечка.

# 3. Qdrant — ACL-фильтрация

Используется единая коллекция с ACL-фильтрами через `build_qdrant_filter()`.

- [ ] **Каждый вызов** `client.search()`, `client.query_points()`,
      `client.scroll()`, `client.retrieve()` **обязан** содержать
      `query_filter` через `build_qdrant_filter(visibility_conditions)`.
      Найти хотя бы один вызов **без** `query_filter` в пользовательских
      путях — CRITICAL.
- [ ] **Постфильтрация в Python** (`[r for r in results if ...]` после
      `client.search()` без `query_filter`) — CRITICAL: (1) top-k может
      целиком оказаться "чужим", (2) при ошибке в условии чужие документы
      утекают в контекст LLM.
- [ ] **`scroll`/`retrieve` без фильтра.** Если через них проходят
      пользовательские запросы — фильтр обязателен.
- [ ] **ID точек и коллизии.** ID точки Qdrant должен включать
      `user_id`/`document_id` для исключения коллизий между разными
      пользователями.
- [ ] **Multi-vector / named vectors.** Фильтр по ACL применяется
      независимо от имени вектора (dense/sparse).

# 4. Авторизация на уровне документа

- [ ] `DocumentVisibility` учитывается в `VisibilityCondition`:
      - `INTERNAL_GROUP` → проверяется пересечение `group_ids` документа
        и `group_ids` пользователя.
      - `INTERNAL_PRIVATE` → проверяется `owner_id == user_id` (или
        `user_id in managed_user_ids` для кураторов).
      - `CLIENT_PRIVATE` → проверяется `owner_id == user_id`.
- [ ] Эти проверки происходят **в `get_visibility_conditions()`**, а не
      разбросаны по роутерам.

# 5. Контекст, отправляемый в LLM

- [ ] В финальный промпт попадают только чанки, прошедшие полный ACL-фильтр
      текущего пользователя (через `build_qdrant_filter()`).
- [ ] История диалога принадлежит тому же `user_id`, что и текущий запрос.

# 6. Кэш

- [ ] Ключ `answer_cache` включает visibility scope hash
      (`infrastructure/ml/answer_cache.py:4`). Без этого ответ,
      сгенерированный для одного пользователя, может быть отдан другому.
- [ ] Semantic cache (если есть) — ключ обязан включать `user_id`.

# 7. Квоты и биллинг

- [ ] Есть лимиты на пользователя по числу запросов, токенов, стоимости.
- [ ] Поведение при превышении — явное (HTTP 429 + `Retry-After`).
- [ ] Общий rate limit к LLM-провайдеру защищён от монополизации одним
      пользователем (per-user семафор/очередь).

# 8. PII и секреты

- [ ] Персональные данные в документах/диалогах шифруются at rest
      (или явно описано, почему это не требуется).
- [ ] Логи не содержат сырого содержимого документов/сообщений без
      необходимости; есть маскирование чувствительных полей.
- [ ] Есть механизм удаления данных пользователя (retention policy /
      GDPR-подобное "право на забвение"), покрывающий реляционную БД,
      Qdrant и кэш.
- [ ] Секреты (ключи LLM, DSN БД) — не в коде/репозитории, не всплывают
      в сообщениях об ошибках.

# 9. Аудит

- [ ] Обращения к LLM (кто, когда, сколько токенов) логируются с
      привязкой к `user_id` для расследования инцидентов, без логирования
      чувствительного содержимого в открытом виде.

# 10. Prompt injection как вектор атаки

- [ ] Если `INTERNAL_PUBLIC` документы видны всем INTERNAL-пользователям —
      оцени, может ли вредоносный документ, загруженный одним
      пользователем, содержать инструкции, влияющие на ответы другим
      пользователям.
- [ ] Промпт явно разделяет "инструкции системы" и "извлечённый контент",
      модели явно указано не выполнять команды из контента документов.

# Формулировка находок

Для каждой находки уровня CRITICAL/HIGH обязательно опиши конкретный
воспроизводимый сценарий эксплуатации ("пользователь A с ролью USER
отправляет запрос X → получает в ответе фрагмент документа с visibility
INTERNAL_PRIVATE, принадлежащего пользователю B"), а не только абстрактное
"нет изоляции".
