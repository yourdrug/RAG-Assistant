---
name: review-security
description: >
  Быстрый security-аудит multi-tenant изоляции данных без остальных областей ревью.
  Запускает multitenancy-security-auditor. CRITICAL/HIGH — с воспроизводимым сценарием эксплуатации.
---

# Security-аудит изоляции данных

Проведи аудит изоляции данных между пользователями/тенантами.

**Scope:** уточни у пользователя; если нет — весь репозиторий, с упором на:
- код аутентификации (`presentation/api/auth_dependencies.py`)
- авторизацию и ACL (`domain/services/access_control.py`, `infrastructure/vector/acl.py`)
- retrieval и работу с векторной БД (`infrastructure/ml/rag/`, `infrastructure/vector/`)

**Алгоритм:**

1. Загрузи skill `multi-tenant-rag-security` — используй её чек-лист как
   основной источник правды.
2. Вызови custom agent `multitenancy-security-auditor`.
3. Для каждой находки CRITICAL/HIGH обязательно приведи конкретный
   воспроизводимый сценарий эксплуатации (шаги "как пользователь A увидит данные Б").
4. Выведи результат в чат, без сохранения в файл, если явно не попросили.
