---
name: review-arch
description: >
  Ревизия архитектуры — только DDD-слои и граф Dependency Injection.
  Вызывает ddd-architecture-reviewer и di-container-auditor. Без сохранения в файл.
---

# Ревизия архитектуры (DDD + DI)

Проведи ревизию только по двум областям: DDD-архитектура и Dependency Injection.

**Scope:** уточни у пользователя; если нет — весь репозиторий.

**Алгоритм:**

1. Сначала загрузи skills `fastapi-ddd-layering` и `dependency-injection-patterns` —
   они являются источником правды для этих областей.
2. Вызови только двух custom agents (не остальных):
   - `ddd-architecture-reviewer` — слои и границы агрегатов
   - `di-container-auditor` — граф зависимостей и их время жизни
3. Собери находки, приведи к формату из `review-orchestrator`.
4. Выведи в чат приоритизированный список (CRITICAL → HIGH → MEDIUM → LOW),
   **без** сохранения в файл, если явно не попросили сохранить отчёт.
