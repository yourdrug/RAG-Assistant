---
name: review-full
description: >
  Полная многоступенчатая ревизия проекта (DDD, DI, API, RAG, multi-tenant
  безопасность, производительность, тесты) с итоговым отчётом в REVIEW_REPORT.md.
  Вызывает все семь reviewer-агентов через оркестратор review-orchestrator.
---

# Полная ревизия проекта

Проведи полную многоступенчатую ревизию проекта по workflow из skill
`review-orchestrator`.

**Scope:** уточни у пользователя или возьми весь репозиторий.
Если scope не указан — начни с разведки структуры каталогов.

**Алгоритм:**

1. Загрузи skill `review-orchestrator` и следуй его workflow.
2. Вызови по очереди (или параллельно, где независимы) всех семерых custom agents:
   - `ddd-architecture-reviewer`
   - `di-container-auditor`
   - `api-contracts-reviewer`
   - `rag-pipeline-reviewer`
   - `multitenancy-security-auditor`
   - `performance-reliability-reviewer`
   - `test-quality-reviewer`
3. Собери находки, нормализуй формат, дедуплицируй.
4. Сохрани единый приоритизированный отчёт в `REVIEW_REPORT.md` по формату из
   `review-orchestrator` (Executive summary, топ-5 действий, CRITICAL/HIGH/MEDIUM/LOW/INFO, по областям).
5. В чат выведи только Executive summary и топ-5 неотложных действий + путь к файлу.
