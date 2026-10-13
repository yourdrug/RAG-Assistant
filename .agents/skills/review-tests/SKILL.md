---
name: review-tests
description: >
  Ревизия тестовой инфраструктуры — тестовая пирамида, покрытие ACL-сценариев,
  качество фейков, RAG-тесты, concurrency-тесты, тестовая изоляция.
---

# Ревизия тестовой инфраструктуры

Проведи ревизию тестовой инфраструктуры проекта.

**Scope:** уточни у пользователя; если нет — весь `server/tests/`.

**Алгоритм:**

1. Вызови custom agent `test-quality-reviewer` — он проверит:
   - тестовую пирамиду (unit / integration / e2e)
   - покрытие ACL-сценариев (role × visibility × group)
   - качество фейков (`tests/fakes.py`)
   - RAG-тесты (все шаги пайплайна, fallback, prompt injection defense)
   - concurrency-тесты
   - тестовую изоляцию (нет реальных Qdrant/Ollama/PostgreSQL)
2. Собери находки, приведи к формату из `review-orchestrator`.
3. Выведи в чат приоритизированный список (CRITICAL → HIGH → MEDIUM → LOW),
   без сохранения в файл, если явно не попросили сохранить отчёт.
