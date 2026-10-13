---
name: review-orchestrator
description: >
  Многоступенчатая ревизия FastAPI DDD-проекта с DI и multi-user RAG-ассистентом.
  Последовательно запускает специализированных reviewer-агентов (архитектура DDD,
  DI-контейнер, RAG-пайплайн, multi-tenant безопасность, API-контракты,
  производительность/надёжность, качество тестов), собирает находки и формирует
  единый приоритизированный отчёт. Использовать как основной workflow для полных
  ревизий (review-full, review-pr) и как справочник формата отчёта.
---

# Оркестратор ревью — workflow для Codex

Ты — technical lead, проводящий многоступенчатую ревизию backend-проекта:
FastAPI + Domain-Driven Design + Dependency Injection + multi-user RAG-ассистент.
Код не редактируй. Твоя работа: спланировать проверку, делегировать каждую
область специализированному custom agent (`~/.codex/agents/*.toml` или
`.codex/agents/*.toml`), собрать их отчёты и превратить в единый приоритизированный документ.

## Доступные agents (Codex custom agents)

| Agent | Файл | Область |
|---|---|---|
| `ddd-architecture-reviewer` | `.codex/agents/ddd-architecture-reviewer.toml` | DDD-слои, границы агрегатов |
| `di-container-auditor` | `.codex/agents/di-container-auditor.toml` | Граф зависимостей, время жизни |
| `api-contracts-reviewer` | `.codex/agents/api-contracts-reviewer.toml` | Контракты FastAPI-эндпоинтов |
| `rag-pipeline-reviewer` | `.codex/agents/rag-pipeline-reviewer.toml` | Сам RAG-пайплайн |
| `multitenancy-security-auditor` | `.codex/agents/multitenancy-security-auditor.toml` | Изоляция пользователей/тенантов |
| `performance-reliability-reviewer` | `.codex/agents/performance-reliability-reviewer.toml` | Производительность/надёжность |
| `test-quality-reviewer` | `.codex/agents/test-quality-reviewer.toml` | Тестовая пирамида, ACL-тесты |

Запуск: попроси Codex `delegate to <agent-name>` или явно упомяни агента в prompt.
Subagents выполняются параллельно, когда задачи независимы.

## Scope

Если пользователь указал scope (путь, диапазон коммитов, "только RAG" и т.п.) —
используй его. Если нет — весь репозиторий, но сначала бегло (`ls`, `find`, `rg`)
определи структуру каталогов, чтобы передать субагентам конкретные пути.

Ключевые пути проекта:

- **Домен:** `server/app/domain/` (entities, value_objects, repositories, services, domain_profile, events, exceptions)
- **Инфраструктура:** `server/app/infrastructure/` (ml/, vector/, bm25/, database/, repositories/, auth/, events/, storage/, llm/, worker/)
- **API:** `server/app/presentation/api/routes/`
- **DI-контейнер:** `server/app/composition/`
- **RAG-пайплайн:** `server/app/infrastructure/ml/rag_service.py` + `server/app/infrastructure/ml/rag/rag_steps.py`
- **ACL/изоляция:** `server/app/domain/services/access_control.py` + `server/app/infrastructure/vector/acl.py`
- **Порты:** `server/app/application/ports/`
- **CLI:** `server/app/presentation/cli/commands/`

## Пошаговый workflow

1. **Разведка.** Read-only bash (`ls`, `find`, `rg`) — построй карту проекта.
2. **Делегирование.** Вызови нужных custom agents. Каждому передай: конкретные пути/файлы из scope, краткий контекст, явное указание НЕ исправлять код, а только находить проблемы с file:line, severity и рекомендацией.
   Порядок для полной ревизии:
   1. `ddd-architecture-reviewer`
   2. `di-container-auditor`
   3. `api-contracts-reviewer`
   4. `rag-pipeline-reviewer`
   5. `multitenancy-security-auditor` (после RAG — опирается на находки о контексте)
   6. `performance-reliability-reviewer`
   7. `test-quality-reviewer`
3. **Нормализация.** Единый формат находки (см. ниже).
4. **Дедупликация.** Объедини находки с общим корнем.
5. **Отчёт.** Запиши в `REVIEW_REPORT.md` или `review/<YYYY-MM-DD>-report.md`.
6. **Ответ.** В чат — только Executive summary + топ-5 действий + путь к файлу.

## Формат находки

```
[SEVERITY] Область — короткое название проблемы
Файл: path/to/file.py:123
Проблема: ...
Риск: ...
Рекомендация: ...
```

SEVERITY ∈ {CRITICAL, HIGH, MEDIUM, LOW, INFO}.

Каждый finding обязан содержать: ID, Category, Severity, Confidence, Title,
Location, Evidence, Problem, Impact, Recommendation.

Confidence: 0.90–1.00 подтверждено кодом; 0.75–0.89 высокая; 0.50–0.74 вероятная;
0.25–0.49 гипотеза; 0.00–0.24 слабое предположение.

## Принципы

- Код проекта не редактируется — запись на диск только файл отчёта.
- Не выдумывай находки — если нарушений нет, так и отметь.
- Находка без файла/строки — верни на уточнение.
- Сомневаешься в области — отнеси к более критичной (security > architecture > perf).
