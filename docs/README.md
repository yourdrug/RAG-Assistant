# Документация проекта

Постоянные документы описывают реализованное поведение и ограничения проекта.
При расхождении с кодом документ нужно уточнить; локальный план не заменяет проверку реализации.

| Направление | Назначение и доступные документы |
|---|---|
| getting started | [Руководство разработчика](development/developer-guide.md): настройка окружения, структура кода и цикл изменений. [Пользовательские сценарии](usage/user-scenarios.md): вход, работа с документами и чатом, администрирование. |
| api | [Быстрый старт API](api/quickstart.md): аутентификация, чат и SSE, поиск, загрузка и удаление документов. Полный контракт генерируется FastAPI и доступен в Swagger. |
| architecture | Слои и потоки: [обзор](architecture/overview.md). Подробно: [модель доступа и ACL](architecture/access-control.md), [конвейер RAG](architecture/rag-pipeline.md). |
| domain | Термины и предметные правила: [глоссарий](domain/glossary.md). |
| operations | [Эксплуатация](operations/operations.md), [диагностика инцидентов](operations/troubleshooting.md), [резервное копирование и восстановление](operations/recovery.md), [конфигурация](operations/configuration.md). |
| adr | Принятые архитектурные решения: [runtime-конфигурация в БД](adr/0001-runtime-config-in-db.md), [пространства знаний и пакеты](adr/0003-knowledge-spaces-and-domain-packs.md). |
| quality | [Оценка качества RAG](quality/README.md): подготовка benchmark-набора, запуск сравнения и чтение метрик. Сырые выгрузки локальны. |
| work | [Рабочие планы и заметки](work/README.md); синхронизируются через Git, но не являются источником истины. |

Сырые benchmark-выгрузки сохраняйте в `docs/quality/artifacts/`, а устойчивые выводы
оформляйте отдельно в постоянных документах. CLI не выбирает этот каталог автоматически:
передавайте путь через `--out`. Из корня репозитория при установленном окружении:

```bash
mkdir -p docs/quality/artifacts
cd server/app
../.venv/bin/python -m composition.cli benchmark export 42 --format json --out ../../docs/quality/artifacts/run-42.json
```

`42` замените существующим ID запуска из БД. Экспорт требует доступной БД и корректной
bootstrap-конфигурации; существующий файл не перезаписывается. CLI экспорта не изменён.
