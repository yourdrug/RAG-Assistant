# Источники конфигурации

Bootstrap PostgreSQL/Redis, CORS/trusted proxy, S3 connection и deployment —
`server/.env`. Пароли и API-ключи — только `server/.env.secrets` (0600).
`task env` создаёт недостающие файлы и закрывает права secret env, сохраняя содержимое.
`client/.env` содержит `VITE_API_URL`: для изменения SPA нужна пересборка.

LLM/OCR/RAG, endpoints, quotas, retention и Qdrant corpus управляются из PostgreSQL.
Лимит файла preview (`preview_max_size_mb`) и срок его хранения
(`preview_cache_ttl_seconds`) также меняются через админку, в разделе Ingestion.
Размер применяется к следующей загрузке preview независимо от лимита документов;
TTL — при следующем чтении или очистке, включая ранее сохранённые preview.
Сохранённое значение может отличаться от применённого: проверяйте applied/error
status и readiness API/worker. Cost control учитывает физические HTTP-попытки
OpenRouter (text/structured/judges), DeepInfra BAAI/bge-m3 embedding и
Qwen/Qwen3-Reranker-4B reranking. При включённых лимитах неизвестная цена или
неподдерживаемый billing contract блокирует запрос; при выключенных лимитах
budget reservations/accounting обходятся. Реальная сверка с provider billing
остаётся обязательной перед live-приёмкой. Смена embedding identity
требует [отдельной миграции](embedding-migration.md).

`COMPOSE_PROFILES=tei` включает локальные TEI сервисы. Пустое значение оставляет
их выключенными. Выбор `ml_provider` в БД не меняет Docker-профиль.
`task pull-model MODEL=qwen2.5:7b` скачивает модель Ollama; затем выберите её в БД.

## Переход существующей установки

Live действия выполняются только в согласованное окно. До него подготовьте
backup PostgreSQL и S3, проверьте восстановление, сохраните protected snapshots
рабочих env/secrets (0600). Не заменяйте рабочий env новым шаблоном.

1. Остановите старые config/indexing writers, API и worker в согласованном окне.
   Миграция выполняется новым CLI с доступом к прежним БД и env, без запуска ML.
2. Из `server/app`, с настроенным bootstrap подключения, выполните dry run:
   `../.venv/bin/python -m presentation.cli config migrate-env --env-file ../.env --trim-env`.
   Для контейнера передайте путь к защищённой доступной на запись копии env и
   соседнему `<env-file>.secrets`; штатные env_file не монтируют файлы внутрь контейнера.
3. Сверьте отчёт ключей и существующие DB values в защищённом окружении.
   Existing DB values имеют приоритет и не перезаписываются. Конфликты S3,
   credentials и legacy pools блокируют apply; исправьте источник и повторите dry run.
   Убедитесь, что `COMPOSE_PROFILES` задан явно для нужных сервисов. Если его нет,
   миграция сохраняет прежний выбор TEI из env перед удалением ML_PROVIDER.
4. Повторите ту же команду с `--apply`. Commit БД предшествует записи env.
   Создаются protected backups; dynamic и legacy keys удаляются только с `--trim-env`.
   Все static overrides, включая CORS, FORWARDED_ALLOW_IPS, REDIS_USER и S3,
   сохраняются. Совпадающие static defaults автоматически не удаляются.
   При ошибке записи устраните причину и повторите: existing DB values сохраняются.
5. Примените `alembic upgrade head` новым кодом с прежними writers остановленными;
   нужны migrations cost budget/audit, corpus control и benchmark reports.
   Пересоздайте new server/worker; `restart` не обновляет env_file. Проверьте
   readiness, applied settings, API→worker update, active requests при смене clients,
   upload/rate/cost enforcement, прежний S3 corpus и history/regression/export без DATA_DIR.

Не публикуйте env, backups, rendered Compose или значения credentials в отчётах.
Откат требует совместимости schema/env/API: восстановите env из protected snapshot,
legacy S3 DB rows — из pre-migration backup при необходимости старой версии.
Сохраните benchmark history/checkpoints/S3 artifacts. Откат Qdrant — отдельная
операция с проверкой новых writes; одного возврата URL недостаточно.

Полный каталог и состояние приёмки: [план](configuration-plan.md).

Постоянное файловое хранение приложения — только S3-совместимое (включая MinIO).
`FILE_BACKEND` больше не выбирает backend; удалите его из env. `DATA_DIR` не нужен
для загрузки документов, preview и файловых отчётов. Временные файлы парсеров очищаются;
checkpoint/history рантайма хранятся в PostgreSQL.
