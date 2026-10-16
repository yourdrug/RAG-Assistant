# План разработки: конфигурация через админку и backend без локального постоянного состояния

> **Для агента-исполнителя:** прочитать этот файл целиком и AGENTS.md. Реализовывать последовательно по этапам с checkbox-проверками. Применить superpowers:executing-plans или аналогичный workflow текущего окружения; делегирование — только по разрешённым в сессии правилам. Этот документ достаточен без истории диалога.

**Дата:** 2026-10-13; актуализация после независимого ревью — 2026-10-15. **Статус:** этапы 1–4 и 6 реализованы; код этапа 5 реализован, проверка на настоящем Redis остаётся открытой; этап 7 выполнен частично (текстовый OpenRouter, остальные платные пути блокируются); реализация этапа 8 присутствует с ограничениями приёмки ниже. Этапы 9–11 реализованы и проверены локально; этап 12 ожидает реализации. Рабочие env, БД и сервисы не изменены. Изменения этапов 5–11 пока в рабочем дереве, отдельные reviewable commits не созданы.

**Goal:** сократить env до секретов, bootstrap/deployment-настроек и необходимых статических overrides; перенести управляемые настройки в БД с настоящим применением; убрать обязательное локальное хранение benchmark-истории, отчётов и checkpoints у S3 backend.

**Architecture:** существующие Pydantic Settings, config_parameters, UoW, EventBus/NOTIFY, PostgreSQL и FileStorage/S3. Новый settings.toml, YAML, конфигурационный сервис и замена DI не нужны. DB settings и фактически применённые ресурсы должны совпадать либо явно показывать состояние применения.

**Tech Stack:** Python 3.11, FastAPI, Pydantic Settings, SQLAlchemy, PostgreSQL, Redis/pyrate-limiter, httpx, Qdrant, S3/MinIO, Typer, React, pytest, ruff. Использовать существующие зависимости.

**Spec:** цели, решения, контракты, этапы и полный каталог ниже — единая спецификация и план. Ссылки на исходники служат навигацией, не заменяют требования. Старые таблица и план конфигурации заменены этим документом.

## 1. Принятые решения

1. У каждого параметра один источник управления. `config.py` содержит типы/defaults/валидацию, а не настройки конкретного сервера.
2. Секреты — только secret env/окружение. Пароли и ключи нельзя редактировать через config_parameters или читать из legacy DB-записей.
3. Env — подключение к PostgreSQL/Redis, CORS/trusted proxy, S3 connection/bucket, Docker/релиз и необходимые статические overrides. Не перечислять весь каталог defaults в env.
4. Настройки LLM/OCR/RAG, URL ML-провайдеров, лимиты загрузки, retention фоновых задач, rate quotas и рабочие cost limits — БД через админку.
5. QDRANT_URL и COLLECTION_NAME также становятся управляемыми из БД, но отдельным этапом с проверкой нового корпуса, согласованием процессов и кэшей. Не реализовывать их как безусловный setattr.
6. Показ используемых адресов/корпуса доступен независимо от возможности редактирования. Сохранённое значение не выдавать за применённое, если процесс ещё использует прежний клиент.
7. HTTP_POOL_* сейчас относятся только к embedding/reranker, не всему приложению. Заменить на четыре явно названных параметра с текущими эффективными defaults 40/20 и 20/10. Ограничения оставить: это открытые и сохраняемые свободные соединения каждого клиента в каждом процессе.
8. CURATOR_SCOPE_MAX_IDS сохранить: дешёвая проверка размера ACL scope защищает от огромных фильтров. В этой задаче оставить статический default 1000 и override в env, без удаления проверки.
9. RATE_LIMIT_REDIS_PREFIX и RATE_LIMIT_MAX_BUCKETS оставить техническими статическими параметрами; quotas, enabled и fail_open перенести в БД. Cost limits показывать как работающие только после реализации учёта/ограничения.
10. История/метрики/baseline и checkpoints benchmark — PostgreSQL. Сохраняемые CSV/JSON и крупные артефакты — S3. Небольшой экспорт можно генерировать по запросу из БД. Локальный экспорт — только явная опция CLI.
11. Временные файлы допустимы для библиотек, которым нужен путь, и чтобы не держать большие документы в RAM. Они не являются постоянным состоянием и не должны требовать DATA_DIR при S3.
12. ML_PROVIDER выбирает провайдера приложения; COMPOSE_PROFILES выбирает запускаемые Docker-сервисы. Не связывать эти решения одним ключом.

## 2. Контракты и ограничения

- Existing DB values сохраняются; отсутствующие dynamic settings создаются из defaults. Старые значения env переносятся до удаления строк, без перезаписи существующих DB rows.
- Dynamic env после завершения соответствующего этапа не управляет параметром. Явные Settings(...) kwargs для тестов и benchmark context overrides сохраняются.
- Старые и новые имена HTTP-pool параметров мигрируются с сохранением эффективных 40/20 embedding и 20/10 reranker; отличие старых overrides сохраняется с той же арифметикой, а недопустимые значения требуют решения, не молчаливого округления.
- Подключение к БД/Redis и security checks доступны до чтения DB-конфига. Проверки фактического ML provider — после загрузки DB settings, до первой ML-операции.
- Изменение API endpoint требует admin, допустимой схемы URL и правил доверия адресам; нельзя отправлять существующий provider API key на произвольный новый хост. Для публичных облачных provider — явно разрешённые origin, для TEI/Ollama — настроенные внутренние адреса. Не следовать редиректам с credentials на другой origin.
- Multi-key конфигурация, нужная для одного переключения, применяется одним согласованным snapshot. In-flight операция заканчивается на своём snapshot; новый запрос не смешивает старый клиент с новой коллекцией/моделью.
- DB transaction не удерживается во время сетевого probe. Сначала подготовка/проверка, затем короткий commit с проверкой актуальности версии; не закрывать старый клиент до успешного переключения и завершения его операций.
- Snapshot/version и состояния применения хранить только в объёме, нужном для нескольких процессов и показа ошибки; не строить универсальную платформу конфигурации. Существующий EventBus ловит ошибки подписчиков: публикация события не равна успешному применению.
- API/worker и CLI используют одни правила bootstrap. DB-конфиг не загрузился/не прошёл обязательную валидацию — нет готовности с тихим fallback на другую модель или корпус.
- Import layering, enums/constants, отсутствие production assert и правила имён из AGENTS.md обязательны. Для god-files сначала characterization baseline на затронутые публичные методы.
- Не менять ACL, семантику фильтров, JWT security, embedding identity и значения работающего приложения ради сокращения env.
- Изменение secret env требует recreate контейнера, обычный docker compose restart не обновляет env_file.
- Поэтапная разработка не означает разрешение на live миграцию/перезапуск. Планирование не запускает тесты или сервисы. На реализации выполнять проверки ниже; live действия — в согласованное окно.

## 3. Подтверждённая база и ограничения исследования

- На момент анализа: 144 поля Settings; 44 активных server/.env параметра; 9 secrets. Реальные значения БД не сверялись, секреты не выводились.
- DYNAMIC_PARAMS уже содержит LLM/OCR/RAG и S3 connection/credentials. S3 secrets сейчас могут применяться из БД — этот путь требуется закрыть.
- ML-провайдер проверяется при создании Settings, до bootstrap. API, worker и ingestion CLI сейчас проходят разные global/domain загрузки.
- ML clients кэшируют URL и pool limits. Простое изменение settings не переключает все клиенты, включая health probes, auxiliary LLM и benchmark judges.
- Rate policies/limiter создаются при старте. Cost settings есть в config.py, но потребителей для контроля расходов не найдено. Token usage capture/metrics уже имеются — использовать их как основу, а не писать второй сборщик.
- Web sweep уже использует DatabaseBenchmarkCheckpoints и benchmark_runs; файловый checkpoint адаптер нужен для CLI и импорта legacy запусков. Не заменять уже работающие DB checkpoints заново.
- benchmark/persistence.py всё ещё пишет обязательные JSON/CSV и JSONL history; history/regression имеют файловый путь. DATA_DIR также используется в local storage и path validation API benchmark.
- Старые EMBED_MODEL/RERANK_MODEL/RERANK_DEVICE не имеют активных потребителей в просмотренном коде; README содержит старые упоминания.

## 4. Review Focus

1. Сохранение текущих значений, domain_key, secrets и импорт старой benchmark-истории без дублей.
2. Одновременные обновления настроек, потерянный NOTIFY, cold start worker и CLI до первого запуска API.
3. Незавершённые requests/streams и indexing jobs во время переключения endpoint/корпуса; корректное закрытие клиентов.
4. Одновременное потребление rate/cost quotas, уменьшение квоты без обнуления истории и ошибка backend.
5. Backend с S3, недоступным для постоянной записи DATA_DIR и другим worker после retry; результаты/checkpoints не теряются.

## 5. Этапы разработки

Каждый этап: сначала тесты на заданный контракт, подтвердить FAIL новых сценариев, реализовать минимальную полную часть, получить PASS и отдельный reviewable commit. Не менять рабочие env до этапа 12. По завершении этапа обновить checkbox и записать команды/результаты в этот файл.

### Этап 1. Зафиксировать источники и миграционные правила

**Files:** config.py, domain/value_objects/dynamic_config.py, infrastructure/initialization.py; tests/test_configuration_sources.py (новый); каталог в разделе 6 этого файла.

- [x] Проверить каталог против текущего checkout; неизвестные новые поля классифицировать, не терять. Secret list канонизировать в подходящем слое, не импортировать infrastructure constants в application/domain.
- [x] Добавить проверки полного покрытия Settings, scope доменных параметров и отсутствия secret keys в будущем dynamic registry.
- [x] Для LLM_PROVIDER/LLM_MODEL/ML_PROVIDER задать defaults ollama/qwen2.5:7b/tei существующими enum там, где есть; deployment migration сохраняет реальные значения. Новые endpoint defaults отмечены в каталоге, а не выдаются за уже реализованные.
- [x] Зафиксировать набор static/S3 и новые dynamic поля. Пока не удалять legacy env и DB rows и не показывать неработающие cost controls.
- [x] Проверка: `cd server && .venv/bin/pytest tests/test_configuration_sources.py -q`; green baseline существующих config/domain tests.

**Результаты этапа 1 (2026-10-13):**

- Общая ветка всех этапов: `configuration-management`.
- Сверка checkout: 144 поля Settings; каталог содержит 146 целевых ключей
  (два legacy HTTP pool поля заменяются четырьмя на этапе 4), 8 доменных defaults.
  Неизвестных полей нет. Канонические 9 secret keys находятся в domain;
  masking/broadcast/listener используют этот список через существующий импорт.
- `STATIC_SETTING_KEYS`, `SECRET_SETTING_KEYS`, `METADATA_SETTING_KEYS` и
  `TARGET_DYNAMIC_SETTING_KEYS` фиксируют целевое владение. Последний — декларация,
  не активный hot-reload registry. Legacy `DYNAMIC_PARAMS`, включая S3, пока сохранён;
  удаление согласованных rows — этап 2, запрет legacy secret/static updates — этап 3.
- Defaults `ollama` / `qwen2.5:7b` / `tei` реализованы. Env и явные constructor
  kwargs пока сохраняют приоритет; seed берёт текущие значения, existing DB rows
  не перезаписывает. Новые endpoint defaults и HTTP pools пока только предложены.
- Baseline до изменений: `cd server && .venv/bin/pytest
  tests/test_domain_settings_hotreload.py tests/test_domain_profiles.py
  tests/test_domain_registry.py tests/test_domain_coverage_gaps.py
  tests/test_postgres_config_listener.py tests/test_cli_domain_bootstrap.py -q`
  → **166 passed**.
- RED: `cd server && .venv/bin/pytest tests/test_configuration_sources.py -q`
  → **7 failed, 2 passed** (нет деклараций источников/defaults и `build_defaults`).
  GREEN после реализации и дополнительных проверок → **12 passed**.
- Итоговая проверка тем же baseline + `tests/test_configuration_sources.py`
  и `tests/test_storage_lifecycle.py` → **187 passed**, 12 существующих
  deprecation warnings.
- Ruff check и format затронутых Python-файлов; `git diff --check`;
  domain/application import checks → без нарушений.
- Решение: этап 1 не активирует будущие dynamic keys, включая cost controls,
  и не удаляет legacy env/DB. Это сохраняет применение ресурсов до этапов 2–8.
  Миграция эффективных HTTP pool overrides относится к этапам 2/4.

### Этап 2. Безопасная миграция env и секретов

**Files:** presentation/cli/commands/config.py; infrastructure/config_migration.py и tests/test_config_env_migration.py (новые); существующие config repositories/UoW.

**Interface:** `python main.py config migrate-env --env-file PATH` — dry run по умолчанию; `--apply` выполняет перенос. Функция `async migrate_env_configuration(uow_factory, env_file: Path, apply: bool = False) -> None`.

- [x] Читать старый env установленным dotenv parser, независимо от будущей фильтрации Settings. Выводить имена ключей и действия, не секретные значения.
- [x] Existing DB rows сохранять. Отсутствующие rows заполнить валидированными legacy values с правильным domain_key. Для новых динамических адресов/pool/quotas/retention переносить действующие overrides, не заменять defaults.
- [x] S3 endpoint/bucket/region/credentials из БД сверить с окружением: несовпадение блокирует очистку с перечислением ключей. Не выбирать автоматически один секрет из двух разных значений.
- [x] Удалить согласованные legacy S3 DB rows только после проверки сохранения подключения. Перед правками создать защищённый локальный backup env (0600). DB commit предшествует атомарной замене env; ошибка записи оставляет безопасно повторяемый перенос. Missing credential блокирует apply.
- [x] Подготовить COMPOSE_PROFILES из прежнего deployment выбора TEI. Реестр признаёт dynamic ключ только после готовности соответствующего этапа; команда не удаляет env для ещё не внедрённого поведения.
- [x] Тесты: dry run без mutations, двойной apply без дублей, env/DB conflict, domain rows, S3 mismatch, commit/file-write failures, отсутствие секретов в logs.
- [x] Проверка: `cd server && .venv/bin/pytest tests/test_config_env_migration.py -q`.

**Результаты этапа 2 (2026-10-13):**

- Добавлена команда `python main.py config migrate-env --env-file PATH` (из `server/app/`).
  Без `--apply` выполняются чтение БД и отчёт об именах/действиях; записи,
  backup и изменения файлов отсутствуют. Команда освобождает DB connection при отказе.
- Existing `(key, domain_key)` не перезаписываются. Legacy values валидируются по
  типам Settings и ограничениям DYNAMIC_PARAMS; scoped chunk settings сохраняют
  legal/decree scope. INSERT использует существующий unique constraint: конкурентная
  вставка приводит к rollback, а не перезаписи чужого значения.
- Будущие keys сохраняются с category `migration_pending`, без включения в активный
  registry; frontend не включает эту категорию в editable tabs. Все dynamic строки
  env остаются до готовности этапов 3–12. HTTP pools сохраняют старую арифметику
  `// 2`, включая нечётные overrides; конфликт старого/нового имени блокирует перенос.
- S3 rows всех scopes удаляются только при точном совпадении с effective env/secrets
  и наличии обязательных credentials. Сетевой S3 probe не выполняется: проверяется
  сохранение подключения по значениям, live доступ нужно подтвердить в этапе 12.
- Секреты из PATH переносятся в соседний `PATH.secrets` (для `.env` — `.env.secrets`).
  Несовпадения между env, secret-файлом и экспортированным окружением блокируются.
  Пустой ключ неиспользуемого провайдера разрешён; обязательные DB/Redis/JWT/S3 и
  provider credentials проверяются по используемой конфигурации, включая DB provider.
- До DB mutations создаются уникальные соседние `*.backup-*` обоих существующих
  файлов с mode 0600. После commit выполняются fsync и атомарные замены: сначала
  secret-файл, затем env. Частичный отказ безопасен для повтора; обнаруженная правка
  оператора после чтения не перезаписывается. Secret-файл получает mode 0600.
  Backup/temp patterns добавлены в gitignore.
- Решения: продолжена общая ветка `configuration-management`; live apply выполняется
  только в согласованное окно этапа 12 с остановленными старыми writers. До этапа 3
  старый bootstrap ещё может снова seed S3 rows, поэтому production migration сейчас
  не запускалась. Dotenv interpolation отключена: unresolved dynamic/secret `${...}`
  требует явного значения, чтобы не сохранить данные из постороннего окружения.
- Baseline config/domain: **19 passed**. RED исходного мигратора: **15 failed**;
  дополнительные RED проверки: source conflict, secret permissions, missing provider,
  новые pool names, optional blank key и регистронезависимые env-имена.
  Итоговый migration/CLI набор: **29 passed** (`cd server && .venv/bin/pytest
  tests/test_config_env_migration.py -q`), 5 существующих deprecation warnings.
- `UV_OFFLINE=true task lock` обновил uv.lock: уже установленный python-dotenv 1.0.1
  объявлен прямой зависимостью, новые пакеты не добавлены. Ruff check/format затронутых
  файлов, `git diff --check`, domain/application import checks — без нарушений.
- Тесты с asyncio.to_thread внутри sandbox зависали в ожидании callback event loop;
  разрешённый запуск вне sandbox устранил зависание. Итоговая полная проверка:
  `cd server && .venv/bin/pytest -q` → **2719 passed, 46 skipped, 13 warnings**.
  Opt-in PostgreSQL/MinIO/Redis integrations не запускались; live доступ S3/БД
  и фактические production values не проверены.
- Отдельный финальный reviewer не запустился из-за usage limit сервиса. Выполнен
  самостоятельный review, найденные проблемы pool migration, optional credentials
  и регистронезависимых secret keys покрыты RED→GREEN тестами.
  Независимый review остаётся непроведённым.

Рабочие env/secret-файлы, БД, Compose services и Qdrant не изменялись.

### Этап 3. Единая загрузка конфигурации и границы управления

**Files:** config.py, infrastructure/initialization.py, infrastructure/ml/config/config_subscribers.py, composition/container.py, composition/infrastructure.py, composition/application.py, main.py, CLI config/worker/ingest; application/services/config_service.py; admin_config routes; config ports/adapters по необходимости.

**Interfaces:** async `seed_config_defaults(uow_factory)`, `load_config_from_db(uow_factory)`, `InfrastructureContainer.bootstrap_configuration()`; `ProviderConfigurationPort.validate_change(key: str, value: str) -> None`, adapter SettingsProviderConfiguration, DI в ConfigService.

- [x] Исключать готовые dynamic поля из env/dotenv sources; seed получать из defaults, не из уже мутированного singleton. Constructor kwargs/context overrides для tests/benchmark работают.
- [x] Запрет secret/static updates обеспечить в application/domain, не только в HTTP. Legacy DB rows/events не меняют secret env. Не использовать маскированное значение как новый секрет.
- [x] Global/domain bootstrap одинаков для API/worker/ML CLI; подписки до применения событий, ML calls после загрузки. Bootstrap admin не запускать из worker/CLI.
- [x] Secret/CORS/connection checks при старте сохранить, provider validation перенести после DB load. Обязательные seed/load/apply failures прерывают startup с cleanup.
- [x] Применять одну версию snapshot; пропущенный NOTIFY догонять существующим resync. Admin API показывает saved/applied version и ошибку применения без раскрытия credentials; определить минимальную process readiness через существующие heartbeat/health механизмы.
- [x] Тесты: DB beats legacy env, defaults при пустой БД, restart сохранения settings, DeepInfra production без TEI URL, missing key, malformed DB rows, lost NOTIFY, concurrent changes, worker/CLI cold start, cleanup после отказа.
- [x] Проверка: новые test_configuration_bootstrap.py/test_config_static_boundaries.py + существующие test_postgres_config_listener.py, test_domain_settings_hotreload.py, test_cli_domain_bootstrap.py.

**Результаты этапа 3 (2026-10-13):**

- Общая ветка `configuration-management`. Рабочие env/secret-файлы, БД, Compose
  services и Qdrant не изменялись. Пользовательские `.gitignore` и
  `docs/operations/configuration.md` сохранены отдельно.
- Env/dotenv фильтруются до декодирования для действующих DYNAMIC_PARAMS;
  constructor kwargs и benchmark context overrides сохранены. Seed получает
  объявленные defaults независимо от singleton и сохраняет existing DB rows.
  Endpoints/pools, quotas, upload/retention, cost controls и corpus ожидают своих этапов.
- S3 connection/credentials исключены из действующего registry. Application
  запрещает secret/static/pending updates, включая не-HTTP вызовы. Legacy events
  не меняют credentials/storage и не выводят deployment values в audit log.
- `seed_config_defaults`, `load_config_from_db` и
  `InfrastructureContainer.bootstrap_configuration()` объединяют global/domain
  загрузку API, worker, ingestion CLI и config resync. InfrastructureContainer
  владеет подписками; bootstrap admin запускается только в API.
- Security/CORS checks сохранены в Settings. Provider validation — после DB load,
  до ML, с обязательными API keys. DeepInfra production работает без TEI URL.
  Обязательные seed/load/validation/reaction failures прерывают startup с cleanup.
- Snapshot валидируется по каноническим определениям перед синхронным применением
  global/domain значений без await; обязательные реакции вызываются strict publish.
  Ошибка сохраняет предыдущие значения/applied version, readiness отрицательная.
  После bootstrap snapshot обязан быть полным. Версия — SHA-256 управляемых rows,
  без secrets/static values; новая схема БД не потребовалась.
- Admin-only `GET /admin/config/status` показывает saved/applied version, ready
  и безопасную ошибку текущего API-процесса. Это не подтверждение версии всех
  replicas/worker; координация обслуживающих процессов остаётся этапу 8.
  API `/ready` учитывает application status; worker проходит bootstrap до завершения
  Arq on_startup. Live readiness/heartbeat не проверялись.
- NOTIFY инициирует чтение актуального writer snapshot. Fetch/apply сериализованы
  asyncio lock, seed/update — общим PostgreSQL transaction advisory lock.
  DB transaction завершается перед apply/reactions. Сохранён periodic resync API,
  worker получает periodic resync каждые 60 с.
- Baseline: из server `.venv/bin/pytest tests/test_configuration_sources.py
  tests/test_postgres_config_listener.py tests/test_domain_settings_hotreload.py
  tests/test_cli_domain_bootstrap.py -q` → **46 passed**.
- RED bootstrap/boundary контрактов → **81 failed, 11 errors** (нет новых интерфейсов,
  действуют legacy env/update пути). Дополнительные status/domain completeness
  сценарии подтверждены RED→GREEN. Итоговая выборка новых tests + sources/listener/
  domain/CLI/container/LLM wiring/worker lifecycle → **189 passed**.
- Полный `.venv/bin/pytest -q` из server вне sandbox → **2840 passed, 46 skipped,
  13 существующих warnings**. Sandbox воспроизводит threadpool/asyncio.to_thread
  таймауты; opt-in PostgreSQL/MinIO/Redis integrations и нагрузка не запускались.
- `ruff check app tests`, format check 33 затронутых Python-файлов,
  `git diff --check`, domain/application import checks → без нарушений.
  `task lint` недоступен из-за snap launcher; его команды выполнены напрямую.
  `mypy app` → **6 существующих ошибок** в config_migration.py
  (137, 145, 176, 219, 292, 297). Архив исходного HEAD в /tmp дал те же ошибки;
  новых ошибок типов в этапе 3 нет.
- Независимый read-only reviewer нашёл оставшийся legacy config resync.
  Находка повышена с Minor до Important: ложный успех на malformed DB settings
  нарушал контракт этапа. Исправлено одним fix pass: два CLI теста
  **RED (2 failed) → GREEN**, затем полный suite выше. Команда применяет snapshot
  в собственном процессе и не обещает принудительный resync соседних процессов.
  Незакрытых находок reviewer нет; повторное ревью не запускалось.

### Этап 4. Динамические ML endpoints и явно названные HTTP pools

**Files:** infrastructure/ml/clients/factories.py, client_registry.py, tei_clients.py, deepinfra_clients.py, instructor_client.py, managed_llm.py; ml/config/config_subscribers.py; health/system_health_probe.py; benchmark judge/answer generators; config.py/DYNAMIC_PARAMS; client/src/pages/admin/settings.tsx.

**Target keys:** ollama_base_url, openrouter_base_url, tei_embed_url, tei_rerank_url, deepinfra_base_url; embedding_http_max_connections/embedding_http_max_keepalive_connections; rerank_http_max_connections/rerank_http_max_keepalive_connections.

- [x] Сохранить effective pool defaults 40/20 и 20/10. Валидировать positive max connections, nonnegative keepalive и keepalive <= connections. Убрать магическое деление /2 из factories.
- [x] Добавить в registry/DB/UI только вместе с реакциями. Endpoint change валидирует адрес/credential origin и подготавливает новый клиент. Cached health/auxiliary/benchmark clients тоже обновляются.
- [x] После успешного snapshot switch новые операции берут новый клиент, старые завершаются со старым; retired clients закрываются после освобождения, а не только на shutdown. Не закрыть async client в середине stream.
- [x] TEI embedding URL change проверяет identity обслуживаемой модели; смена модели/provider не смешивает вектора существующего индекса. Изменение адреса само по себе не означает разрешение переиндексации.
- [x] Тесты: URL/pool change реально меняет следующий request, active request survives, failures retain previous applied resource, repeated changes не копят connections; old pool migration preserves overrides; SSRF/credential-origin rejection.
- [x] Проверка: новые test_dynamic_ml_configuration.py; существующие tests для managed LLM, TEI/DeepInfra и client lifecycle.

**Результаты этапа 4 (2026-10-14):**

- Ветка `configuration-management`; рабочие env/secret-файлы, БД и сервисы
  не изменены. Пользовательские `.gitignore` и `docs/operations/configuration.md`
  сохранены отдельно. Следующий этап — 5 (rate limiting).
- Пять ML endpoints и четыре HTTP pool ключа включены в действующий registry,
  bootstrap/seed, DB validation и админку. Env/dotenv больше не управляют ими;
  явные Settings kwargs сохраняются. Defaults TEI — Docker адреса `:8080`.
  Эффективные pools сохранены: embedding **40/20**, reranker **20/10**;
  деление на 2 осталось только в offline legacy migration. Старые поля Settings
  удалены, parser миграции по-прежнему принимает их и сохраняет odd overrides.
- `ML_INTERNAL_ALLOWED_ORIGINS` — новый deployment-owned allowlist (статический
  env, не админка). У нового внутреннего endpoint должен совпадать exact origin;
  custom origins требуется явно включить **до bootstrap**. Облачные credentials
  разрешены только canonical HTTPS origins OpenRouter/DeepInfra. URLs с userinfo,
  query/fragment, недопустимой схемой и неизвестным origin отклоняются до сети.
  HTTP/SDK clients не следуют redirects. Каталог актуализирован: **147 полей**.
- Admin update сначала готовит/probes кандидат вне DB transaction, затем сверяет
  snapshot под configuration lock перед короткой записью. Resync повторно читает
  writer version после probes. Settings/domain и подготовленные resources
  публикуются без await; ошибка подготовки/reaction сохраняет прежние ресурсы.
  Constructors читают один copy snapshot, включая provider/endpoint/API key.
- Обновляется вся cached LLM family (generation/breadth/auxiliary/Instructor),
  embedding/reranker, private standalone benchmark registry и sync judge cache.
  Health probes читают применённые Settings. Private benchmark refresh
  сериализован, stale preparation освобождается и повторяется с новым snapshot
  (не более трёх попыток); старая generation не устанавливается после новой.
- Retired inference/LLM pools закрываются после освобождения leases, удаляются
  из retired registry и не ждут общего shutdown. Lease защищает active streams,
  auxiliary queue/wait_for, все embedding batches одной ingestion operation и
  sync judge admission/retries. Queued retrieval получает клиент после admission.
  Shutdown дожидается owned benchmark cleanup; pending cleanup tasks не остаются.
- TEI embedding endpoint проверяется через `/info` (model_id/model_sha/dtype/pooling)
  и настоящий embedding probe (configured dimension, finite values). При смене
  действующего endpoint immutable metadata должны совпасть с прежним сервером.
  Local weights с отсутствующим model_sha или несовпадающим configured model_id
  **не разрешают endpoint rotation**: нужны проверяемые serving metadata, а не
  догадка по имени каталога. Реальные модели/endpoints не проверялись.
  Изменение активной embedding identity/provider требует отдельной corpus migration,
  новый индекс автоматически не создаётся и ACL не меняется.
- Админка различает saved/applied/failed для текущего API process, обновляет status
  и показывает безопасную ошибку. Metadata глобальных/scoped declarations
  накладываются при listing поверх мигрированных rows, сохраняя DB values:
  действующие параметры не показывают старую надпись pending и получают min/max.
  Applied status не объявляется подтверждением всех replicas/worker (этап 8).
- Baseline config/bootstrap/migration/client lifecycle → **197 passed**;
  новые endpoint/pool/resource сценарии **RED: 23 failed, 8 passed → GREEN**.
  Characterization до изменения benchmark judge → **72 passed**, standalone
  generator/worker paths → **27 passed**. Отдельно подтверждены RED→GREEN:
  failed admin probe без commit, cached judge URL switch, stale writer version,
  ingestion batches, semaphore/threadpool admission, constructor snapshot/credentials,
  ожидание benchmark cleanup на shutdown. Итоговая focused выборка → **77 passed**.
- Независимый read-only reviewer: Important — stale concurrent standalone refresh;
  Minor — pending metadata в админке. Metadata повышена до Important по контракту
  корректного показа действующих controls. Обе находки воспроизведены
  **RED (2 failed) → GREEN**, исправлены одним pass. Повторное ревью не запускалось;
  отложенных findings нет. Model/corpus switching и live deploy относятся к другим
  этапам и не исключены из общего плана.
- Полный `.venv/bin/pytest -q` из server: **2890 passed, 46 skipped, 13 существующих warnings**.
  Тесты запускались вне sandbox из-за воспроизводимых threadpool stalls в sandbox.
  Opt-in live PostgreSQL/MinIO/Redis integrations и нагрузка не запускались.
- Ruff check `app tests`, format check всех затронутых Python-файлов,
  `npm run check` (Biome + TypeScript), `git diff --check`, обязательные
  domain/application import checks — без нарушений. `mypy app`: только **6
  существующих ошибок** в config_migration.py (138, 146, 177, 220, 293, 298),
  новых ошибок этапа 4 нет. Browser/live UI и реальные provider probes не запускались.

### Этап 5. Rate limiting через БД с настоящим обновлением

**Files:** infrastructure/rate_limit/policies.py, limiter.py, bucket_factory.py, composition/infrastructure.py, config subscribers, admin UI, config.py/DYNAMIC_PARAMS; существующие test_rate_limit_*.py и новый test_rate_limit_hotreload.py.

- [x] Перенести enabled, fail_open и все *_PER_* quotas в БД. REDIS connection, key prefix и max_buckets пока static. Validate quotas > 0; отключение — через enabled, а не неясную квоту 0.
- [x] Реализовать замещение policy snapshot/rebuild кэша limiter без удаления Redis sorted sets. Не раздавать новую полную квоту при каждом изменении настройки.
- [x] При увеличении окна rate policy сохранить достаточную историю: текущие minute/hour/day интервалы в этой задаче не менять; только их quota values.
- [x] Согласовать значения на API/worker, учитывать запросы во время update. Сохранить каталог policy enum, 429 envelope и headers.
- [x] Unit-проверки: lowering quota учитывает прежние запросы, increasing допускает дополнительный остаток, enabled/fail_open переключаются, два адаптера разделяют историю, concurrent update/acquire, Redis failure. Пройдены test_rate_limit_hotreload.py и policies/adapter/dependency/route_coverage; shared Redis моделируется fakeredis.
- [ ] Проверка на настоящем Redis в отдельном разрешённом test окружении. Unit-тест с двумя адаптерами не заменяет live multi-process проверку. Load tests не запускать автоматически.

**Результат этапа 5:** DynamicRateLimiter выбирает поколение policy на запрос, сохраняет имена Redis keys и историю при rotation, закрывает старый limiter после завершения его запросов. API/worker используют DB bootstrap и resync; доставка новой конфигурации между процессами асинхронная. Реальная Redis integration в текущем ревью и исправлении не запускалась.

### Этап 6. Upload и retention, показ фактической конфигурации

**Files:** config.py/DYNAMIC_PARAMS, presentation/api/dependencies.py, routes/documents.py и ingest.py, infrastructure/worker/cron.py; admin schemas/routes/UI; client nginx конфигурация; новые test_dynamic_operational_settings.py.

- [x] MAX_UPLOAD_SIZE_MB в БД: целое > 0; snapshot лимита на начале upload, без изменения лимита текущего запроса. Проверяется фактический объём чтения UploadFile, а не один Content-Length.
- [x] Согласовать с фиксированным upstream nginx hard ceiling: админка не разрешает значение больше поддерживаемого потолка либо явно требует отдельного deployment изменения. Потолок определить по действующему nginx config, не выдумывать число.
- [x] JOB_CLEANUP_DAYS в БД: целое >= 1; следующее cron execution использует новый retention. Active jobs не удаляются; уменьшение retention явно сообщает о последствиях в админке.
- [x] CURATOR_SCOPE_MAX_IDS не удалять и не дублировать ACL. DATA_DIR пока static; его дальнейшая судьба — этапы 9–11.
- [x] Админке показывать source, configured/applied value, режим применения, состояние/ошибку. Секреты — только наличие/статус. Read-only deployment поля не делать editable ради отображения.
- [x] Проверены test_dynamic_operational_settings.py и существующие config boundaries/ACL/curator tests: новый upload limit, запрет изменения deployment/secret settings, upload boundary, cleanup cutoff/active jobs, curator scope limit и ACL invariants.

**Результат этапа 6:** upload dependency создаёт неизменяемый лимит для обработчика; допустимо 1–63 MiB на файл при консервативном nginx ceiling 64 MiB на запрос с multipart metadata. Cleanup читает новый retention при каждом cron execution и удаляет только старые DONE/FAILED jobs. UI показывает configured/applied values и предупреждает о последствиях уменьшения retention. Browser/live upload через nginx не проверялся.

### Этап 7. Рабочие cost limits, без фиктивных переключателей

**Статус: частично выполнен.** Работает резервирование и учёт текстовых ManagedLLM-вызовов OpenRouter. Полный accounting embedding/rerank, structured calls и benchmark judges ещё не реализован; при включённых cost limits эти платные пути fail-closed. Их блокировка не считается завершённым учётом расходов.

**Files:** existing infrastructure/ml/usage_capture.py, managed_llm.py и usage metrics; application ports/services для budget, infrastructure Redis budget adapter, необходимые domain DTO/entities/repository и migrations для цен/usage, provider adapters; config registry/API/UI.

- [ ] Полный scope: currency USD, per authenticated principal, rolling hour/day, accounting платных chat/auxiliary/embedding/rerank/benchmark. Principal propagation и общая точка допуска реализованы; для unsupported путей требуется подтверждённый billing contract, bounded reservation и usage reconciliation перед разрешением вызовов.
- [x] Добавить управляемые в БД rates по provider/model с версией и единицей цены, используя Decimal/целые денежные единицы, не float arithmetic. Не вшивать изменяемые цены и не считать неизвестную стоимость нулевой. Existing token usage capture использовать как основу; embedding/rerank без usage требуют явного unit accounting.
- [x] COST_* defaults сохранить: enabled=false, hour=1 USD, day=5 USD; при enabled=true обе квоты > 0, неизвестная цена/неподдерживаемая тарификация блокирует платный вызов с понятной ошибкой.
- [x] Атомарно резервировать верхнюю оценку стоимости до внешнего вызова в shared Redis, затем идемпотентно сверять с фактическим usage. Установить конечный output budget для резерва. Сетевой timeout/cancellation после возможного списания не возвращает резерв как гарантированно бесплатный вызов; сохраняет pending accounting для reconciliation.
- [x] Usage ledger/идемпотентные operation IDs в PostgreSQL, recovery после worker retry/Redis restart; при недоступном budget backend включённый cost limiter fail-closed. Rate limiter fail_open не переносить автоматически на денежный лимит.
- [x] Полный scope должен отражаться в UI; если провайдер не позволяет получить/ограничить стоимость, не утверждать строгий потолок, сохранить unsupported status и не включать enforcement для такого пути. Этот этап — реальный use case, не перенос трёх чисел.
- [x] Тесты: параллельные запросы не резервируют сверх квоты, retry не списывает дважды, accounting всех поддерживаемых платных операций, неизвестная цена/usage, cancellation, quota changes, pending recovery и backend failure.
- [x] Если контракт стоимости конкретного provider требует уточнения, продолжать независимые этапы 8–11, но этап 7 не объявлять завершённым и controls не выдавать за работающие.

**Реализованный контракт (2026-10-15):** USD, отдельная rolling hour/day квота каждого `user:<id>`; задания наследуют инициатора, CLI использует явный `system:ingest-cli`. Цены — неизменяемые версии provider/model/unit в PostgreSQL, расчёт Decimal и целые nanodollars. PostgreSQL сериализует admission по principal, Redis выполняет атомарную проверку; после Redis restart история восстанавливается из ledger. Pending резерв не протухает до подтверждённого reconciliation. Квоты читаются из БД перед платным вызовом, независимо от доставки NOTIFY.

Поддерживается текстовый OpenRouter через ManagedLLM: проверка context bound по provider model catalogue, конечный output budget, ограничение provider price и одного физического HTTP send на резерв. Фактический usage закрывает резерв, неоднозначный исход сохраняет pending и response ID при наличии. В админке доступны публикация версии тарифа и reconciliation с подтверждающим основанием. Неизвестные цены, raw structured/benchmark judge и DeepInfra embedding/rerank при включённом лимите блокируются **до платного запроса**; их нельзя считать поддерживаемым измерением расходов. Локальные модели бесплатны. Автоматические SDK retries не получают повторного разрешения на тот же резерв.

### Этап 8. Управляемое переключение Qdrant и корпуса

**Files:** clients/client_registry.py и factories.py; repositories/vector/qdrant_vector_store_repository.py, qdrant_ops.py, embedding_identity.py; rag/rag_retrieval.py; BM25 persistence/invalidation; answer cache; health probes; config service/API/UI и migrations для минимального switch state при необходимости.

- [x] QDRANT_URL и COLLECTION_NAME в БД; при новой установке defaults Docker http://qdrant:6333 и company_docs, существующие значения мигрируются. Admin UI показывает endpoint/alias/physical collection/embedding identity и применение каждым обслуживающим процессом.
- [x] Подготовка проверяет доступ с env QDRANT_API_KEY (новый origin должен быть разрешён), существование выбранного корпуса, размерность, immutable identity, ACL payload и согласованность с document/chunk IDs PostgreSQL. Не выбирать произвольный чужой индекс по наличию коллекции.
- [x] Предпочесть стабильный collection alias и атомарное переключение физического корпуса. Первая миграция physical name в alias — отдельное проверенное действие; совпадение имени существующей physical collection не решать разрушительным rename/delete.
- [x] Qdrant resource+collection+BM25 scope+answer cache corpus namespace — единый snapshot. Ответы из старого corpus cache не выдавать после switch. Старые in-flight запросы завершаются на прежнем snapshot.
- [x] Для API/worker switch установить краткий barrier на новые ingestion writes, дождаться безопасной точки активных записей и readiness обоих процессов; при ошибке/недоступном процессе switch не объявлять успешным. Для CLI mutations выполнять ту же проверку версии/lock; не держать DB transaction во время сети.
- [x] Сохранить обе ACL barriers dense/sparse resolve. Новый индекс не создавать/не переиндексировать автоматически ради переключения.
- [x] Тесты: old client + new collection не смешиваются; invalid identity/ACL rejects; stale cache отсутствует; concurrent ingestion/switch/NOTIFY loss; failure rollback; ACL property tests.
- [x] Здесь не требуется распределённая транзакция между PG/Qdrant; нужен retryable журнал минимального switch состояния и фактическая проверка результата. Сбой посередине восстанавливается без потери writes.

**Реализованное переключение (2026-10-15):** выбирает существующий alias/physical collection, фиксируя физическую коллекцию на поколение; создание/перенацеливание alias и переиндексация остаются отдельными действиями. Новый endpoint должен быть явно внесён в deployment `ML_INTERNAL_ALLOWED_ORIGINS` до передачи API key. Проверяются immutable embedding identity, dimension и полное соответствие point IDs/content/ACL PostgreSQL. Холодный старт проверяет активную привязку и encoder; пустая новая установка допускает создание коллекции штатной ingestion.

Клиент Qdrant, BM25 и cache namespace публикуются одним snapshot; начатый запрос сохраняет прежнее поколение. PostgreSQL advisory barrier и write epoch защищают API/worker/CLI записи; подготовка проверяет фактическую готовность API и worker, финальная короткая транзакция повторно проверяет конфигурацию/корпус. Журнал preparing/committing/active/failed восстанавливается polling даже без NOTIFY; неподтверждённый committed switch блокирует записи до готовности процессов. Ошибка до commit оставляет старое поколение.

Результаты первоначальной реализации (до независимого ревью): полный unit suite — **2917 passed**, PostgreSQL integration — **6 passed**; Ruff check/format, client TypeScript и Biome — passed. На тот момент mypy содержал 7 ошибок в `infrastructure/config_migration.py` и `infrastructure/rate_limit/limiter.py`; они устранены при исправлении ревью ниже. `git diff --check` — passed.

Проверки: `test_cost_budget.py`, `test_corpus_switch.py`, полный unit suite и `tests/integration/test_configuration_cost_corpus_postgres.py` на отдельном PostgreSQL с локальным Qdrant и Redis Lua test adapter. Реальные платные вызовы OpenRouter/DeepInfra, live multi-host Qdrant и production миграции/перезапуски не выполнялись. Перед live вводом нужны применение двух новых Alembic migrations, deployment allowlist и опубликованные проверенные тарифы; strict accounting неподдерживаемых путей не заявляется.

**Исправления независимого ревью (2026-10-15):**

- Холодный полный DB bootstrap проверяет TEI `/info` и реальный embedding даже при совпадении URL с defaults: model_id должен совпадать с настройками, вектор должен иметь настроенную размерность и конечные значения. Ошибка прерывает bootstrap до готовности и запуска corpus control. Схема существующих embedding identity не меняется; это не доказательство соответствия старых векторов произвольно заменённым весам той же модели. Изменение весов/revision по-прежнему требует embedding migration.
- После BM25 invalidation новый запрос сначала загружает индекс, затем без await фиксирует согласованную пару Qdrant/BM25. Уже начатый запрос сохраняет старый snapshot. Если во время загрузки переключился корпус, stale download отбрасывается; BM25LoadError явно логируется, текущий запрос использует dense, следующий повторяет загрузку.
- Мигратор учитывает default FILE_BACKEND=s3 и без legacy S3 rows: отсутствие S3_ACCESS_KEY/S3_SECRET_KEY блокирует dry run/apply до backup, DB mutations и изменения env. Тесты переноса отдельных секретов используют явный FILE_BACKEND=local.
- Регрессионные тесты: **RED — 9 failed** до исправлений; исправления и связанные bootstrap/migration/BM25/rate tests — **198 passed**. Команда: из server `.venv/bin/pytest -q tests/test_dynamic_ml_configuration.py tests/test_corpus_switch.py tests/test_config_env_migration.py tests/test_rate_limit_hotreload.py tests/test_configuration_bootstrap.py tests/test_bm25_async_loading.py tests/test_bm25_lifecycle.py`.
- `.venv/bin/mypy app` — **Success: no issues found in 543 source files**; `.venv/bin/ruff check app tests` и format check семи изменённых Python-файлов — passed.
- Расширенная проверка из server: `timeout 120 .venv/bin/pytest -q -m 'not integration and not slow'` — **2913 passed, 89 deselected, 13 существующих deprecation warnings**, exit 0. Расширение до unit suite обосновано изменениями общего bootstrap и admission всех corpus operations; integration/slow в этом запуске исключены явно.
- Live integration, платные provider calls, env migration, Qdrant switch, deploy/recreate в этом исправлении не выполнялись. Проверка настоящего Redis и multi-host приёмка этапа 8 остаются открытыми; этап 7 остаётся частичным.

### Этап 9. История и результаты benchmark только в PostgreSQL

**Files:** infrastructure/benchmark/persistence.py, benchmark_history.py, runner.py, runner_adapter.py; application/services/benchmark_result_service.py; application/ports/benchmark_history.py; domain/repositories/benchmark_run_repository.py; repositories/benchmark/sqlalchemy_benchmark_run_repository.py; infrastructure/worker/sweep.py и benchmark worker; presentation/api/routes/benchmark.py/benchmark_admin.py.

- [x] Использовать существующие benchmark_runs/sweeps и DatabaseBenchmarkCheckpoints, не создавать параллельную таблицу истории. PostgreSQL — source of truth metrics/config/per-question results/baseline.
- [x] Отделить сериализацию отчёта от сохранения run: runner возвращает данные и не обязан вызывать локальный save_results/append JSONL. Worker сохраняет через application use case/UoW; retry идемпотентен по существующему run/job/sweep identity.
- [x] Regression и history читают DB repository/порт. Baseline выбирается для совместимого dataset/evaluation mode, а не просто последний JSONL record другого корпуса; сохранить существующие правила сравнения метрик и явно фиксировать выбранный run_id.
- [x] Импорт legacy JSONL и JSON отчётов: dry run, validation, provenance/hash для дедупликации; сводки без per-question не выдумывают детали. Перенести доступные результаты и baseline, старые файлы сохранять до подтверждения импорта.
- [x] Тесты: DB persistence/retry, summaries/history/regression без локального файла, partial legacy records, malformed input отчёт об ошибках, повторный import без дублей, concurrent sweeps не перепутаны.
- [x] Проверка существующих test_benchmark*.py/test_sweep*.py и новых test_benchmark_database_history.py. Characterization baseline на затрагиваемые benchmark публичные методы до изменения согласно AGENTS.md.

### Этап 10. Сохраняемые benchmark-артефакты в S3 и экспорт по запросу

**Files:** benchmark/persistence.py (serializers), application benchmark export service/port при необходимости, FileStorage port/adapter, benchmark routes/schemas, worker enqueue contracts, client benchmark download flows; CLI commands при наличии.

**Interfaces:** endpoint `GET /benchmark/results/{run_id}/export?format=csv|json` под существующим require_admin; CLI `benchmark export RUN_ID --format csv|json --out PATH`. Проверить реальный CLI registry: если команды benchmark отсутствуют, добавить только явный export command, не восстанавливать весь исторический CLI.

- [x] Для небольших выгрузок сериализовать DB run и отправить bytes/stream без файла в DATA_DIR. Размер/порог определить по текущим допустимым datasets и ограниченному буферу, не загружать неограниченный результат целиком.
- [x] Для сохраняемых отчётов/крупных артефактов использовать FileStorage/S3 prefix `benchmarks/runs/{run_id}/`; metadata object key/size/status хранить в DB. Sensitive reports доступны через auth endpoint или краткоживущую разрешённую ссылку, bucket не делать публичным.
- [x] DB commit результата предшествует export/upload. S3 сбой не превращает успешно сохранённый benchmark в failed evaluation; report получает failed/pending state и идемпотентный retry. Тот же run/format имеет стабильный ключ, без бесконечных дублей при retry.
- [x] Убрать пользовательский out_dir из server enqueue/storage contract. Legacy API out_dir временно принимать только прежнее default значение как deprecated/no-op; нестандартный путь отклонять с понятной ошибкой и новым export workflow. Обновить клиент/схемы; больше не давать HTTP API писать в произвольный каталог сервера.
- [x] Локальный файл появляется только по явному CLI --out. Не добавлять автоматические local JSON/CSV/JSONL exports по окончании web run.
- [x] Тесты: auth download, CSV/JSON корректность, export без writable DATA_DIR, S3 failure+retry не теряет DB result, legacy API compatibility, memory bounded large export.

### Этап 11. Убрать зависимость S3 backend от DATA_DIR

**Files:** config.py, benchmark/sweep_settings.py и sweep_engine.py, checkpoint.py/checkpoint_cleanup.py; worker checkpoint cleanup; routes/helpers/route_configs; storage/file_storage.py и parsing/OCR temp consumers при затрагивании; Docker mounts/env/docs.

- [x] Web sweep checkpoints остаются в существующей PostgreSQL реализации. Legacy local recovery оставить как явный import path/переходный режим, не обязательный root и не filesystem fallback, скрывающий отсутствие DB checkpoint.
- [x] Выделить исключительно локальные CLI export/checkpoint paths из default web runtime. Удалить серверные DATA_DIR consumers history/report/out_dir после этапов 9–10; cleanup управляет DB checkpoint retention и S3 артефактами по их назначению.
- [x] Для промежуточного файла использовать TemporaryDirectory/существующий безопасный temp mechanism с cleanup в finally/context, включая exception/cancellation. Если библиотеке не нужен pathname, передавать stream/bytes с bounded size. Blocking IO не выполнять в event loop.
- [x] Не переносить все документы/OCR в RAM и не удалять полезные temp файлы из принципа. Они disposable; восстановление после restart опирается на БД/S3, не на temp.
- [x] DATA_DIR не требуется при FILE_BACKEND=s3; local backend, если поддерживается, сохраняет статический root. Наличие поля с default не означает необходимость строки env: удалить только ненужные S3 строки/mounts, не volumes PostgreSQL/Qdrant/ML caches.
- [x] Тесты: S3 workflow с отсутствующим/readonly DATA_DIR; sweep retry другим worker восстанавливается из DB; exception/cancellation cleans temp; CLI --out работает; supported local storage сохраняет path/ACL validation.

**Проверено 2026-10-15 (этапы 9–11):** 352 целевых теста прошли: `test_benchmark*.py`, `test_sweep*.py`, worker/resource lifecycle, async local/S3 storage, `integration/test_benchmark_history_postgres.py` и `integration/test_benchmark_checkpoint_storage.py`. PostgreSQL проверен в отдельном временном кластере, включая конкурентные retry/sweeps, upgrade/downgrade новой миграции и восстановление sweep новым экземпляром worker без DATA_DIR. `task lint` (Ruff + mypy, 548 модулей), TypeScript typecheck, Biome затронутых клиентских файлов и `git diff --check` прошли. Запрещённых обратных импортов в затронутых Domain/Application файлах не найдено. Тесты с файловым I/O/threadpool и сокетами выполнялись вне sandbox: внутри него эти операции зависают/запрещены.

Реализация: миграция `r8h9i0j1k2l3_benchmark_reports` добавляет идентификаторы job/import, metadata артефактов и время обновления DB checkpoint. Regression возвращает `run_id` и `baseline_run_id`, учитывает dataset, evaluation mode и известную идентичность корпуса/dataset. JSON/CSV читают по одному вопросу и отдают блоки до 64 KiB; S3 multipart использует части по 5 MiB. Ошибка артефакта не отменяет сохранённую оценку; повтор — `POST /benchmark/results/{run_id}/artifacts/{csv|json}`. Скачать можно через admin-only `GET /benchmark/results/{run_id}/export?format=csv|json`, кнопки в таблице запусков или явный `benchmark export RUN_ID --format csv|json --out PATH`.

Legacy import: `benchmark import-legacy PATH --dataset DATASET` выполняет dry run; `--apply` сохраняет проверенные записи. Ошибки валидации блокируют apply, файлы не удаляются; дедупликация использует hash содержимого и dataset, provenance сохраняется в run. Размер одного входного файла ограничен 64 MiB. Для JSON без config/timestamp сохраняются доступные вопросы/метрики и mtime файла; неизвестные параметры корпуса не выдумываются. Сводки JSONL остаются без per-question details. Прежнее правило baseline — предыдущий совместимый запуск — сохранено; запись без известного корпуса не сравнивается с записью известного корпуса.

DATA_DIR остаётся статическим root только local backend; в web S3 runtime файловые checkpoints, reports и history не используются. Автоматические bind mounts `/code/project/data` удалены; для persistent local backend добавлен явный `docker-compose.local-storage.yml` (`LOCAL_DATA_DIR` задаёт host path). Web cleanup удаляет DB checkpoints успешных sweeps либо неактивные старше суток под блокировкой sweep; существующие локальные recovery adapters остаются только явно подключаемым переходным режимом. Сохранённые отчёты в S3 не удаляются вместе с disposable checkpoints.

**Границы приёмки:** настоящий S3/MinIO upload/download и live deployment не выполнялись; S3 failure/retry/multipart/cancellation проверены SDK-фейками. Рабочие env/БД/контейнеры не изменялись, новая миграция в рабочую БД не применена. Общая env migration, live приёмка и финальная Docker-проверка относятся к этапу 12. Отдельные commits не создавались, исходные незавершённые правки сохранены.

### Этап 12. Docker, сокращение env, документация и финальная проверка

**Files:** Taskfile.yml, docker-compose*.yml, scripts/operations/compose.sh, deploy.sh, env templates, client/.env, README.md, AGENTS.md (только устаревшие инструкции), migration CLI и этот файл.

- [ ] Убрать ML_PROVIDER-based profiles из Task и production operations helper; использовать COMPOSE_PROFILES. task pull-model требует явного MODEL и не читает удалённый LLM_MODEL. task env создаёт secret template без перезаписи существующих файлов, secret mode 0600.
- [ ] Env example содержит обязательный deployment/bootstrap набор, без копии dynamic каталога/defaults. Secret example — единственное место ключей. VITE_API_URL остаётся client build setting.
- [ ] Сохранить нестандартные static overrides, CORS, FORWARDED_ALLOW_IPS, REDIS_USER, S3 параметры/секреты и Docker image selection. Удалять одинаковые defaults только после проверки всех потребителей. Legacy keys удалить с обновлением README.
- [ ] Перед live переходом согласовать окно, backup PostgreSQL/S3 и protected env snapshots, остановить старых writers. Dry run migration → сверка conflict → apply → recreate new server/worker → readiness/applied settings/benchmark exports. Не делать deploy/реальные DB mutations без авторизации этой сессии.
- [ ] Финальная проверка: task lint; ruff format --check затронутых файлов; task test; git diff --check; обязательные domain/application import checks AGENTS.md (пустой результат). Значения секретов и env backups не попадали в diff/logs.
- [ ] `docker compose config --quiet` с тестовыми env/secrets, profile tei и без него; `task --list`. Не печатать полный rendered compose с secrets. Реальные сервисы/loadtests не запускать автоматически.
- [ ] В согласованном test/live окружении: динамический update API→worker, client switching с active requests, upload/rate/cost enforcement, прежний S3 corpus, history/regression/export без DATA_DIR. Сохранить фактические результаты и оставшиеся limitations.

## 6. Полный каталог параметров

Каталог ниже включает все 144 исходных поля Settings, 8 дополнительных доменных DB defaults, Docker/deployment и legacy keys. Два исходных HTTP_POOL_* заменены четырьмя целевыми ключами. Default — из текущего кода, кроме явно помеченных предлагаемых значений. Возможность hot reload становится контрактом только после реализации указанного этапа.

## Все поля Settings

В строках БД используется имя ключа в админке; в строках env/secrets — имя переменной. Строки, перенесённые в БД после обсуждения, используют новые целевые имена. Default — из текущего кода, кроме явно помеченных предлагаемых значений.

| Параметр | Целевой источник | Default | Применение | Действие при сокращении env |
|---|---|---|---|---|
| DB_HOST | env | — (обязательное поле) | Пересоздать потребителей | Явно задать в env |
| DB_PORT | env | — (обязательное поле) | Пересоздать потребителей | Явно задать в env |
| DB_USER | env | — (обязательное поле) | Пересоздать потребителей | Явно задать в env |
| DB_PASSWORD | secrets | Нет рабочего default; задать секрет | Пересоздать потребителей | Оставить только в secrets; не помещать в БД |
| DB_NAME | env | — (обязательное поле) | Пересоздать потребителей | Явно задать в env |
| DB_SLAVE_HOSTS | env | `""` | Пересоздать потребителей | Default в коде; env только при необходимом переопределении |
| DB_SLAVE_PORTS | env | `""` | Пересоздать потребителей | Default в коде; env только при необходимом переопределении |
| DB_POOL_SIZE | env | `40` | Пересоздать потребителей | Default в коде; env только при необходимом переопределении |
| DB_MAX_OVERFLOW | env | `20` | Пересоздать потребителей | Default в коде; env только при необходимом переопределении |
| DB_POOL_RECYCLE_SEC | env | `1800` | Пересоздать потребителей | Default в коде; env только при необходимом переопределении |
| DB_STATEMENT_TIMEOUT_MS | env | `600000` | Пересоздать потребителей | Default в коде; env только при необходимом переопределении |
| DB_IDLE_IN_TRANSACTION_TIMEOUT_MS | env | `120000` | Пересоздать потребителей | Default в коде; env только при необходимом переопределении |
| qdrant_url | БД | Сейчас нет; для нового Docker-окружения предлагается `http://qdrant:6333` | Управляемое переключение API/worker (этап 8) | Перенести существующее значение; убрать env только после реализации этапа |
| QDRANT_API_KEY | secrets | Нет рабочего default; задать секрет | Пересоздать потребителей | Оставить только в secrets; не помещать в БД |
| collection_name | БД | Сейчас нет; для нового Docker-окружения предлагается `company_docs` | Проверка корпуса и переключение alias (этап 8) | Перенести существующее значение; убрать env только после реализации этапа |
| QDRANT_TIMEOUT | env | `10` | Пересоздать потребителей | Default в коде; env только при необходимом переопределении |
| QDRANT_SEARCH_MAX_CONCURRENT | env | `16` | Пересоздать потребителей | Default в коде; env только при необходимом переопределении |
| QDRANT_WRITE_MAX_CONCURRENT | env | `8` | Пересоздать потребителей | Default в коде; env только при необходимом переопределении |
| RAG_RETRIEVAL_SCORE_THRESHOLD | env | `0.0` | Пересоздать потребителей | Default в коде; env только при необходимом переопределении |
| ollama_base_url | БД | `http://localhost:11434` | Замена LLM-клиентов (этап 4) | Перенести существующее значение; убрать env только после реализации этапа |
| llm_provider | БД | `ollama` (реализовано на этапе 1) | Hot reload¹ | Управлять через админку; убрать из env после сверки |
| llm_model | БД | `qwen2.5:7b` (реализовано на этапе 1) | Hot reload¹ | Управлять через админку; убрать из env после сверки |
| llm_temperature | БД | `0.1` | Hot reload¹ | Управлять через админку; убрать из env после сверки |
| llm_top_p | БД | `0.9` | Hot reload¹ | Управлять через админку; убрать из env после сверки |
| llm_num_ctx_narrow | БД | `8192` | Hot reload¹ | Управлять через админку; убрать из env после сверки |
| llm_num_ctx_broad | БД | `16384` | Hot reload¹ | Управлять через админку; убрать из env после сверки |
| llm_num_predict_narrow | БД | `400` | Hot reload¹ | Управлять через админку; убрать из env после сверки |
| llm_num_predict_broad | БД | `2048` | Hot reload¹ | Управлять через админку; убрать из env после сверки |
| LLM_MAX_CONCURRENT | env | `16` | Пересоздать потребителей | Default в коде; env только при необходимом переопределении |
| LLM_GENERATION_MAX_CONCURRENT | env | `12` | Пересоздать потребителей | Default в коде; env только при необходимом переопределении |
| LLM_AUXILIARY_MAX_CONCURRENT | env | `4` | Пересоздать потребителей | Default в коде; env только при необходимом переопределении |
| LLM_REQUEST_TIMEOUT | env | `120` | Пересоздать потребителей | Default в коде; env только при необходимом переопределении |
| LLM_AUXILIARY_TIMEOUT | env | `30` | Пересоздать потребителей | Текущая строка отличается от default; сохранить переопределение |
| EMBEDDING_MAX_CONCURRENT | env | `16` | Пересоздать потребителей | Default в коде; env только при необходимом переопределении |
| INGESTION_MAX_CONCURRENT | env | `8` | Пересоздать потребителей | Default в коде; env только при необходимом переопределении |
| RERANKER_MAX_CONCURRENT | env | `8` | Пересоздать потребителей | Default в коде; env только при необходимом переопределении |
| BM25_SEARCH_MAX_CONCURRENT | env | `8` | Пересоздать потребителей | Default в коде; env только при необходимом переопределении |
| SEMAPHORE_ACQUIRE_TIMEOUT | env | `30` | Пересоздать потребителей | Default в коде; env только при необходимом переопределении |
| BENCHMARK_MAX_CONCURRENT | env | `10` | Пересоздать потребителей | Default в коде; env только при необходимом переопределении |
| BENCHMARK_JUDGE_MAX_CONCURRENT | env | `3` | Пересоздать потребителей | Default в коде; env только при необходимом переопределении |
| BENCHMARK_JUDGE_GROUPED_ENABLED | env | `true` | Пересоздать потребителей | Default в коде; env только при необходимом переопределении |
| BENCHMARK_JUDGE_INITIAL_TOKENS | env | `2048` | Пересоздать потребителей | Default в коде; env только при необходимом переопределении |
| BENCHMARK_JUDGE_RETRY_TOKENS | env | `8192` | Пересоздать потребителей | Default в коде; env только при необходимом переопределении |
| BENCHMARK_JUDGE_MODEL | env | `""` | Пересоздать потребителей | Default в коде; env только при необходимом переопределении |
| OPENROUTER_API_KEY | secrets | Нет рабочего default; задать секрет | Пересоздать потребителей | Оставить только в secrets; не помещать в БД |
| openrouter_base_url | БД | `https://openrouter.ai/api/v1` | Замена LLM-клиентов (этап 4) | Перенести существующее значение; убрать env только после реализации этапа |
| openrouter_model | БД | `qwen/qwen-2.5-7b-instruct` | Hot reload¹ | Управлять через админку; убрать из env после сверки |
| ml_provider | БД | `tei` (реализовано на этапе 1) | Hot reload¹ + проверка embedding identity² | Управлять через админку; убрать из env после сверки |
| tei_embed_url | БД | `http://tei-embed:8080` (этап 4) | Замена клиента; проверка embedding identity (этап 4) | Перенести существующее значение; убрать env только после реализации этапа |
| TEI_EMBED_MODEL | env | `BAAI/bge-m3` | Пересоздать потребителей; миграция индекса при смене² | Default в коде; env только при необходимом переопределении |
| EMBEDDING_REVISION | env | `v1` | Пересоздать потребителей; миграция индекса при смене² | Default в коде; env только при необходимом переопределении |
| tei_rerank_url | БД | `http://tei-rerank:8080` (этап 4) | Замена reranker-клиента (этап 4) | Перенести существующее значение; убрать env только после реализации этапа |
| DEEPINFRA_API_KEY | secrets | Нет рабочего default; задать секрет | Пересоздать потребителей | Оставить только в secrets; не помещать в БД |
| ML_INTERNAL_ALLOWED_ORIGINS | env | `["http://localhost:11434", "http://ollama:11434", "http://tei-embed:8080", "http://tei-rerank:8080"]` | Пересоздать потребителей | Deployment allowlist для внутренних ML origins. До миграции добавить нестандартные origins явно; админка не может расширить список |
| deepinfra_base_url | БД | `https://api.deepinfra.com/v1` | Замена embedding/reranker-клиентов (этап 4) | Перенести существующее значение; убрать env только после реализации этапа |
| deepinfra_embed_model | БД | `BAAI/bge-m3` | Hot reload¹ + проверка embedding identity² | Управлять через админку; убрать из env после сверки |
| deepinfra_rerank_model | БД | `Qwen/Qwen3-Reranker-0.6B` | Hot reload¹ | Управлять через админку; убрать из env после сверки |
| embedding_http_max_connections | БД | `40`; сохраняет текущий эффективный лимит | Замена HTTP-клиента (этап 4) | Заменяет HTTP_POOL_MAX_CONNECTIONS; убрать неявное деление reranker на 2 |
| rerank_http_max_connections | БД | `20`; сохраняет текущий эффективный лимит | Замена HTTP-клиента (этап 4) | Заменяет HTTP_POOL_MAX_CONNECTIONS; убрать неявное деление reranker на 2 |
| embedding_http_max_keepalive_connections | БД | `20`; сохраняет текущий эффективный лимит | Замена HTTP-клиента (этап 4) | Заменяет HTTP_POOL_MAX_KEEPALIVE; убрать неявное деление reranker на 2 |
| rerank_http_max_keepalive_connections | БД | `10`; сохраняет текущий эффективный лимит | Замена HTTP-клиента (этап 4) | Заменяет HTTP_POOL_MAX_KEEPALIVE; убрать неявное деление reranker на 2 |
| ocr_engine | БД | `paddleocr` | Hot reload¹ | Управлять через админку; убрать из env после сверки |
| ocr_enabled | БД | `true` | Hot reload¹ | Управлять через админку; убрать из env после сверки |
| ocr_lang_paddle | БД | `ru` | Hot reload¹ | Управлять через админку; убрать из env после сверки |
| ocr_lang_surya | БД | `['ru', 'en']` | Hot reload¹ | Управлять через админку; убрать из env после сверки |
| ocr_dpi | БД | `300` | Hot reload¹ | Управлять через админку; убрать из env после сверки |
| ocr_min_chars | БД | `40` | Hot reload¹ | Управлять через админку; убрать из env после сверки |
| retriever_fetch_k | БД | `15` | Hot reload¹ | Управлять через админку; убрать из env после сверки |
| retriever_top_k | БД | `4` | Hot reload¹ | Управлять через админку; убрать из env после сверки |
| retriever_fetch_k_broad | БД | `40` | Hot reload¹ | Управлять через админку; убрать из env после сверки |
| retriever_top_k_broad | БД | `10` | Hot reload¹ | Управлять через админку; убрать из env после сверки |
| history_window | БД | `8` | Hot reload¹ | Управлять через админку; убрать из env после сверки |
| rerank_min_score | БД | `0.15` | Hot reload¹ | Управлять через админку; убрать из env после сверки |
| rerank_score_gap_ratio | БД | `0.1` | Hot reload¹ | Управлять через админку; убрать из env после сверки |
| source_min_score | БД | `0.3` | Hot reload¹ | Управлять через админку; убрать из env после сверки |
| citation_filter_enabled | БД | `true` | Hot reload¹ | Управлять через админку; убрать из env после сверки |
| exact_ref_sparse_boost | БД | `2.5` | Hot reload¹ | Управлять через админку; убрать из env после сверки |
| DOCUMENT_DOMAIN_MARKER_THRESHOLD | env | `2.0` | Пересоздать потребителей | Default в коде; env только при необходимом переопределении |
| chunk_size | БД | `550` | Hot reload¹ | Управлять через админку; убрать из env после сверки |
| chunk_overlap | БД | `200` | Hot reload¹ | Управлять через админку; убрать из env после сверки |
| legal_chunk_size [domain=legal] | БД | `1000` | Hot reload¹ | Управлять через админку; убрать из env после сверки |
| legal_chunk_overlap [domain=legal] | БД | `250` | Hot reload¹ | Управлять через админку; убрать из env после сверки |
| decree_chunk_size [domain=decree] | БД | `1200` | Hot reload¹ | Управлять через админку; убрать из env после сверки |
| decree_chunk_overlap [domain=decree] | БД | `200` | Hot reload¹ | Управлять через админку; убрать из env после сверки |
| embed_batch_size | БД | `32` | Hot reload¹ | Управлять через админку; убрать из env после сверки |
| EMBED_DIM | env | `1024` | Пересоздать потребителей; миграция индекса при смене² | Default в коде; env только при необходимом переопределении |
| hybrid_enabled | БД | `true` | Hot reload¹ | Управлять через админку; убрать из env после сверки |
| bm25_fetch_k | БД | `25` | Hot reload¹ | Управлять через админку; убрать из env после сверки |
| rrf_k | БД | `30` | Hot reload¹ | Управлять через админку; убрать из env после сверки |
| dense_weight | БД | `1.5` | Hot reload¹ | Управлять через админку; убрать из env после сверки |
| sparse_weight | БД | `0.5` | Hot reload¹ | Управлять через админку; убрать из env после сверки |
| CURATOR_SCOPE_MAX_IDS | env | `1000` | Пересоздать потребителей | Default в коде; env только при необходимом переопределении |
| relevance_gate_enabled | БД | `true` | Hot reload¹ | Управлять через админку; убрать из env после сверки |
| condense_enabled | БД | `false` | Hot reload¹ | Управлять через админку; убрать из env после сверки |
| decomposition_enabled | БД | `false` | Hot reload¹ | Управлять через админку; убрать из env после сверки |
| rolling_summary_enabled | БД | `true` | Hot reload¹ | Управлять через админку; убрать из env после сверки |
| cache_enabled | БД | `true` | Hot reload¹ | Управлять через админку; убрать из env после сверки |
| pii_redaction_enabled | БД | `true` | Hot reload¹ | Управлять через админку; убрать из env после сверки |
| REDIS_HOST | env | `redis` | Пересоздать потребителей | Текущая строка совпадает с default; кандидат на удаление³ |
| REDIS_PORT | env | `6379` | Пересоздать потребителей | Текущая строка совпадает с default; кандидат на удаление³ |
| REDIS_USER | env | `""` | Пересоздать потребителей | Текущая строка отличается от default; сохранить переопределение |
| REDIS_PASSWORD | secrets | Нет рабочего default; задать секрет | Пересоздать потребителей | Оставить только в secrets; не помещать в БД |
| REDIS_DB | env | `0` | Пересоздать потребителей | Текущая строка совпадает с default; кандидат на удаление³ |
| WORKER_MAX_CONCURRENT | env | `4` | Пересоздать потребителей | Текущая строка отличается от default; сохранить переопределение |
| REDIS_SOCKET_TIMEOUT | env | `10` | Пересоздать потребителей | Default в коде; env только при необходимом переопределении |
| REDIS_MAX_CONNECTIONS | env | `50` | Пересоздать потребителей | Default в коде; env только при необходимом переопределении |
| FILE_BACKEND | env | `s3` | Пересоздать потребителей | Текущая строка совпадает с default; кандидат на удаление³ |
| S3_ENDPOINT | env | `http://minio:9000` | Пересоздать потребителей | Оставить в env; сейчас также в реестре БД — сверить и убрать |
| S3_BUCKET | env | `rag-documents` | Пересоздать потребителей | Оставить в env; сейчас также в реестре БД — сверить и убрать |
| S3_ACCESS_KEY | secrets | Нет рабочего default; задать секрет | Пересоздать потребителей | Оставить только в secrets; не помещать в БД; сейчас также в реестре БД — сверить и убрать |
| S3_SECRET_KEY | secrets | Нет рабочего default; задать секрет | Пересоздать потребителей | Оставить только в secrets; не помещать в БД; сейчас также в реестре БД — сверить и убрать |
| S3_REGION | env | `us-east-1` | Пересоздать потребителей | Оставить в env; сейчас также в реестре БД — сверить и убрать |
| S3_MAX_CONCURRENT | env | `10` | Пересоздать потребителей | Default в коде; env только при необходимом переопределении |
| SUPPORTED_EXTENSIONS | env | `('.pdf', '.docx', '.doc', '.rtf', '.md', '.txt')` | Пересоздать потребителей | Default в коде; env только при необходимом переопределении |
| max_upload_size_mb | БД | `50` | Новые запросы; согласование с nginx (этап 6) | Перенести существующее значение; убрать env только после реализации этапа |
| DATA_DIR | env | `/code/project/data` | Пересоздать потребителей | После этапов 9–11 не нужен S3 backend; оставить только для поддерживаемого local backend |
| UPLOADS_PREFIX | env | `uploads/` | Пересоздать потребителей | Текущая строка совпадает с default; кандидат на удаление³ |
| ALLOWED_ORIGINS | env | — (обязательное поле) | Пересоздать потребителей | Явно задать в env |
| FORWARDED_ALLOW_IPS | env | `127.0.0.1` | Пересоздать потребителей | Default в коде; env только при необходимом переопределении |
| JWT_SECRET_KEY | secrets | Нет рабочего default; задать секрет | Пересоздать потребителей | Оставить только в secrets; не помещать в БД |
| JWT_ALGORITHM | env | `HS256` | Пересоздать потребителей | Default в коде; env только при необходимом переопределении |
| JWT_EXPIRE_MINUTES | env | `1440` | Пересоздать потребителей | Текущая строка совпадает с default; кандидат на удаление³ |
| ADMIN_EMAIL | env | `""` | Пересоздать потребителей | Текущая строка отличается от default; сохранить переопределение |
| ADMIN_PASSWORD | secrets | Нет рабочего default; задать секрет | Пересоздать потребителей | Оставить только в secrets; не помещать в БД |
| TIMEZONE | env | `UTC` | Пересоздать потребителей | Текущая строка совпадает с default; кандидат на удаление³ |
| VERSION | Код / метаданные | `""` | При запуске / назначении метаданных | Не включать в env/админку; читается из корневого VERSION, fallback 0.0.0 |
| SERVICE_START_DATETIME | Код / метаданные | `""` | При запуске / назначении метаданных | Не включать в env/админку; сейчас пустое значение, это не настройка пользователя |
| STAGE | env | `development` | Пересоздать потребителей | Текущая строка совпадает с default; кандидат на удаление³ |
| job_cleanup_days | БД | `30` | Следующая cron-очистка (этап 6) | Перенести существующее значение; убрать env только после реализации этапа |
| stuck_job_timeout_minutes | БД | `30` | Hot reload¹ | Управлять через админку; убрать из env после сверки |
| stale_pending_timeout_minutes | БД | `30` | Hot reload¹ | Управлять через админку; убрать из env после сверки |
| rate_limit_enabled | БД | `true` | Новые запросы; обновление limiter (этап 5) | Перенести существующее значение; убрать env только после реализации этапа |
| rate_limit_fail_open | БД | `true` | Новые запросы; обновление limiter (этап 5) | Перенести существующее значение; убрать env только после реализации этапа |
| RATE_LIMIT_REDIS_PREFIX | env | `rag:ratelimit:v1:` | Пересоздать потребителей | Default в коде; env только при необходимом переопределении |
| rate_limit_login_per_minute | БД | `5` | Новые запросы; сохранить Redis-историю (этап 5) | Перенести существующее значение; убрать env только после реализации этапа |
| rate_limit_chat_per_minute | БД | `10` | Новые запросы; сохранить Redis-историю (этап 5) | Перенести существующее значение; убрать env только после реализации этапа |
| rate_limit_chat_per_day | БД | `100` | Новые запросы; сохранить Redis-историю (этап 5) | Перенести существующее значение; убрать env только после реализации этапа |
| rate_limit_upload_per_hour | БД | `20` | Новые запросы; сохранить Redis-историю (этап 5) | Перенести существующее значение; убрать env только после реализации этапа |
| rate_limit_search_per_minute | БД | `60` | Новые запросы; сохранить Redis-историю (этап 5) | Перенести существующее значение; убрать env только после реализации этапа |
| rate_limit_write_per_minute | БД | `60` | Новые запросы; сохранить Redis-историю (этап 5) | Перенести существующее значение; убрать env только после реализации этапа |
| rate_limit_benchmark_per_hour | БД | `60` | Новые запросы; сохранить Redis-историю (этап 5) | Перенести существующее значение; убрать env только после реализации этапа |
| RATE_LIMIT_MAX_BUCKETS | env | `10000` | Пересоздать потребителей | Default в коде; env только при необходимом переопределении |
| cost_rate_limit_enabled | БД | `false` | Только после рабочего контроля расходов (этап 7) | Перенести существующее значение; убрать env только после реализации этапа |
| cost_hourly_limit | БД | `1.0` | Только после рабочего контроля расходов (этап 7) | Перенести существующее значение; убрать env только после реализации этапа |
| cost_daily_limit | БД | `5.0` | Только после рабочего контроля расходов (этап 7) | Перенести существующее значение; убрать env только после реализации этапа |
| LLM_BREAKER_FAIL_MAX | env | `5` | Пересоздать потребителей | Default в коде; env только при необходимом переопределении |
| LLM_BREAKER_TIMEOUT_DURATION | env | `300` | Пересоздать потребителей | Default в коде; env только при необходимом переопределении |

¹ Реестр декларирует hot reload, но включение записи в реестр само по себе не доказывает обновление каждого кэшированного клиента. При реализации проверить subscribers, server/worker и CLI. Новые параметры без поддержки применения не переносить в БД.

² Смена embedding-модели/размерности/revision или провайдера не должна автоматически ломать существующий индекс. Следовать [процессу миграции embedding](embedding-migration.md); одного обновления settings недостаточно. Для OCR/chunking изменения относятся к последующей обработке, уже сохранённые документы не переобрабатываются автоматически.

³ Совпадение сравнивается с текущим локальным env без публикации его значений. Удаление допустимо лишь после проверки Compose, scripts, CI и остальных потребителей; разные default в Python и Docker не считать одинаковыми.

## Дополнительные настройки доменов в БД

Эти ключи не являются полями Settings и не должны появляться в env. Defaults определены в доменных профилях и общем `domain/domain_profile/base.py`. Для профиля general дополнительных defaults нет.

| Ключ БД | domain_key | Default | Применение |
|---|---|---|---|
| max_unit_chars | legal | 1500 | Hot reload; последующее разбиение документов |
| fingerprint_min_articles | legal | 2 | Hot reload; последующая классификация |
| classification_threshold | legal | 1.5 | Hot reload; последующая классификация |
| effective_date_auto_trust_threshold | legal | 0.85 | Hot reload; последующая обработка дат |
| max_unit_chars | decree | 1200 | Hot reload; последующее разбиение документов |
| fingerprint_min_points | decree | 1 | Hot reload; последующая классификация |
| classification_threshold | decree | 2.0 | Hot reload; последующая классификация |
| effective_date_auto_trust_threshold | decree | 0.85 | Hot reload; последующая обработка дат |

## Docker, сборка и frontend

Эти параметры не заменяются записями config_parameters. CLI-флаги операций и отдельный каталог параметров backup/monitoring не являются настройками backend Settings и не переносятся в админку в этой задаче.

| Параметр | Источник | Текущее поведение/default | Применение / целевое действие |
|---|---|---|---|
| BUILD_TARGET | server/.env / deployment | cpu в шаблоне и production helper | Сборка/выбор образа; сохранить |
| STAGE | server/.env / deployment | development в Settings/шаблоне; production для Docker target в deploy | Различать Docker stage и AppStage; production helper задаёт APP_STAGE=prod |
| APP_STAGE | deployment env | Нет обязательного значения; Compose fallback STAGE/development | Env приложения STAGE: development/staging/prod; оставить явным для production |
| SERVER_IMAGE | deployment env | rag-server:${STAGE}-${BUILD_TARGET}-${VERSION} | Pull/recreate; сохранить override для GHCR |
| CLIENT_IMAGE | deployment env | rag-client:${VERSION} | Pull/recreate; сохранить при использовании |
| VERSION | Файл VERSION / release env | Task читает файл; production helper допускает override | Метаданные релиза и теги Docker; не настройка админки |
| COMPOSE_PROFILES | deployment env | Предлагается tei для локального TEI; без профиля для облачного ML | Целевой выбор дополнительных сервисов; заменить зависимость Task/operations от ML_PROVIDER |
| GPU_COUNT | deployment env | 1 в docker-compose.gpu.yml | Пересоздать GPU-сервисы; env только при изменении |
| DOCKER_MTU | deployment env | 1500 | Применяется к Docker network при создании; не обычным restart контейнера |
| TRUSTED_PROXY_CIDR | deployment env | 127.0.0.1 в Compose; deploy требует явное значение | Пересоздать client; не путать с FORWARDED_ALLOW_IPS backend |
| VITE_API_URL | client/.env / frontend build | /api в шаблоне и runtime Compose | Изменение адреса SPA требует пересборки frontend; runtime client также получает переменную |
| DEPLOY_DIR | deployment/operations env | Корень проекта, вычисленный скриптом | Следующий запуск deployment/operations |
| DEPLOY_WAIT_TIMEOUT | deployment env | 600 секунд | Следующая deployment-команда |
| DEPLOY_MIGRATION_TIMEOUT | deployment env | 300 секунд | Следующий запуск миграций в deploy |
| DEPLOY_SCHEMA_BACKWARD_COMPATIBLE | deployment env | false | Явное разрешение rollback после изменения схемы; сохранить операционное назначение |

## Старые и вычисляемые параметры

| Параметр | Статус | Действие |
|---|---|---|
| EMBED_MODEL | Нет поля Settings и активного потребителя в просмотренном коде; осталось упоминание в README | Удалить legacy строку и обновить README; не считать алиасом TEI_EMBED_MODEL/DEEPINFRA_EMBED_MODEL |
| RERANK_MODEL | Нет поля Settings и активного потребителя в просмотренном коде; осталось упоминание в README | Удалить legacy строку и обновить README; провайдерские модели имеют другие канонические ключи |
| RERANK_DEVICE | Нет поля Settings и активного потребителя в просмотренном коде | Удалить legacy строку; GPU-настройки остаются в Docker |
| db_slave_hosts_list / db_slave_ports_list | Вычисляются из DB_SLAVE_HOSTS / DB_SLAVE_PORTS | Не создавать отдельные env-переменные |
| redis_url | Вычисляется из REDIS_* и включает credentials | Не вводить второй источник URL и не публиковать его значение |
| allowed_origins_list | Вычисляется из ALLOWED_ORIGINS | Не создавать отдельную настройку |
| tz | Вычисляется из TIMEZONE | Не создавать отдельную настройку |
| uptime_seconds | Вычисляется из service_start_datetime | Служебное состояние процесса |


## 7. Приёмка и откат

- [ ] Для каждого поля определён один источник, bootstrap до ML одинаков во всех execution paths.
- [ ] Dynamic keys удалены из env только после migration; existing DB values и static overrides сохранены.
- [ ] Нет secrets в config_parameters/admin responses/logs; old DB rows/events не переопределяют credentials.
- [ ] Endpoint/pool changes обновляют реальные клиенты; UI различает saved/applied, ошибки видимы.
- [ ] Qdrant switch не смешивает данные/клиенты/ACL/BM25/cache и не теряет ingestion writes.
- [ ] Rate quotas меняются с сохранением истории; cost limits действительно учитывают и ограничивают заявленный scope, а не просто существуют в UI.
- [ ] Benchmark DB history/results/checkpoints и S3 artifacts переживают retry/restart/смену worker. По умолчанию нет local reports/JSONL.
- [ ] S3 backend работает без обязательного DATA_DIR; настоящие temp files bounded и очищаются.
- [ ] Полная проверка и migration report приложены; непроведённые live проверки перечислены явно.

Откат выпусков выполнять по проверенной совместимости schema, env и API. Env восстанавливается из protected snapshots. Existing dynamic rows не перезаписывались; удалённые legacy S3 rows при необходимости старой версии восстанавливаются из pre-migration DB backup. Историю/checkpoints и S3 artifacts не удалять вместе с откатом кода. Переключение Qdrant откатывается отдельной операцией после проверки, что новые writes не потеряются; простого возврата одного URL недостаточно.

## 8. Передача агенту

Рекомендуемый способ — последовательная реализация. Сначала этапы 1–3, затем управляемые настройки 4–8 и benchmark/storage 9–11, затем финальная миграция 12. Этап 7 — отдельный существенный use case; его можно выполнять после независимых storage этапов, но не исключать молча из общего результата.

Стартовый запрос исполнителю: «Реализуй этот план по этапам. Начни с проверки текущего checkout и baseline tests. Сохрани production invariants и текущие значения. Не выполняй live migrate-env --apply, переключение Qdrant, deploy или recreate сервисов без авторизации. После каждого этапа записывай проверенные результаты и обновляй checklist. Не добавляй TOML/YAML и не показывай неработающие настройки как действующие».

Текущая передача: этапы 1–4 и 6 реализованы; код этапа 5 готов, настоящая Redis integration ожидает проверки. Этап 7 частичный: supported text OpenRouter работает, unsupported платные пути блокируются. Этап 8 реализован и имеет unit/ранее выполненные локальные PostgreSQL проверки; live multi-host приёмка остаётся открытой. Дефекты независимого ревью исправлены и покрыты регрессиями выше. Этапы 9–11 реализованы; 352 целевых теста, локальный PostgreSQL, Ruff/mypy и frontend проверки прошли, настоящий S3/live приёмка остаются открытыми. Этап 12 не выполнялся. Рабочие БД, env и сервисы не изменены; production values/цены провайдеров/live переключение не проверены. Изменения этапов 5–11 и исправления ревью ещё не оформлены отдельными commits.
