# Load Testing

Нагрузочное тестирование RAG-сервера (`/chat`, `/chat/sync`, `/documents`).

## Быстрый старт

```bash
# 1. Поднять стек (если ещё не поднят)
task up

# 2. Создать тестовых пользователей (50 штук)
task loadtest:setup-users

# 3. Запустить smoke-тест (1-5 VU, проверка что всё работает)
task loadtest:smoke

# 4. Запустить нагрузочный тест (0→500 VU за 5 мин, держать 10 мин)
task loadtest:run

# 5. Посмотреть результаты
cat loadtest/results/summary.json | jq .
```

## Сценарии тестирования

| Этап | Команда | Описание |
|------|---------|----------|
| Smoke | `task loadtest:smoke` | 1-5 VU, проверка корректности |
| Load | `task loadtest:run` | 0→500 VU ramp-up, 10 мин steady |
| Spike | `task loadtest:spike` | Резкий скачок 50→500 за 30с |
| Soak | `task loadtest:soak` | 200 VU на 1 час (поиск утечек) |
| Breakpoint | `task loadtest:breakpoint` | Постепенный рост до отказа |
| SSE | `task loadtest:sse` | Locust: тестирование /chat streaming |
| Upload + chat | `task loadtest:upload-chat` | Ровно 1000 попыток загрузки одного куратора + чат 20 других пользователей |
| Реплики + чат | `task loadtest:replicas replicas=4` | Проверка 4 запущенных API-реплик, затем 4 × 20 пользователей чата |

## Подбор лимитов загрузки и соединений

Новые сценарии работают через `http://server:8001` в Docker-сети `rag-network`
(`LOADTEST_NETWORK` меняет имя сети). Это позволяет нагрузить все API-реплики,
минуя IP-лимит чата в nginx. При необходимости задайте `url=...`.
Отключите HTTP rate limits на тестовом сервере по инструкции ниже; оставьте
`job_max_per_principal`, `job_principal_backlog_limit` и `job_backlog_limit` включёнными.

```bash
# Первый аккаунт — куратор-загрузчик; остальные — пользователи чата.
# Для 4 × 20 пользователей нужно минимум 80 различных аккаунтов.
task loadtest:setup-users count=100 prefix=capacity

task loadtest:upload-chat uploads=1000 upload_vus=20 chat_vus=20 duration=10m

# Перед каждым прогоном поднимите соответствующее число API-реплик.
# Используйте базовый docker-compose.yml без dev override с фиксированным портом 8001.
task loadtest:replicas replicas=1 users_per_replica=20 duration=10m
task loadtest:replicas replicas=2 users_per_replica=20 duration=10m
task loadtest:replicas replicas=4 users_per_replica=20 duration=10m
```

`loadtest:replicas` сверяет фактическое число работающих контейнеров `server`
текущего Compose-проекта с `replicas` и останавливается при несовпадении.
В k6 включены DNS round-robin и новые соединения для распределения по репликам.
Для сравнения при **одинаковой суммарной нагрузке** задавайте, например,
`replicas=1 users_per_replica=80`, `replicas=2 users_per_replica=40`,
`replicas=4 users_per_replica=20`.

Загрузки идут параллельно от одного аккаунта, без повторов: измеряются 1000 попыток,
из которых часть может быть отклонена admission-контролем. У каждого файла уникальны
имя, содержимое и ключ идемпотентности. Чат использует другие аккаунты и уникальные
вопросы, чтобы не свести прогон к попаданиям в кэш ответов. Документы остаются
приватными у загрузчика после прогона. `data/sample.txt` задаёт размер и содержание
нагрузки на ingestion; для оценки рабочих лимитов используйте представительный текст.

JSON-результаты сохраняются отдельно для каждого прогона:
`results/summary_upload_chat_*.json` и `results/summary_replicas_N_*.json`.
Сравнивайте `chat_latency` (p95), `chat_errors`, `upload_admission_latency`,
`uploads_accepted`, `uploads_principal_backlog` и `uploads_global_backlog`.
Ожидаемый backlog-отказ — только ответ API с сообщением о capacity и `Retry-After`;
обычные 429/503 считаются ошибкой. Прогон проваливается, если выполнено меньше
заданного числа попыток, не принято ни одной загрузки, ошибки превышают 1%
или p95 чата достигает `chat_p95_ms` (по умолчанию 120000).

На время прогона включите профиль Compose `monitoring`. Он поднимает Prometheus
и PgBouncer exporter. При обновлении существующего стенда примените новый
`STATS_USERS` пересозданием PgBouncer в согласованное окно обслуживания.
Сопоставляйте результаты с:

- `pgbouncer_pools_client_waiting_connections` (`cl_waiting`);
- `pgbouncer_pools_client_maxwait_seconds` (`maxwait`);
- `background_job_oldest_pending_seconds`, backlog и числом выполняемых задач.

Алерт «PgBouncer ждёт свободное соединение» срабатывает, когда `cl_waiting > 0`
или `maxwait > 0` сохраняется 30 секунд. Это повод проверить длительность транзакций
и `DEFAULT_POOL_SIZE=40` с учётом общего бюджета соединений Postgres. Меняйте
один лимит за раз и повторяйте одинаковый прогон; результат по одному лишь
HTTP admission не подтверждает скорость обработки очереди.

Генератор сохраняет пароли при повторном запуске. Если существующие пользователи
созданы без локального файла паролей или первый аккаунт не имеет права загрузки,
он просит новый `prefix`, вместо записи неработающих учётных данных.

Проверка самих сценариев на локальном HTTP-стенде (нужны Docker и образ k6):

```bash
python3 -m unittest discover -s loadtest/tests -v
docker run --rm --entrypoint /bin/promtool \
  -v "$PWD/deploy/monitoring:/rules:ro" -w /rules \
  prom/prometheus:v3.5.0 test rules alerts.test.yml
```

## Структура

```
loadtest/
├── k6/
│   ├── lib/
│   │   └── helpers.js       # Общие функции (login, random question)
│   ├── smoke.js              # Smoke test
│   ├── load.js               # Baseline/load test
│   ├── spike.js              # Spike test
│   ├── soak.js               # Soak test (1 час)
│   ├── breakpoint.js         # Breakpoint test
│   ├── upload-chat.js        # 1000 загрузок одного пользователя + чат других
│   └── replicas.js           # N API-реплик × чат
├── locust/
│   ├── locustfile.py         # Locust: SSE streaming test
│   └── requirements.txt      # Python deps (locust, httpx)
├── data/
│   ├── generate_users.py     # Генерация тестовых пользователей
│   ├── test_users.json       # Сгенерированные пользователи (gitignored)
│   ├── test_questions.json   # Пул вопросов для тестов
│   └── sample.txt            # Тестовый файл для upload
├── docker-compose.loadtest.yml  # Docker Compose для k6/locust
├── setup-users.sh           # Создание пользователей через API
└── README.md
```

## Требования

- **k6**: `docker compose` (контейнер) или [установленный локально](https://k6.io/docs/get-started/installation/)
- **Locust**: `pip install locust httpx` (или через Docker)
- **Стек**: сервер должен быть поднят (`task up`) и проиндексирован (`task ingest`)

## Rate Limits

Сервер лимитирует запросы per-user (Redis sorted sets, `RATE_LIMIT_*` в `server/.env`):

| Policy | Scope | Default | Endpoints |
|---|---|---|---|
| `login` | IP | 5/мин | `POST /auth/login` |
| `chat` | user | 10/мин + 100/день | `POST /chat`, `POST /chat/sync` |
| `upload` | user | 20/час | `POST /documents`, `/ingest`, `/ingest/file`, `/upload` |
| `search` | user | 60/мин | `POST /search/exact` |
| `write` | user | 60/мин | мутации (documents/chunks/groups/curators/api-keys/admin) |
| `benchmark` | user | 3/час | `POST /benchmark`, benchmark sweep/import/apply |

**Перед нагрузочными тестами обязательно выключи лимиты** — иначе 500 VU упрутся в 429:

```bash
# в server/.env
RATE_LIMIT_ENABLED=false
task restart -- server
```

Либо подними лимиты (`RATE_LIMIT_CHAT_PER_MINUTE=10000` и т.д.). После тестов верни `true`.

## Результаты

- k6 JSON: `loadtest/results/results_*.json`
- Новые k6-сценарии: `loadtest/results/summary_upload_chat_*.json`, `summary_replicas_N_*.json`
- Locust HTML: `loadtest/results/locust_report.html`
- Для Grafana: добавьте `--out influxdb=...` в k6

## CI интегrazione

```yaml
# .github/workflows/loadtest.yml
# Запуск по расписанию или вручную
- run: task loadtest:smoke
```
