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
│   └── breakpoint.js         # Breakpoint test
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
- k6 summary: `loadtest/results/summary.json`
- Locust HTML: `loadtest/results/locust_report.html`
- Для Grafana: добавьте `--out influxdb=...` в k6

## CI интегrazione

```yaml
# .github/workflows/loadtest.yml
# Запуск по расписанию или вручную
- run: task loadtest:smoke
```
