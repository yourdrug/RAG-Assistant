# RAG — Корпоративный ассистент (локальный, без облака)

**Стек:** FastAPI · LangChain · Qdrant · Ollama (qwen2.5:14b) · PostgreSQL ·
BAAI/bge-m3 (эмбеддинги) · BAAI/bge-reranker-v2-m3 (реранкер) · PaddleOCR (+ опционально Surya)

**Инструменты разработки:** [uv](https://docs.astral.sh/uv/) (пакеты и venv) ·
[Task](https://taskfile.dev/) (единая точка входа) · Docker Compose (оркестрация всех сервисов)

Набор моделей подобран как один из наиболее безопасных сегодня с точки зрения лицензий:
всё работает локально и бесплатно для личного использования и большинства коммерческих
проектов — при соблюдении условий соответствующих лицензий (раздел «Лицензии» ниже).

---

## Структура проекта

```
├── Dockerfile              ← multi-stage (python-base → builder-base → uv-base → development/production)
├── docker-compose.yml         ← локальная разработка (сборка из исходников, live-reload)
├── docker-compose.prod.yml     ← продакшн (готовый образ из GHCR, без сборки)
├── Caddyfile                 ← реверс-прокси + авто-HTTPS, только для `task up:public`
├── install.sh                 ← установка одной командой (curl | bash), см. «Быстрый старт»
├── Taskfile.yml              ← task --list — полный список команд
├── .github/workflows/
│   ├── ci.yml                  ← lint + тесты + валидация конфигов на каждый push/PR
│   └── release.yml              ← сборка и публикация образа в GHCR (main → :edge, теги vX.Y.Z → :latest)
├── VERSION
├── LICENSE                    ← Elastic License 2.0, см. «Лицензии»
├── .env.example               ← переменные для самого docker-compose.yml (DOCKER_MTU и т.п.)
├── .dockerignore
├── data/                       ← ВСЕ вспомогательные данные, один bind-mount в контейнер server
│   ├── docs_sample/              ← сюда кладём документы для индексации
│   ├── models/                    ← опционально: предзагруженные веса bge-m3/bge-reranker (см. ниже)
│   ├── db/postgres/
│   │   ├── init.sql                ← схема (users, conversations, messages), выполняется при первом старте
│   │   ├── postgresql.conf          ← тюнинг под локальную разработку
│   │   └── backups/                 ← дампы (task db:backup), в .gitignore
│   ├── test_questions.json           ← вопросы для бенчмарка
│   ├── qdrant_storage/                ← данные Qdrant (авто, в .gitignore)
│   ├── ollama_models/                  ← модели Ollama (авто, в .gitignore)
│   ├── postgres/                        ← данные Postgres (авто, в .gitignore)
│   ├── ingestion_registry.json           ← реестр индексации (авто, в .gitignore)
│   └── ingestion.log                      ← лог индексации (авто, в .gitignore)
├── client/                      ← React + Vite frontend (порт 3001)
│   ├── src/
│   │   ├── app/                   ← страницы (React Router)
│   │   ├── widgets/              ← виджеты (chat, layout)
│   │   ├── features/             ← фичи (auth, documents, admin)
│   │   ├── entities/             ← доменные модели
│   │   ├── shared/               ← API клиент, UI компоненты, хуки, типы
│   │   ├── stores/               ← Zustand (auth)
│   │   └── providers/            ← QueryClient, ThemeProvider
│   ├── package.json
│   └── vite.config.ts
└── server/
    ├── app/                     ← код приложения (bind-mount цель в dev-режиме)
    │   ├── main.py                ← FastAPI, lifespan, include_router
    │   ├── bootstrap.py           ← автосоздание admin при первом старте
    │   ├── config.py              ← настройки (Settings — pydantic-settings)
    │   ├── api/                   ← HTTP-слой
    │   │   ├── routes/            ← эндпоинты (auth, chat, ingest, documents, groups, clients, conversations, health)
    │   │   └── schemas.py         ← Pydantic-модели запросов/ответов
    │   ├── cli/                   ← CLI (typer) — запуск сервера, индексация, бенчмарк, диагностика PDF
    │   │   ├── cli.py             ← CLI entry-point
    │   │   └── commands/          ← runserver, ingest, benchmark, pdf_diag
    │   ├── domain/                ← бизнес-логика (чистая, без инфраструктуры)
    │   │   ├── rag.py             ← промпт, реранк, форматирование, извлечение источников
    │   │   ├── ingestion.py       ← парсинг документов, OCR, сплиттинг
    │   │   ├── benchmark.py       ← оценка качества retriever + LLM-судьи
    │   │   └── pdf_diag.py        ← диагностика проблемных PDF
    │   ├── infrastructure/        ← инфраструктура и внешние сервисы
    │   │   ├── auth.py            ← пароли (bcrypt), JWT, FastAPI-зависимости
    │   │   ├── database.py        ← SQLAlchemy-операции (users, conversations, messages, documents)
    │   │   ├── clients.py         ← ленивые клиенты (LLM, embeddings, reranker, vector store)
    │   │   ├── qdrant_ops.py      ← работа с Qdrant (collection, upload, search)
    │   │   ├── storage.py         ← файловое хранилище (local / S3)
    │   │   ├── registry.py        ← реестр индексации (JSON)
    │   │   ├── acl.py             ← контроль доступа к документам (visibility, groups)
    │   │   └── logging.py         ← конфигурация логирования
    │   └── services/              ← оркестрация (вызывают domain + infrastructure)
    │       ├── chat_service.py    ← чат (RAG-цепочка, история, стриминг)
    │       ├── user_service.py    ← аутентификация, создание/управление пользователями
    │       ├── document_service.py← загрузка/удаление документов, фоновая обработка
    │       └── ingest_service.py  ← полная/поштучная индексация, реестр
    ├── entrypoint.sh              ← точка входа контейнера (cd app && exec ...)
    ├── pyproject.toml             ← зависимости (uv)
    ├── uv.lock                     ← зафиксированные версии (коммитится в git)
    ├── .env.example                ← скопировать в server/.env
    └── tests/                      ← pytest (unit-тесты бизнес-логики, моки внешних сервисов)
```

Все пути к данным приложения строятся от одной настройки — `DATA_DIR` (`config.py`, по
умолчанию `/code/project/data` внутри контейнера). `docker-compose.yml` монтирует туда весь
host-каталог `./data` одним томом — не нужно синхронизировать несколько отдельных путей.

Архитектура приложения следует принципу Чистой Архитектуры:

- **`domain/`** — чистая бизнес-логика, не зависит от фреймворков и БД. Легко тестируется.
- **`infrastructure/`** — внешние сервисы (БД, Qdrant, S3, авторизация). Реализует интерфейсы из domain.
- **`services/`** — оркестрация: вызывают domain + infrastructure для выполнения use-case.
- **`api/`** — HTTP-слой (FastAPI routes + Pydantic schemas). Тонкая обёртка над services.
- **`cli/`** — CLI (typer): альтернативный вход для индексации, бенчмарка, диагностики.

---

## Frontend (React + Vite)

Полноценный SPA-интерфейс с административной панелью. Порт **3001**.

### Стек

- React 19 + TypeScript (strict)
- Vite (сборка)
- React Router (роутинг)
- TailwindCSS v4 + shadcn/ui компоненты
- TanStack Query (кеширование API)
- TanStack Table (таблицы)
- Zustand (состояние auth)
- React Hook Form + Zod (формы)
- React Dropzone (загрузка файлов)
- React Markdown + remark-gfm (рендеринг Markdown)
- Framer Motion (анимации)
- next-themes (светлая/тёмная тема)

### Архитектура

Feature-Sliced Design:

```
src/
├── app/          ← страницы и роутинг
├── widgets/      ← составные виджеты (chat, layout)
├── features/     ← бизнес-фичи (auth, documents, admin)
├── entities/     ← доменные сущности
└── shared/       ← переиспользуемые: API, UI, хуки, типы
```

### Запуск

```bash
cd client
npm install
npm run dev       # http://localhost:3001
```

API проксируется через Vite на `localhost:8001` (настройка в `vite.config.ts`).

### API Coverage

Frontend использует все 27 существующих backend endpoint:

| Модуль | Эндпоинты | Хуки |
|--------|-----------|------|
| Auth | `POST /auth/login`, `GET /auth/me`, `POST /auth/users`, `GET /auth/users`, `PATCH /auth/users/{id}` | `useLogin`, `useCurrentUser`, `useUsers`, `useCreateUser`, `useToggleUserActive` |
| Chat | `POST /chat` (SSE), `POST /chat/sync` | `streamChat` (SSE client), `useSyncChat` |
| Conversations | `POST /conversations`, `GET /conversations/{id}` | `useCreateConversation`, `useConversationHistory` |
| Documents | `POST /documents`, `GET /documents`, `GET /documents/{id}`, `DELETE /documents/{id}` | `useDocuments`, `useDocument`, `useUploadDocument`, `useDeleteDocument` |
| Ingest | `POST /ingest`, `POST /ingest/file`, `GET /ingest/registry`, `POST /upload` | `useIngestAll`, `useIngestFile`, `useIngestRegistry`, `useUploadFiles` |
| Groups | `POST /groups`, `GET /groups`, `GET /groups/{id}/members`, `POST /groups/{id}/members`, `DELETE /groups/{id}/members/{uid}` | `useGroups`, `useCreateGroup`, `useGroupMembers`, `useAddGroupMember`, `useRemoveGroupMember` |
| Clients | `POST /clients/{id}/assignments`, `DELETE /clients/{id}/assignments/{uid}`, `GET /clients/{id}/assignments` | `useAssignClient`, `useUnassignClient`, `useClientAssignments` |
| Health | `GET /health` | `useHealth` |
| Benchmark | `POST /benchmark` | `useBenchmark` |

### Отсутствующие backend endpoints

Следующие фичи требуют добавления backend endpoints:

- `GET /conversations` — список диалогов пользователя
- `DELETE /conversations/{id}` — удаление диалога
- `PATCH /conversations/{id}` — переименование диалога
- `PATCH /auth/me` — обновление профиля
- `POST /auth/change-password` — смена пароля
- `GET/POST /admin/models` — управление LLM моделями
- `GET/PUT /admin/rag/settings` — настройки RAG
- `GET /admin/vectordb/collections` — коллекции Qdrant
- `GET /admin/ocr/settings`, `GET /admin/ocr/history` — OCR
- `GET /admin/jobs` — фоновые задачи
- `GET /admin/metrics` — метрики
- `GET /admin/logs` — логи
- `GET /admin/settings` — системные настройки
- `GET /search` — полнотекстовый поиск

До добавления этих endpoints в frontend стоят заглушки "Coming Soon".

---

## Авторизация и роли

Саморегистрации нет намеренно — это закрытый инструмент компании, а не публичный сервис.

- **admin** — заводит новых пользователей (`POST /auth/users`), индексирует документы
  (`/ingest*`), видит список пользователей.
- **user** — только общается с чатом (`/chat`, `/chat/sync`) и видит свои диалоги.

Первый admin создаётся автоматически при первом старте контейнера `server` из
`ADMIN_EMAIL`/`ADMIN_PASSWORD` в `server/.env` (см. `main.py:bootstrap_admin`). Дальше новых
пользователей заводит сам admin через API.

```bash
task login email=admin@example.com password=change-me-please   # токен → .auth_token
task me                                                          # кто я
task user:add email=vasya@company.com password=pass123 role=user  # [admin] завести коллегу
task user:list                                                   # [admin] все пользователи
task chat -- "Какие товары подлежат обязательной маркировке?"    # доступно и user, и admin
```

Все защищённые эндпоинты ждут `Authorization: Bearer <token>`. `task login` сохраняет токен в
`.auth_token` (в `.gitignore`), остальные `task`-команды подхватывают его сами. Токен живёт
`JWT_EXPIRE_MINUTES` (по умолчанию 24 часа) — после истечения `task login` нужно повторить.

`GET /health` — единственный эндпоинт без авторизации (нужен для healthcheck).

---

## Требования

- Docker + Docker Compose v2 (`docker compose`)
- [Task](https://taskfile.dev/installation/) — `brew install go-task/tap/go-task` / `snap install task --classic` / см. ссылку для других ОС
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — только для локальной разработки/тестов вне Docker
- ~16 GB свободной RAM (CPU-инференс `qwen2.5:14b` + `bge-m3` + `bge-reranker-v2-m3` + PaddleOCR
  одновременно) или GPU с 12+ GB VRAM

---

## Быстрый старт

### Вариант А — одной командой (install.sh)

Ставит Docker/Task, если их нет, спрашивает домен (опционально) и данные admin-аккаунта,
генерирует секреты, поднимает стек:

```bash
curl -fsSL https://raw.githubusercontent.com/yourdrug/RAG-Assistant/main/install.sh | bash
```

Безопасно перезапускать — если конфигурация уже есть, скрипт её не трогает, просто
пересобирает и поднимает стек. По умолчанию собирает образ из исходников (надёжно, но
дольше); чтобы вместо этого скачать готовый образ из GHCR — см. «Прекомпилированные
образы» ниже — добавь `RAG_USE_PREBUILT=1`:

```bash
curl -fsSL https://raw.githubusercontent.com/yourdrug/RAG-Assistant/main/install.sh -o install.sh
RAG_USE_PREBUILT=1 bash install.sh
```

Дальше — `task ingest`, `task chat -- "вопрос"` и т.д., как обычно (см. таблицу команд ниже).

### Вариант Б — вручную (для тех, кто хочет понимать каждый шаг)

```bash
task init                # .env + server/.env из .env.example + сборка образа
task up                   # поднять qdrant, ollama, postgres, server (в фоне)
task pull-model            # скачать qwen2.5:14b в контейнер ollama (~9 GB, один раз)

task login email=admin@example.com password=change-me-please   # см. server/.env — ADMIN_EMAIL/ADMIN_PASSWORD

cp /path/to/your/docs/* data/docs_sample/
task ingest                 # проиндексировать data/docs_sample/ [admin]

task chat -- "Какие товары подлежат обязательной маркировке?"
task health                  # проверить статус сервисов
```

Сервисы:
- API + Swagger UI: http://localhost:8001/docs
- Qdrant dashboard: http://localhost:6333/dashboard

Порт API — **8001** (не 8000): так он не конфликтует с другими локальными сервисами на 8000.
Порт Ollama на хосте — **11435** (контейнер слушает 11434 как обычно) — на случай, если на
машине уже установлен свой Ollama на стандартном порту.

---

## Команды (Taskfile)

| Команда | Что делает |
|---|---|
| `task init` | Первый запуск: создать `.env`/`server/.env`, собрать образ |
| `task env` | Создать `.env` и `server/.env` из `.env.example`, если их ещё нет |
| `task up` / `task down` | Поднять / остановить весь стек (без домена/HTTPS) |
| `task up:public` | То же + Caddy с авто-HTTPS на своём домене (см. «Свой домен и HTTPS») |
| `task prod:pull` / `task prod:up` / `task prod:up:public` | Готовый образ из GHCR вместо сборки (см. «Прекомпилированные образы») |
| `task restart -- server` | Перезапустить один сервис |
| `task build` | Пересобрать образ `server` (после правок Dockerfile/pyproject.toml) |
| `task logs -- postgres` | Логи сервиса (по умолчанию `server`) |
| `task ps` | Статус контейнеров |
| `task pull-model -- mistral-nemo:12b` | Скачать LLM в контейнер Ollama (по умолчанию `qwen2.5:14b`) |
| `task login email=... password=...` | Залогиниться, сохранить токен в `.auth_token` |
| `task me` | Кто я (по текущему токену) |
| `task user:add email=... password=... role=user` | [admin] Завести пользователя |
| `task user:list` | [admin] Список пользователей |
| `task user:toggle -- id=1 active=false` | [admin] Активировать/деактивировать пользователя |
| `task ingest` | [admin] Индексация `data/docs_sample/` (только новые/изменённые файлы) |
| `task ingest:reset` | [admin] Полная переиндексация с нуля |
| `task ingest:file -- /code/project/data/docs_sample/x.pdf` | [admin] Проиндексировать один файл |
| `task ingest:list` | [admin] Реестр уже проиндексированных файлов |
| `task ingest:upload -- file.pdf` | [admin] Загрузить файл в хранилище |
| `task chat -- "вопрос"` | Синхронный запрос к `/chat/sync` |
| `task health` | Проверка `/health` (без авторизации) |
| `task bench` | Оценка качества retriever + LLM-судьи (`benchmark.py`) |
| `task pdf:diag -- /path/file.pdf` | Диагностика PDF-файла (тип, текст, сканы, кодировка) |
| `task db:shell` | `psql` внутрь контейнера Postgres |
| `task db:backup` | Дамп БД в `data/db/postgres/backups/` |
| `task install` | Установить зависимости локально через uv (для разработки/тестов вне Docker) |
| `task install:surya` | Доустановить опциональный OCR-движок Surya |
| `task test` / `task lint` / `task fmt` | pytest / ruff check / ruff format + автофиксы |
| `task clean` | ⚠️ Остановить стек и удалить `data/postgres`, `data/qdrant_storage`, `data/ollama_models`, `data/caddy_*` |

---

## Свой домен и HTTPS

По умолчанию (`task up`) сервис доступен только на `localhost:8001` — без домена и без HTTPS,
для локальной работы этого достаточно. Чтобы поставить продукт на реальный домен клиента,
поднимается ещё один сервис — `caddy` (реверс-прокси с автоматическим Let's Encrypt HTTPS) —
но **не** по умолчанию, а отдельной командой, осознанно.

### Перед запуском

1. DNS A-запись домена (`rag.клиент.com`) уже указывает на публичный IP этого сервера.
2. Порты **80** и **443** открыты в файрволе сервера (Let's Encrypt проверяет владение доменом
   через порт 80, сертификат отдаётся на 443):
   ```bash
   sudo ufw allow 80/tcp
   sudo ufw allow 443/tcp
   ```

### Запуск

```bash
# в .env (корневом, не server/.env!):
DOMAIN=rag.клиент.com
ACME_EMAIL=admin@клиент.com   # необязательно, для уведомлений об истечении сертификата

task up:public
```

Caddy сам получит сертификат при первом запросе к домену и будет продлевать его автоматически —
никакого `certbot`/ручного перевыпуска. Логи выпуска сертификата — `task logs -- caddy`.

### Что при этом меняется

- `http://rag.клиент.com` автоматически редиректит на `https://`.
- `localhost:8001` продолжает работать параллельно (для локальной отладки/Swagger) — Caddy не
  заменяет прямой порт, а добавляет вход через домен поверх него.
- **Сузь CORS**, когда переходишь на реальный домен — иначе любой сайт в интернете сможет
  дёргать API из браузера от имени залогиненного пользователя:
  ```bash
  # server/.env
  ALLOWED_ORIGINS=https://rag.клиент.com
  ```
  (по умолчанию `*` — ок для localhost-разработки, но не для домена, смотрящего в интернet)

### Несколько доменов / клиентов на одном сервере

Текущий `Caddyfile` — один домен на один запущенный стек. Если нужно несколько инстансов на
одной машине (мультиарендность на уровне инфраструктуры, а не внутри приложения) — поднимай
отдельный `docker compose -p <клиент>` на каждого клиента со своим `.env`/`server/.env`/портами,
и добавляй в `Caddyfile` домены клиентов один за другим как отдельные блоки `{$DOMAIN_N} { ... }`.
Для десятков клиентов на одном сервере это уже отдельная задача автоматизации (шаблонизация
`Caddyfile`+`.env` при онбординге нового клиента) — не проблема на старте, но не решена «из коробки».

---

## Docker-образ: стадии сборки

`Dockerfile` — multi-stage на `uv`, без `requirements.txt`:

```
python-base    → общие ENV (PYTHONUNBUFFERED, пути, PATH)
  └─ builder-base  → системные зависимости (libmagic1, libgl1, libgomp1 — нужны PyMuPDF/PaddleOCR)
       └─ uv-base       → uv sync --frozen --no-dev  (venv в /code/.venv, ОТДЕЛЬНО от кода)
            ├─ development → + dev-зависимости (pytest, ruff); используется docker-compose.yml
            │                 (target: development) с bind-mount ./server/app → live reload
            └─ production   → non-root пользователь, самодостаточный образ без dev-зависимостей
```

Код приложения лежит в `$PYSETUP_PATH/app` (`/code/project/app`), а venv — отдельно, в
`/code/.venv`. В dev-режиме `docker-compose.yml` монтирует `./server/app` поверх
`/code/project/app` для live-reload (плюс `./server/entrypoint.sh` — правишь entrypoint без
пересборки образа); `pyproject.toml`, `uv.lock` и сам venv лежат вне этих путей, поэтому
bind-mount их не затрагивает. `entrypoint.sh` делает `cd app` перед запуском команды (по умолчанию `python main.py runserver`).

Продакшн-образ собрать отдельно:
```bash
docker build --target production -t rag-server:prod .
```

---

## Прекомпилированные образы и CI/CD

### CI (`.github/workflows/ci.yml`)

На каждый push/PR: `ruff check` + `pytest`, `shellcheck` (`install.sh`, `entrypoint.sh`),
валидация `docker-compose.yml`/`docker-compose.prod.yml`/`Taskfile.yml`/`Caddyfile`, пробная
сборка `development`-стадии образа. Мёрж с непройденным CI — сигнал, что что-то сломано, не
зависящий от памяти "я же проверил у себя".

### Публикация образа (`.github/workflows/release.yml`)

`production`-стадия (см. выше) собирается под `linux/amd64` и `linux/arm64` и публикуется в
GitHub Container Registry:

| Когда | Тег |
|---|---|
| push в `main` | `ghcr.io/yourdrug/rag-assistant:edge` |
| git-тег `vX.Y.Z` | `:X.Y.Z`, `:X.Y`, `:latest` |

Публикация блокируется, если не прошли тесты/линт — в реестр не может попасть образ, который
не проходит CI.

### Использовать готовый образ вместо сборки из исходников

`docker-compose.yml` (обычный `task build`/`task up`) всегда собирает `development`-стадию
локально — это гарантированно работает даже если ни одного релиза ещё не публиковалось, и
даёт live-reload для разработки. Если сборка не нужна (просто хочешь поднять готовый продукт
как можно быстрее — не тянуть локально torch/paddleocr и не ждать компиляции):

```bash
task prod:pull        # скачать готовый образ вместо сборки
task prod:up           # поднять (или task prod:up:public — с Caddy и доменом)
```

`install.sh` тоже это умеет — `RAG_USE_PREBUILT=1 bash install.sh` тянет готовый образ вместо
сборки из исходников (сильно быстрее на слабом сервере).

`docker-compose.prod.yml` — намеренно отдельный файл, а не `-f docker-compose.yml -f
docker-compose.prod.yml`: Compose мёржит `volumes`/`ports` по пути монтирования, а не заменяет
список целиком, так что оверлей не смог бы просто «убрать» dev-бинд-маунты базового файла.
Оба файла держат сервисы `qdrant`/`ollama`/`postgres`/`caddy` идентичными намеренно — при
правке одного не забудь синхронизировать второй.

---

## Пайплайн ответа (retrieval)

```
вопрос → Qdrant (bge-m3, top-25 кандидатов)
       → реранкер (bge-reranker-v2-m3, top-6 лучших)
       → LLM (qwen2.5:14b через Ollama, стриминг)
```

### Реранкер (BAAI/bge-reranker-v2-m3)

После поиска в Qdrant (широкий top-k, `RETRIEVER_FETCH_K=25` по умолчанию в `config.py`)
кросс-энкодер `bge-reranker-v2-m3` пересчитывает релевантность каждого кандидата вопросу, и в
промпт уходят только `RETRIEVER_TOP_K=6` лучших. Модель грузится один раз при первом запросе и
кешируется в процессе (`rag_chain.py:get_reranker`). На CPU реранк 25 чанков — доли секунды;
для больших нагрузок поставь `RERANK_DEVICE=cuda`.

### Локальные модели (опционально)

По умолчанию `bge-m3`/`bge-reranker-v2-m3` качаются из HuggingFace Hub при первом запросе и
кешируются внутри `data/` (том persist-ится между рестартами). Если нужен полностью офлайн
старт или не хочется ждать скачивания при первом запуске на новой машине — положи веса заранее:

```bash
uv run huggingface-cli download BAAI/bge-m3 --local-dir data/models/bge-m3
uv run huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir data/models/bge-reranker-v2-m3
```

и в `server/.env` укажи путь внутри контейнера вместо id модели:
```bash
EMBED_MODEL=/code/project/data/models/bge-m3
RERANK_MODEL=/code/project/data/models/bge-reranker-v2-m3
```

Код не меняется — `HuggingFaceEmbeddings`/`CrossEncoder` одинаково принимают и id из Hub, и
локальный путь к каталогу с весами.

### OCR для сканов внутри PDF

`ingestion.py` сам рендерит страницы без текстового слоя в изображение и распознаёт текст:

- **PaddleOCR** (по умолчанию, `OCR_ENGINE=paddleocr`) — Apache-2.0, быстрый, хорошо
  справляется с кириллицей и таблицами.
- **Surya** (`OCR_ENGINE=surya` или `auto`) — точнее на сложной вёрстке, но требует отдельной
  установки (`task install:surya`) и имеет ограничение по лицензии весов — см. «Лицензии».
- `OCR_ENABLED=false` — полностью выключить OCR.

---

## Поддерживаемые форматы документов

| Формат | Парсер | Примечание |
|--------|--------|------------|
| `.pdf` | PyMuPDF, страницы-сканы → OCR (PaddleOCR/Surya) | Текстовый слой — напрямую; сканы — через OCR |
| `.docx` | python-docx + XML fallback | Параграфы + таблицы; битые файлы читаются через raw XML |
| `.doc` | python-docx | Частичная поддержка старого формата |
| `.rtf` | striprtf | Без pandoc и libreoffice |
| `.md` | markdown | Рендер → plain text |
| `.txt` | встроенный | Автоопределение кодировки (utf-8 / cp1251) |

---

## Конфигурация

Два `.env`-файла с разным назначением:

- **`.env`** (корень, из `.env.example`) — переменные самого `docker-compose.yml`: `DOCKER_MTU`
  (см. Troubleshooting) и `DOMAIN`/`ACME_EMAIL` (см. «Свой домен и HTTPS», нужны только для
  `task up:public`).
- **`server/.env`** (из `server/.env.example`) — настройки приложения:

```bash
DATABASE_URL=postgresql://raguser:ragpassword@postgres:5432/ragdb   # креды совпадают с docker-compose.yml
QDRANT_URL=http://qdrant:6333
COLLECTION_NAME=company_docs

OLLAMA_BASE_URL=http://ollama:11434    # внутри docker-сети — всегда 11434, порт 11435 только снаружи
LLM_MODEL=qwen2.5:14b                  # или mistral-nemo:12b, qwen2.5:7b

EMBED_MODEL=BAAI/bge-m3
RERANK_MODEL=BAAI/bge-reranker-v2-m3
RERANK_DEVICE=cpu                      # cuda — если есть NVIDIA GPU

ALLOWED_ORIGINS=*                      # на проде с доменом — конкретный https://rag.клиент.com

OCR_ENABLED=true
OCR_ENGINE=paddleocr                   # paddleocr | surya | auto
OCR_LANG_PADDLE=ru

JWT_SECRET_KEY=change-me-in-production   # openssl rand -hex 32
JWT_EXPIRE_MINUTES=1440

ADMIN_EMAIL=admin@example.com
ADMIN_PASSWORD=change-me-please
```

Пароль Postgres задан прямо в `docker-compose.yml` (сервис `postgres`, дефолт для локальной
разработки). Меняешь его там — обнови и `DATABASE_URL` в `server/.env`. Порт Postgres наружу не
публикуется (`expose`, не `ports`) — доступ только изнутри docker-сети и через `task db:shell`.

### Смена LLM модели

```bash
task pull-model -- mistral-nemo:12b
# поменять LLM_MODEL в server/.env
task restart -- server
```

| Модель | Лицензия | RAM | Русский | Скорость |
|--------|----------|-----|---------|----------|
| `qwen2.5:14b` (по умолчанию) | Apache-2.0 | ~10 GB | ⭐⭐⭐⭐ | средне |
| `mistral-nemo:12b` | Apache-2.0 | ~9 GB | ⭐⭐⭐ | средне |
| `qwen2.5:7b` | Apache-2.0 | 8 GB | ⭐⭐⭐ | быстро |

Контейнер `ollama` настроен на `OLLAMA_MAX_LOADED_MODELS=1`, `OLLAMA_NUM_PARALLEL=1`,
`OLLAMA_KEEP_ALIVE=5m` — держит в памяти только одну модель и выгружает её через 5 минут
простоя. Это компромисс для однопользовательского/командного локального стенда, а не
высоконагруженного сервиса; ресурсы контейнера ограничены `deploy.resources` (до 8 CPU / 20 GB).

---

## Лицензии

### Лицензия проекта — Elastic License 2.0 (ELv2)

Сам код этого репозитория распространяется под [Elastic License 2.0](LICENSE) — не MIT/Apache.
Простыми словами, три ограничения:

1. **Нельзя предлагать это как хостинг/managed-сервис третьим лицам** — то есть нельзя взять
   этот код и продавать как свой SaaS, где чужие компании платят тебе за доступ к нему как
   к сервису. Ставить и использовать у себя (в том числе в коммерческой компании,
   в том числе много инсталляций для разных внутренних команд) — можно свободно.
2. Нельзя обходить/отключать функциональность лицензионных ключей (в текущей версии таких
   нет, но пункт — часть стандартного текста лицензии).
3. Нельзя убирать копирайты и упоминания лицензии из исходников.

Что можно без ограничений: self-host на своём сервере/домене (в том числе по этому README),
форкать, модифицировать под себя, использовать внутри компании любого размера, продавать
консалтинг/внедрение/поддержку вокруг этого продукта. Нельзя — только "продавать сам продукт
как облачный сервис для чужих клиентов, конкурируя с автором в его же нише". Официальный текст
и FAQ: https://www.elastic.co/licensing/elastic-license.

Это не юридическая консультация — если ELv2 принципиален для твоего кейса, проверь применимость
с юристом.

### Лицензии моделей внутри стека

Отдельно от лицензии самого проекта — лицензии LLM/эмбеддингов/OCR, которые стек скачивает
и запускает:

| Компонент | Модель/пакет | Лицензия | Ограничения |
|---|---|---|---|
| LLM | Qwen2.5-14B-Instruct | Apache-2.0 | Нет |
| LLM (альтернатива) | Mistral-Nemo-12B-Instruct | Apache-2.0 | Нет |
| Эмбеддинги | BAAI/bge-m3 | MIT | Нет |
| Реранкер | BAAI/bge-reranker-v2-m3 | MIT | Нет |
| OCR | PaddleOCR | Apache-2.0 (код и веса) | Нет |
| OCR (опционально) | Surya (datalab-to/surya) | Apache-2.0 — код; **отдельная лицензия — веса модели** | Веса бесплатны для research, личного использования и организаций с выручкой/финансированием до $5M; выше — нужна платная коммерческая лицензия у разработчиков. Не входит в базовые зависимости, ставится осознанно (`task install:surya`). |

Дефолтная конфигурация (Qwen2.5-14B + bge-m3 + bge-reranker-v2-m3 + PaddleOCR) не содержит
ограничений по выручке или сфере использования. Это не юридическая консультация: перед
продакшн-использованием в компании стоит свериться с актуальными текстами лицензий на
HuggingFace/GitHub каждой модели.

---

## Работа без интернета

Коротко: **сам чат/RAG после первоначальной настройки работает полностью локально**, без
единого обращения наружу. Но настройка "с нуля" интернет использует — по разным поводам,
не все из которых очевидны. Ниже — полный список.

### Нужен интернет один раз, при настройке

| Что | Когда | Объём |
|---|---|---|
| Docker-образы (qdrant, ollama, postgres, caddy) | `task up` (первый раз) | ~1-2 GB суммарно |
| Сборка образа `server` из исходников | `task build` (apt + PyPI: torch, paddleocr и т.д.) | несколько GB, либо `task prod:pull` — тянет уже готовый образ из GHCR, не требует apt/PyPI вообще |
| Модель LLM | `task pull-model` | ~5-10 GB в зависимости от модели |
| Эмбеддинги + реранкер (bge-m3, bge-reranker-v2-m3) | автоматически при первом `/ingest` или `/chat` | ~3 GB суммарно |
| OCR-модели PaddleOCR (детекция/распознавание/угол) | автоматически при первом скане в PDF | ~20 MB |

После этого — интернет **не нужен**: LLM (Ollama), эмбеддинги, реранкер и OCR исполняются
внутри контейнеров на своём железе, без обращений к OpenAI/HuggingFace/куда-либо ещё.

### Что персистится само, а что нет

Модели bge-m3/bge-reranker-v2-m3/PaddleOCR кэшируются в `data/.cache/` (`HOME`/`HF_HOME`
заданы в `Dockerfile` намеренно, а не оставлены как дефолт контейнера) — переживают
`task down && task up` и обновление образа. Без этого модели скачивались бы заново при
каждом пересоздании контейнера, а не только один раз — это баг, который был в более ранней
версии инструкции ниже и уже исправлен.

Модель Ollama хранится в `data/ollama_models/` — тоже персистит.

### Полностью офлайн с самого первого запуска (air-gapped)

Если у целевого сервера **вообще нет** доступа в интернет (даже разового) — всё нужно
подготовить заранее на другой машине и перенести файлами:

1. **Образ `server`**: собери (`task build`) или скачай (`task prod:pull`) на машине с
   интернетом, затем `docker save ghcr.io/yourdrug/rag-assistant:latest | gzip > server.tar.gz`,
   перенеси и на целевом сервере `docker load < server.tar.gz` (аналогично для образов
   `qdrant/qdrant`, `ollama/ollama`, `postgres:16-alpine`, `caddy:2-alpine`, если Caddy нужен).
2. **Модель LLM**: `ollama pull qwen2.5:14b` на машине с интернетом (можно прямо в контейнере
   `ollama`), затем скопировать `data/ollama_models/` целиком на целевой сервер.
3. **Эмбеддинги/реранкер**: скачать заранее и положить в `data/models/` — это уже описано
   выше в «Локальные модели» — и указать `EMBED_MODEL`/`RERANK_MODEL` как локальный путь в
   `server/.env`, а не как id из HuggingFace Hub.
4. **PaddleOCR**: модели маленькие (~20 MB), но встроенного механизма их предзагрузки без
   единого обращения к сети сейчас нет — если OCR критичен, а сервер air-gapped с первой
   секунды, дай знать, добавим (сейчас `OCR_ENABLED=false` — рабочий обходной путь: PDF со
   сканами просто не индексируются, а обычные текстовые PDF/DOCX работают как обычно).
5. Перенеси весь `data/` целиком, `.env`/`server/.env`, `docker-compose.prod.yml`,
   `Caddyfile` (если используешь) на сервер, дальше `task prod:up` (без `pull` — образы уже
   локально через `docker load`).

Это не покрыто автотестами (негде взять air-gapped стенд в CI) — если что-то не сойдётся,
`task logs -- server` покажет, на каком именно шаге не хватает файла/модели.

---

## Troubleshooting

### Caddy не получает сертификат / `task up:public` виснет на старте

Проверить в таком порядке:

```bash
task logs -- caddy
```

Самые частые причины:
- **DNS ещё не обновился** — `dig +short rag.клиент.com` должен вернуть IP этого сервера.
  Пропагация A-записи может занимать от минут до часов.
- **Порты 80/443 закрыты в файрволе** — см. «Свой домен и HTTPS» → `ufw allow`.
- **Порт 80/443 уже занят другим процессом** на хосте (`sudo ss -ltnp | grep -E ':80|:443'`) —
  у Caddy должен быть монопольный доступ к ним.
- **Rate limit Let's Encrypt** — при частых перезапусках с одним и тем же доменом можно словить
  лимит (5 неудачных попыток/час на домен). Подожди час или используй staging-окружение Let's
  Encrypt на время отладки (см. документацию Caddy про `acme_ca`).

### `Error: EOF` при `ollama pull` больших моделей (маленькие скачиваются нормально)

Типичный признак заниженного MTU на пути к интернету (чаще всего — VPN на хосте).

```bash
ip link show | grep mtu                # узнать реальный MTU VPN-интерфейса
echo "DOCKER_MTU=1400" >> .env         # подставь своё значение (это корневой .env, не server/.env!)
task down && task up                    # пересоздаёт сеть — restart недостаточно
task pull-model                          # ollama докачивает с места обрыва, не с нуля
```

### `401 Unauthorized` при `/chat`, `/ingest` и т.д.

Токена нет, истёк или неверный. `task login email=... password=...` заново — токен живёт
`JWT_EXPIRE_MINUTES` (по умолчанию 24 часа). Проверить, что токен вообще сохранился:
`cat .auth_token`.

### `403 Forbidden` на `/ingest*` или `/auth/users`

Это эндпоинты только для admin. `task me` покажет твою текущую роль. Роль назначается только
при создании пользователя; сменить роль существующему — через `task db:shell` →
`UPDATE users SET role = 'admin' WHERE email = '...';`.

### `exec .../uvicorn: no such file or directory`

Venv внутри образа затёрт. Проверь, что в `Dockerfile` venv лежит в `/code/.venv`, а bind-mount
в `docker-compose.yml` (`./server/app:/code/project/app`) — только на `app/`, не на весь
`/code/project`. Если правил `.dockerignore` — убедись, что там исключён `**/.venv/`.

### `FileNotFoundError: /code/project/data/ingestion.log` при локальном запуске без Docker

`DATA_DIR` по умолчанию указывает на путь внутри контейнера. Локально либо экспортируй
`DATA_DIR=$(pwd)/data` перед запуском, либо используй готовые `task`-обёртки (`task bench` уже
делает это через `DATA_DIR=../data`), либо запускай тесты через `task test` — там подставляется
временная директория автоматически (`server/tests/conftest.py`).

---

## Разработка без Docker (uv)

```bash
task install          # uv sync --frozen — основные + dev-зависимости (pytest, ruff)
task test             # pytest (unit-тесты бизнес-логики, моки внешних сервисов)
task test -- -k auth  # запустить только тесты модуля auth
task lint             # ruff check
task fmt              # ruff format + автофиксы

# Запуск сервера локально (нужен доступ к Postgres/Qdrant/Ollama — см. task up):
cd server/app && DATA_DIR=../../data ../.venv/bin/uvicorn main:app --reload --port 8001
```

### Тестирование

Тесты покрывают чистую бизнес-логику (`domain/`, `infrastructure/acl.py`, `services/`) без
реальных БД/Qdrant/Ollama. Внешние зависимости мокаются через `unittest.mock` и `conftest.py`.

```bash
task test                                    # все тесты (~194)
task test -- tests/test_rag_logic.py         # один файл
task test -- tests/test_acl.py -k "admin"   # конкретный тест
```

Зависимости — в `server/pyproject.toml`, зафиксированы в `server/uv.lock` (коммитится в git).
Обновление версий: поправить `pyproject.toml` → `task lock` → `task install`.