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
├── docker-compose.yml
├── Taskfile.yml              ← task --list — полный список команд
├── VERSION
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
├── client/
│   ├── rag-chat.html           ← статический демо-виджет чата
│   └── rag-chat.jsx
└── server/
    ├── app/                     ← код приложения (bind-mount цель в dev-режиме)
    │   ├── main.py                ← FastAPI, эндпоинты
    │   ├── auth.py                 ← пароли, JWT, зависимости для проверки роли
    │   ├── rag_chain.py            ← LangChain LCEL цепочка + реранкер
    │   ├── ingestion.py            ← парсинг, OCR сканов и индексация
    │   ├── database.py             ← пользователи, история диалогов (PostgreSQL)
    │   ├── config.py               ← настройки (LLM, эмбеддинги, реранкер, OCR, JWT, data_dir)
    │   ├── benchmark.py             ← оценка качества retriever + LLM-судьи
    │   └── pdf_diag.py              ← диагностика проблемных PDF
    ├── entrypoint.sh              ← точка входа контейнера (cd app && exec ...)
    ├── pyproject.toml             ← зависимости (uv)
    ├── uv.lock                     ← зафиксированные версии (коммитится в git)
    ├── .env.example                ← скопировать в server/.env
    └── tests/                       ← pytest: auth.py / ingestion.py / rag_chain.py
```

Все пути к данным приложения строятся от одной настройки — `DATA_DIR` (`config.py`, по
умолчанию `/code/project/data` внутри контейнера). `docker-compose.yml` монтирует туда весь
host-каталог `./data` одним томом — не нужно синхронизировать несколько отдельных путей.

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
| `task up` / `task down` | Поднять / остановить весь стек |
| `task restart -- server` | Перезапустить один сервис |
| `task build` | Пересобрать образ `server` (после правок Dockerfile/pyproject.toml) |
| `task logs -- postgres` | Логи сервиса (по умолчанию `server`) |
| `task ps` | Статус контейнеров |
| `task pull-model -- mistral-nemo:12b` | Скачать LLM в контейнер Ollama (по умолчанию `qwen2.5:14b`) |
| `task login email=... password=...` | Залогиниться, сохранить токен в `.auth_token` |
| `task me` | Кто я (по текущему токену) |
| `task user:add email=... password=... role=user` | [admin] Завести пользователя |
| `task user:list` | [admin] Список пользователей |
| `task ingest` | [admin] Индексация `data/docs_sample/` (только новые/изменённые файлы) |
| `task ingest:reset` | [admin] Полная переиндексация с нуля |
| `task ingest:file -- /code/project/data/docs_sample/x.pdf` | [admin] Проиндексировать один файл |
| `task ingest:list` | [admin] Реестр уже проиндексированных файлов |
| `task chat -- "вопрос"` | Синхронный запрос к `/chat/sync` |
| `task health` | Проверка `/health` (без авторизации) |
| `task bench` | Оценка качества retriever + LLM-судьи (`benchmark.py`) |
| `task db:shell` | `psql` внутрь контейнера Postgres |
| `task db:backup` | Дамп БД в `data/db/postgres/backups/` |
| `task install` | Установить зависимости локально через uv (для разработки/тестов вне Docker) |
| `task install:surya` | Доустановить опциональный OCR-движок Surya |
| `task test` / `task lint` / `task fmt` | pytest / ruff check / ruff format + автофиксы |
| `task clean` | ⚠️ Остановить стек и удалить `data/postgres`, `data/qdrant_storage`, `data/ollama_models` |

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
bind-mount их не затрагивает. `entrypoint.sh` делает `cd app` перед запуском uvicorn (модули
приложения — плоские, без пакета).

Продакшн-образ собрать отдельно:
```bash
docker build --target production -t rag-server:prod .
```

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

- **`.env`** (корень, из `.env.example`) — переменные самого `docker-compose.yml`, сейчас только
  `DOCKER_MTU` (см. Troubleshooting).
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

## Лицензии — что важно знать перед продакшн-использованием

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

## Troubleshooting

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
task test             # pytest (DATA_DIR подставляется автоматически, см. tests/conftest.py)
task lint             # ruff check
task fmt              # ruff format + автофиксы

# main.py импортирует соседние модули (config, auth, rag_chain, ...) плоско, без пакета —
# запускать нужно из server/app/, и нужен доступ к Postgres/Qdrant/Ollama (напр. task up
# для инфраструктуры, а server гонять локально):
cd server/app && DATA_DIR=../../data ../.venv/bin/uvicorn main:app --reload --port 8001
```

Зависимости — в `server/pyproject.toml`, зафиксированы в `server/uv.lock` (коммитится в git).
Обновление версий: поправить `pyproject.toml` → `task lock` → `task install`.