# RAG — Корпоративный ассистент

**Стек:** FastAPI · LangChain · Qdrant · Ollama (qwen2.5:14b) · PostgreSQL ·
BAAI/bge-m3 (эмбеддинги) · BAAI/bge-reranker-v2-m3 (реранкер) · PaddleOCR (+ опционально Surya)

**Инструменты разработки:** [uv](https://docs.astral.sh/uv/) ·
[Taskfile](https://taskfile.dev/) ·
Docker Compose

Набор моделей подобран как один из наиболее безопасных сегодня с точки зрения лицензий:
всё работает локально и бесплатно для личного использования и большинства коммерческих
проектов — при соблюдении условий соответствующих лицензий (детали ниже, раздел «Лицензии»).

---

## Требования

- Docker + Docker Compose v2
- [Task](https://taskfile.dev/installation/) — `brew install go-task/tap/go-task` / `snap install task --classic`
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — только если нужна локальная разработка/тесты вне Docker
- ~16 GB свободной RAM (CPU-инференс `qwen2.5:14b` + `bge-m3` + `bge-reranker-v2-m3` + PaddleOCR
  одновременно) или GPU с 12+ GB VRAM для комфортной скорости

---

## Быстрый старт

```bash
task init      # .env из .env.example + сборка образа api
task up        # поднять qdrant, ollama, postgres, api
task pull-model  # скачать qwen2.5:14b в контейнер ollama (~9 GB, один раз)

cp /path/to/your/docs/* data/docs_sample/
task ingest    # проиндексировать data/docs_sample/

task chat -- "Любой вопрос по вашему документу?"
task health    # проверить статус всех сервисов
```

Сервисы:
- API + Swagger UI: http://localhost:8000/docs
- Qdrant dashboard: http://localhost:6333/dashboard

Полный список команд — `task --list` или таблица ниже.

---

## Команды (Taskfile)

| Команда | Что делает |
|---|---|
| `task init` | Первый запуск: создать `.env`, поставить venv локально, собрать образ `api` |
| `task up` / `task down` | Поднять / остановить весь стек |
| `task restart -- api` | Перезапустить один сервис |
| `task build` | Пересобрать образ `api` (после правок Dockerfile/pyproject.toml) |
| `task logs -- postgres` | Логи сервиса (по умолчанию `api`) |
| `task ps` | Статус контейнеров |
| `task pull-model -- mistral-nemo:12b` | Скачать LLM в Ollama (по умолчанию — `LLM_MODEL` из `.env`) |
| `task ingest` | Индексация `data/docs_sample/` (только новые/изменённые файлы) |
| `task ingest:reset` | Полная переиндексация с нуля |
| `task ingest:file -- /code/project/data/docs_sample/x.pdf` | Проиндексировать один файл |
| `task ingest:list` | Реестр уже проиндексированных файлов |
| `task chat -- "вопрос"` | Синхронный запрос к `/chat/sync` |
| `task health` | Проверка `/health` |
| `task bench` | Оценка качества retriever + LLM-судьи (`benchmark.py`) |
| `task db:shell` | `psql` внутрь контейнера Postgres |
| `task db:backup` | Дамп БД в `data/db/postgres/backups/` |
| `task install` | Установить зависимости локально через uv (для разработки/тестов вне Docker) |
| `task test` / `task lint` / `task fmt` | pytest / ruff check / ruff format + fix |
| `task clean` | ⚠️ Снести все локальные данные (Qdrant/Postgres/Ollama volumes) |
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

### Пайплайн ответа (retrieval)

```
вопрос → Qdrant (bge-m3, top-25 кандидатов)
       → реранкер (bge-reranker-v2-m3, top-6 лучших)
       → LLM (qwen2.5:14b через Ollama, стриминг)
```

---

## Конфигурация (.env)

`docker-compose.yml` подхватывает `.env` автоматически (`cp .env.example .env`, либо `task env`).
Все переменные — с дефолтами, менять нужно только то, что действительно хочешь поменять:

```bash
POSTGRES_DB=ragdb
POSTGRES_USER=raguser
POSTGRES_PASSWORD=ragpassword

COLLECTION_NAME=company_docs

LLM_MODEL=qwen2.5:14b              # или mistral-nemo:12b, qwen2.5:7b

EMBED_MODEL=BAAI/bge-m3
RERANK_MODEL=BAAI/bge-reranker-v2-m3
RERANK_DEVICE=cpu                  # cuda — если есть NVIDIA GPU

OCR_ENABLED=true
OCR_ENGINE=paddleocr               # paddleocr | surya | auto
OCR_LANG_PADDLE=ru
```

## Смена LLM модели

```bash
task pull-model -- mistral-nemo:12b
# поменять LLM_MODEL в .env
task restart -- api
```

| Модель | Лицензия | RAM | Русский | Скорость |
|--------|----------|-----|---------|----------|
| `qwen2.5:14b` (по умолчанию) | Apache-2.0 | ~10 GB | ⭐⭐⭐⭐ | средне |
| `mistral-nemo:12b` | Apache-2.0 | ~9 GB | ⭐⭐⭐ | средне |
| `qwen2.5:7b` | Apache-2.0 | 8 GB | ⭐⭐⭐ | быстро |

---

## Реранкер (BAAI/bge-reranker-v2-m3)

После поиска в Qdrant (широкий top-k, `RETRIEVER_FETCH_K=25` по умолчанию в `config.py`)
кросс-энкодер `bge-reranker-v2-m3` пересчитывает релевантность каждого кандидата вопросу, и в
промпт уходят только `RETRIEVER_TOP_K=6` лучших. Это заметно снижает число нерелевантных чанков
в контексте по сравнению с одним только векторным поиском, особенно на длинных/разнородных базах.

Модель грузится один раз при первом запросе и кешируется в процессе (`rag_chain.py:get_reranker`).
На CPU реранк 25 чанков занимает доли секунды; для больших нагрузок поставь `RERANK_DEVICE=cuda`.

---

## OCR для сканов внутри PDF

`ingestion.py` сам рендерит страницы без текстового слоя в изображение и распознаёт текст:

- **PaddleOCR** (по умолчанию, `OCR_ENGINE=paddleocr`) — Apache-2.0, быстрый, хорошо
  справляется с кириллицей и таблицами.
- **Surya** (`OCR_ENGINE=surya` или `auto`) — точнее на сложной вёрстке и рукописном
  тексте, но требует отдельной установки (`task install:surya`, не входит в базовый образ)
  и имеет **ограничение по лицензии весов** — см. раздел «Лицензии».
- `OCR_ENGINE=auto` — сначала PaddleOCR, и только если он не дал текста — Surya (если установлен).
- `OCR_ENABLED=false` — полностью выключить OCR (страницы-сканы будут пропущены).

---

## Лицензии — что важно знать перед продакшн-использованием

| Компонент | Модель/пакет | Лицензия | Ограничения |
|---|---|---|---|
| LLM | Qwen2.5-14B-Instruct | Apache-2.0 | Нет |
| LLM (альтернатива) | Mistral-Nemo-12B-Instruct | Apache-2.0 | Нет |
| Эмбеддинги | BAAI/bge-m3 | MIT | Нет |
| Реранкер | BAAI/bge-reranker-v2-m3 | MIT | Нет |
| OCR | PaddleOCR | Apache-2.0 (код и веса) | Нет |
| OCR (опционально) | Surya (datalab-to/surya) | Apache-2.0 — код; **отдельная лицензия — веса модели** | Веса бесплатны для research, личного использования и организаций с выручкой/финансированием до $5M; выше — нужна платная коммерческая лицензия у разработчиков. Поэтому Surya не входит в базовые зависимости и не является движком по умолчанию. |

**Практический вывод:** дефолтная конфигурация (Qwen2.5-14B + bge-m3 + bge-reranker-v2-m3 +
PaddleOCR) не содержит ограничений по выручке или сфере использования. Единственный компонент
с такими ограничениями — Surya — подключается вручную и осознанно (`task install:surya`), а не
по умолчанию. Это не юридическая консультация: перед продакшн-использованием в компании стоит
свериться с актуальными текстами лицензий на HuggingFace/GitHub каждой модели.

---

## Разработка без Docker (uv)

```bash
task install          # uv sync --frozen — ставит основные + dev-зависимости (pytest, ruff)
task test             # pytest
task lint             # ruff check
task fmt              # ruff format + автофиксы

# нужны локально запущенные qdrant/ollama/postgres — см. docker-compose.yml портов,
# либо `task up` и работать поверх поднятых контейнеров
```

Зависимости описаны в `api/pyproject.toml`, зафиксированы в `api/uv.lock` . 
Обновление версий: поправить `pyproject.toml` → `task lock` → `task install`.

---

## Структура проекта

```
├── Taskfile.yml          ← все команды: task --list
├── docker-compose.yml
├── .env.example           ← скопировать в .env
├── data/                  ← все вспомогательные данные
│   ├── docs_sample/       ← сюда кладём документы
│   ├── models/            ← bge-m3, bge-reranker-v2-m3
│   ├── qdrant_storage/    ← данные Qdrant (авто, в .gitignore)
│   ├── ollama_models/     ← модели Ollama (авто, в .gitignore)
│   ├── postgres/          ← данные Postgres (авто, в .gitignore)
│   ├── db/postgres/       ← init.sql, postgresql.conf, backups/
│   ├── test_questions.json← вопросы для бенчмарка
│   ├── ingestion_registry.json ← реестр индексации (авто)
│   └── ingestion.log      ← лог индексации (авто)
├── client/
│   ├── rag-chat.html
│   └── rag-chat.jsx
└── server/
    ├── app/
    │   ├── main.py        ← FastAPI, эндпоинты
    │   ├── rag_chain.py   ← LangChain LCEL цепочка + реранкер
    │   ├── ingestion.py   ← парсинг, OCR сканов и индексация
    │   ├── database.py    ← история диалогов (PostgreSQL)
    │   ├── config.py      ← настройки (LLM, эмбеддинги, реранкер, OCR)
    │   ├── benchmark.py   ← оценка качества retriever + LLM-судьи
    │   └── pdf_diag.py    ← диагностика проблемных PDF
    ├── pyproject.toml     ← зависимости (uv)
    ├── uv.lock            ← зафиксированные версии (коммитится в git)
    ├── Dockerfile         ← multi-stage сборка через uv
    └── tests/             ← pytest: чистые функции ingestion.py / rag_chain.py
```
