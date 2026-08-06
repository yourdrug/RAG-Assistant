# RAG — Корпоративный ассистент (локальный, без облака)

**Стек:** FastAPI · LangChain · Qdrant · Ollama · PostgreSQL · BAAI/bge-m3 · BAAI/bge-reranker-v2-m3 · PaddleOCR

**Лицензия проекта:** Elastic License 2.0 — можно self-host, нельзя продавать как SaaS.

---

## Быстрый старт

```bash
# Одной командой (автоустановка Docker/Task, сборка, подъём стека):
curl -fsSL https://raw.githubusercontent.com/yourdrug/RAG-Assistant/main/install.sh | bash

# Или вручную:
task init && task up && task pull-model
task login email=admin@example.com password=change-me-please
cp /path/to/docs/* data/docs_sample/
task ingest
task chat -- "Вопрос"
```

После `task up`: API — http://localhost:8001/docs, Qdrant — http://localhost:6333/dashboard

---

## Команды

| Команда | Что делает |
|---|---|
| `task init` | .env + собрать образ |
| `task up` / `task down` | Поднять / остановить стек |
| `task up:public` | + Caddy с авто-HTTPS (нужен `DOMAIN` в `.env`) |
| `task build` | Пересобрать образ server |
| `task pull-model` | Скачать LLM в Ollama |
| `task login email=... password=...` | Залогиниться → `.auth_token` |
| `task ingest` | Проиндексировать `data/docs_sample/` [admin] |
| `task chat -- "вопрос"` | Синхронный запрос |
| `task bench` | Оценка качества |
| `task test` / `task lint` / `task fmt` | pytest / ruff |
| `task db:shell` / `task db:backup` | psql / дамп |
| `task clean` | ⚠️ Удалить все данные |

Полный список: `task --list`

---

## Конфигурация

Два `.env` файла:

- **`.env`** (корень) — `DOCKER_MTU`, `DOMAIN` (для `task up:public`)
- **`server/.env`** — настройки приложения (БД, LLM, OCR, JWT и т.д.)

Ключевые переменные в `server/.env`:

```bash
LLM_MODEL=qwen2.5:14b          # или mistral-nemo:12b, qwen2.5:7b
EMBED_MODEL=BAAI/bge-m3
RERANK_MODEL=BAAI/bge-reranker-v2-m3
DEVICE=cpu                        # cuda если есть GPU
OCR_ENGINE=paddleocr            # paddleocr | surya | auto
JWT_SECRET_KEY=change-me        # openssl rand -hex 32
ADMIN_EMAIL=admin@example.com
ADMIN_PASSWORD=change-me-please
```

Динамические параметры (hot-reload без рестарта) хранятся в БД и управляются через `PUT /admin/config/{key}`.

---

## Архитектура

```
domain/        ← бизнес-логика (чистая, без инфраструктуры)
application/   ← сервисы, порты (протоколы), DTO
infrastructure/← SQLAlchemy, Qdrant, Ollama, S3
presentation/  ← FastAPI routes
```

**Пайплайн:** вопрос → Qdrant (top-25) → реранкер (top-6) → LLM (стриминг)

**Порты:** Application зависит от протоколов (`application/ports/`), infrastructure предоставляет реализации. DI через конструктор.

---

## Форматы документов

| Формат | Парсер |
|--------|--------|
| `.pdf` | PyMuPDF + OCR для сканов (PaddleOCR) |
| `.docx` | python-docx |
| `.rtf` | striprtf |
| `.md` | markdown → plain text |
| `.txt` | автоопределение кодировки |

---

## Лицензии моделей

| Компонент | Модель | Лицензия |
|---|---|---|
| LLM | Qwen2.5-14B / Mistral-Nemo-12B | Apache-2.0 |
| Эмбеддинги | BAAI/bge-m3 | MIT |
| Реранкер | BAAI/bge-reranker-v2-m3 | MIT |
| OCR | PaddleOCR | Apache-2.0 |
| OCR (опц.) | Surya | Apache-2.0 (код), **платная лицензия весов сверх $5M** |

---

## Troubleshooting

**`Error: EOF` при ollama pull** → MTU слишком низкий (VPN). `echo "DOCKER_MTU=1400" >> .env && task down && task up`

**`401 Unauthorized`** → Токен истёк. `task login email=... password=...` заново.

**`403 Forbidden` на /ingest** → Нужна роль admin. `task me` проверить роль.

**Caddy не получает сертификат** → Проверь DNS (`dig +short домен`), порты 80/443 открыты, нет rate limit Let's Encrypt.
