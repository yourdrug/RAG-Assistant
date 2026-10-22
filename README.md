# RAG — Корпоративный ассистент (локальный, без облака)

**Стек:** FastAPI · LangChain · Qdrant · Ollama · PostgreSQL · BAAI/bge-m3 · BAAI/bge-reranker-v2-m3 · PaddleOCR

**Лицензия проекта:** Elastic License 2.0 — можно self-host, нельзя продавать как SaaS.

---

## Быстрый старт

```bash
# Одной командой (автоустановка Docker/Task, сборка, подъём стека):
curl -fsSL https://raw.githubusercontent.com/yourdrug/RAG-Assistant/main/install.sh | bash

# Или вручную:
task init
# Заполните server/.env.secrets перед запуском.
task up
task pull-model MODEL=qwen2.5:7b
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
| `task init` | server/.env + client/.env + собрать образ |
| `task up` / `task down` | Поднять / остановить стек |
| `task up -- gpu` | Поднять с GPU |
| `task build` | Пересобрать образы |
| `task build -- gpu` | Пересобрать с GPU |
| `task pull-model MODEL=qwen2.5:7b` | Скачать LLM в Ollama |
| `task login email=... password=...` | Залогиниться → `.auth_token` |
| `task ingest` | Проиндексировать `data/docs_sample/` [admin] |
| `task chat -- "вопрос"` | Синхронный запрос |
| `task bench` | Оценка качества |
| `task test` / `task lint` / `task fmt` | pytest / ruff |
| `task db:shell` / `task db:backup` | psql / дамп |
| `task clean` | Удалить все данные |

Полный список: `task --list`

---

## Конфигурация

`task env` создаёт файлы из шаблонов без перезаписи:

- **`server/.env`** — PostgreSQL/Redis bootstrap, CORS, S3 и Docker deployment.
- **`server/.env.secrets`** — пароли и API-ключи; права `0600`, заполнить до запуска.
- **`client/.env`** — `VITE_API_URL`, применяется при сборке Vite.

LLM/OCR/RAG, ML endpoints, квоты, retention и corpus хранятся в PostgreSQL и
управляются через админку. Состояние применения проверяйте отдельно от сохранения.
Для локального TEI задайте `COMPOSE_PROFILES=tei`; для облачного ML оставьте профиль
пустым. `ML_PROVIDER` в БД выбирает провайдера приложения, а не Docker-сервисы.
`task pull-model MODEL=...` только скачивает модель; её выбор для запросов задаётся в БД.

Существующий env сокращайте через offline `config migrate-env --trim-env` после
backup и сверки dry run, сохраняя нестандартные static overrides. Полный порядок,
ограничения применения и откат: [конфигурация](docs/operations/configuration.md).

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

## Продакшн (домен + TLS)

Стек внутри Docker не завершает TLS — это делает внешний nginx на хосте с certbot/letsencrypt.

Порты backend (`8001`) и клиентского nginx (`3001`) опубликованы только на `127.0.0.1`.
Внешний доступ к UI и API идёт через HTTPS-домен. Для доступа к API из curl, Postman
или другого приложения UI не требуется: используйте `https://rag.example.com/api/`.
Внутри Docker-сети клиентский nginx обращается к backend по адресу `server:8001`.

Клиентский контейнер уже запускает свой nginx, который проксирует `/api/*` на `server:8001` с таймаутами 600s для SSE-стриминга. Внешнему nginx достаточно проксировать один upstream:

```nginx
upstream rag { server 127.0.0.1:3001; }

server {
    listen 443 ssl;
    server_name rag.example.com;
    ssl_certificate     /etc/letsencrypt/live/rag.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/rag.example.com/privkey.pem;

    location / {
        proxy_pass http://rag;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_buffering off;
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
        client_max_body_size 100m;
    }
}
```

Пример запроса к API (замените домен своим, а `TOKEN` — действующим JWT):

```bash
curl https://rag.example.com/api/documents \
  -H "Authorization: Bearer $TOKEN"
```

После изменения привязки портов пересоздайте контейнеры server и client: обычный
`docker compose restart` не применяет изменения публикации портов.

Деплой на сервер: `./deploy/deploy.sh 0.6.0` (или `./deploy/deploy.sh 0.6.0 --gpu`). Образы тянутся из GHCR.
Скрипт загружает env-файлы, использует `COMPOSE_PROFILES`, сохраняет работающие контейнеры
хранилищ и проверяет `/api/ready` после обновления приложения.

Эксплуатационные утилиты находятся в `scripts/operations/`: `backup.sh` создаёт
холодный бэкап с остановкой стека и проверкой восстановления PostgreSQL,
`restore.sh BACKUP_DIRECTORY NEW_DATA_DIRECTORY` распаковывает его в новую директорию,
`check_health.py` проверяет API, диск и давность бэкапа. Настройки —
`scripts/operations/operations.env.example`, шаблоны systemd — `deploy/systemd/`
(пути в них рассчитаны на `/opt/rag-assistant`). Автоматический откат образов
блокируется при изменении схемы БД, если совместимость явно не подтверждена.

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

**`Error: EOF` при ollama pull** → MTU слишком низкий (VPN). `export DOCKER_MTU=1400 && task down && task up`

**`401 Unauthorized`** → Токен истёк. `task login email=... password=...` заново.

**`403 Forbidden` на /ingest** → Нужна роль admin. `task me` проверить роль.
