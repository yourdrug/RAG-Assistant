# AGENTS.md — RAG Project

## Quick Commands

```bash
# Development (no Docker)
task install          # Install deps locally via uv
task test             # pytest (DATA_DIR auto-set in tests/conftest.py)
task lint             # ruff check
task fmt              # ruff format + auto-fix

# Docker stack
task init             # First setup: .env + build image
task up               # Start all services (qdrant, ollama, postgres, server)
task up -- public     # Start with Caddy HTTPS (needs DOMAIN in .env)
task up -- gpu        # Start with GPU support
task up -- prod       # Start production mode
task up -- gpu prod public  # All flags combined
task down             # Stop stack
task build            # Rebuild server image (supports -- gpu, -- prod)
task restart -- server  # Restart single service

# Kubernetes (k3d)
task k3d:create       # Create k3d cluster
task helm:infra       # Deploy infra (postgres, redis, qdrant, ollama, minio, ingress)
task helm:upgrade     # Deploy app to k3d
task k8s:status       # Show all app resources
task k8s:logs         # Tail server logs
task k8s:port-forward # Port-forward API to localhost:8001

# CLI commands (via Docker)
docker compose exec server python main.py runserver --host 0.0.0.0 --port 8001
docker compose exec server python main.py ingest run --docs-dir /code/project/data/docs_sample
docker compose exec server python main.py ingest file /code/project/data/docs_sample/report.pdf
docker compose exec server python main.py ingest list
docker compose exec server python main.py benchmark run --questions /code/project/data/test_questions.json
docker compose exec server python main.py pdf-diag run /code/project/data/docs_sample/report.pdf

# Load testing
task loadtest:setup-users  # Create 50 test users via API
task loadtest:smoke        # Smoke test (3 VU, 1 min)
task loadtest:run          # Load test (0→500 VU, 17 min)
task loadtest:spike        # Spike test (50→500 in 30s)
task loadtest:soak         # Soak test (200 VU, 1 hour)
task loadtest:breakpoint   # Breakpoint test (find failure point)
task loadtest:sse          # Locust SSE test (streaming /chat)
```

## Architecture

- **Entry point**: `server/app/main.py` (FastAPI app)
- **Clean architecture layers**: `server/app/domain/` → `server/app/application/` → `server/app/infrastructure/` → `server/app/presentation/`
- **CLI**: `server/app/presentation/cli/` — typer-based CLI, invoked via `python main.py <command>`
- **Entrypoint**: `server/entrypoint.sh` does `alembic upgrade head`, then `cd app` then `exec "$@"`
- **API port**: 8001 (not 8000)
- **DATA_DIR**: defaults to `/code/project/data` inside container; tests use `tests/conftest.py` to set it automatically

## Key Paths

```
server/app/              ← Application code (clean architecture layers)
server/app/domain/       ← Business logic (entities, value objects, exceptions, repos interfaces)
server/app/application/  ← Services, ports (protocols), DTOs
server/app/infrastructure/ ← SQLAlchemy, Qdrant, Ollama, S3 implementations
server/app/presentation/ ← FastAPI routes, middleware, exception handlers
server/app/presentation/cli/  ← CLI commands (runserver, ingest, benchmark, pdf-diag)
server/app/config.py     ← Pydantic settings (reads server/.env)
server/app/infrastructure/logging/logging_config.py  ← Logging config dict
server/tests/            ← pytest tests
server/pyproject.toml    ← Dependencies (uv)
server/uv.lock           ← Locked versions (committed to git)
data/docs_sample/        ← Documents for indexing
loadtest/                ← Load testing (k6 + Locust)
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `runserver --host --port --reload` | Run uvicorn server |
| `ingest run --docs-dir DIR --reset` | Full document ingestion |
| `ingest file PATH --force` | Ingest single file |
| `ingest list` | Show indexed files |
| `benchmark run --questions --out --top-k --judge-model` | RAG quality benchmark |
| `pdf-diag run PATH --dump` | Diagnose PDF before ingestion |

## Environment Variables

Two `.env` files:

```
.env              → POSTGRES_*, DOMAIN, ACME_EMAIL, DOCKER_MTU (docker-compose)
server/.env       → Qdrant, Ollama, JWT, CORS, OCR, DATA_DIR (application)
```

Both `.env.example` files exist as templates. Taskfile reads `server/.env` via `dotenv:`.

## Logging

- **`infrastructure/logging/logging_config.py`**: dict-based config with `default`, `detailed`, `uvicorn` loggers
- Custom filters: `ExceptionFilter`, `LevelThresholdFilter`, `LevelMinFilter`, `RequestIDFilter`
- Applied in `lifespan` via `logging.config.dictConfig()`
- All modules use `logging.getLogger("default")` or `logging.getLogger("detailed")`
- No `print()` calls — everything goes through logger

## Model Preloading

- Models (`bge-m3`, `bge-reranker-v2-m3`) are preloaded during startup in `_preload_models()`
- Checks HuggingFace cache first — skips download if already cached
- First request is not blocked by model loading

## Testing

- `task test` runs pytest from `server/` directory
- Tests mock external services (Qdrant, Ollama) — no real services needed
- `tests/conftest.py` sets `DATA_DIR` to temp directory automatically
- Run single test: `cd server && uv run pytest tests/test_rag_chain.py -v`

## Code Style

- **Formatter/Linter**: ruff (line-length=110, target py311)
- `task fmt` runs both `ruff format` and `ruff check --fix`
- `task lint` runs `ruff check` without modifications
- Pre-commit hooks: ruff check, ruff format, pytest

## Gotchas

- `uv.lock` is committed — run `task lock` after changing `pyproject.toml`
- Server runs from `server/` dir via `task install` or Docker
- Local dev without Docker needs `DATA_DIR` env var or use `task test`
- JWT tokens expire after `JWT_EXPIRE_MINUTES` (default 24h) — re-run `task login`
- `task clean` deletes all data (postgres, qdrant, ollama models) — destructive
- First request after restart loads models (~2.5 min) — preloading mitigates this

## Docker

- Multi-stage build: python-base → builder-base → uv-base → development/production
- Dev mode bind-mounts `server/app/` for live reload
- venv lives at `/code/.venv` (separate from code, survives bind-mount)
- Production image: `docker build --target production -t rag-server:prod .`
- Server command: `python main.py runserver` (not direct uvicorn)
- Services: qdrant, ollama, postgres, server, minio (S3), client (web UI)
