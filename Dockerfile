# -----------------------------------------------------------------------------------
# Python-base stage sets up all shared env vars
FROM python:3.12-slim AS python-base

# Metadata (author email)
LABEL author.email="mitkojenia@gmail.com"

# Metadata (author name)
LABEL author.name="Eugene Mitsko"

# Metadata (build version)
ARG SOURCE_VERSION

# Important environment variables
ENV                                                                                 \
    # Force stdout and stderr streams to be unbuffered
    PYTHONUNBUFFERED=1                                                              \
    # Prevents python from creating .pyc files
    PYTHONDONTWRITEBYTECODE=1                                                       \
    # Disable project installation (this is not a Python package)
    UV_NO_INSTALL_PROJECT=1                                                         \
    # Compile bytecode for faster startup (adds build time, saves runtime)
    UV_COMPILE_BYTECODE=1                                                           \
    # Use copy mode for uv (faster than symlink, safer for containers)
    UV_LINK_MODE=copy                                                               \
    # Root of the project inside the container
    ROOT_DIR="/code"                                                                \
    # Make uv to be installed into this location
    UV_INSTALL_DIR="/usr/local/bin"                                                 \
    # This is where pyproject.toml and uv.lock live
    PYSETUP_PATH="/code/project"                                                    \
    # This is where the virtual environment will live
    VENV_PATH="/code/.venv"                                                         \
    # Tell uv to create venv at this path
    UV_PROJECT_ENVIRONMENT="/code/.venv"                                            \
    # Increase download timeout for large packages
    UV_HTTP_TIMEOUT=300                                                             \
    # Source version (passed from docker-compose or build arg)
    SOURCE_VERSION=${SOURCE_VERSION}

# Prepend venv bin to PATH so that python from venv is found first
ENV PATH="/code/.venv/bin:/usr/local/bin:$PATH"

# -----------------------------------------------------------------------------------
# Shared system dependencies for build and runtime
FROM python-base AS builder-base

# Install build dependencies
RUN                                                                                 \
    # Update package index
    apt-get update &&                                                               \
    # Install packages and clean up cache in one layer
    apt-get install -y --no-install-recommends                                      \
    # Bash — better scripting support than default /bin/sh
    bash                                                                            \
    # Curl — for downloading files and making HTTP requests
    curl                                                                            \
    # libmagic1 — required by python-magic (file type detection)
    libmagic1                                                                       \
    # libgomp1 — required by OpenMP
    libgomp1                                                                        \
    &&                                                                              \
    # Clean up apt cache to reduce layer size
    rm -rf /var/lib/apt/lists/*

# Override default shell from sh to bash for better scripting support
SHELL ["/bin/bash", "-c"]

# -----------------------------------------------------------------------------------
# Tokenizer is fetched and verified at build time, never at application startup.
FROM builder-base AS embedding-tokenizer
RUN mkdir -p /tokenizers && \
    curl --fail --location --retry 3 \
      https://huggingface.co/BAAI/bge-m3/resolve/5617a9f61b028005a4858fdac845db406aefb181/tokenizer.json \
      -o /tokenizers/bge-m3.json && \
    echo "21106b6d7dab2952c1d496fb21d5dc9db75c28ed361a05f5020bbba27810dd08  /tokenizers/bge-m3.json" | sha256sum --check

# -----------------------------------------------------------------------------------
# Shared Python dependency stage
FROM builder-base AS uv-base

# Copy uv binary from official image (faster and more reliable than curl install)
COPY --from=ghcr.io/astral-sh/uv:0.9.0 /uv $UV_INSTALL_DIR/uv

# Set working directory to project root
WORKDIR $PYSETUP_PATH

# Copy dependency files first (better Docker layer caching)
# If only pyproject.toml or uv.lock changes, this layer is cached
COPY server/pyproject.toml server/uv.lock ./

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv                                       \
    set -e;                                                                         \
    # ";" keeps a failed sync fatal: the "|| true" cleanup below would mask it
    uv sync --frozen --no-dev;                                                      \
    # Remove __pycache__ directories (saves ~50-100MB)
    find /code/.venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null       \
        || true &&                                                                  \
    # Remove .pyc compiled bytecode files
    find /code/.venv -name "*.pyc" -delete 2>/dev/null || true &&                   \
    # Remove test directories from installed packages
    find /code/.venv -type d -name "tests" -exec rm -rf {} + 2>/dev/null            \
        || true &&                                                                  \
    find /code/.venv -type d -name "test" -exec rm -rf {} + 2>/dev/null             \
        || true

# -----------------------------------------------------------------------------------
# Development server with hot reload and uv available in the container.
FROM uv-base AS development-cpu

# Copy Alembic config (for database migrations)
COPY server/alembic.ini ./

# Copy entrypoint script (sets up env vars, then exec's CMD)
COPY server/entrypoint.sh ./

# Copy VERSION file (displayed on startup)
COPY VERSION ./

# Copy application source code
COPY --from=embedding-tokenizer /tokenizers /code/tokenizers
ENV EMBED_TOKENIZER_PATH=/code/tokenizers/bge-m3.json
COPY server/app ./app

# Entrypoint sets up env vars and exec's the CMD
ENTRYPOINT ["./entrypoint.sh"]

# Dev server with hot-reload on port 8001
CMD [                                                                               \
    "python",                                                                       \
    "main.py",                                                                      \
    "runserver",                                                                    \
    "--host",                                                                       \
    "0.0.0.0",                                                                      \
    "--port",                                                                       \
    "8001",                                                                         \
    "--reload"                                                                      \
]

# Container will listen on this port
EXPOSE 8001

# -----------------------------------------------------------------------------------
# Production server with only runtime dependencies.
FROM builder-base AS prod-cpu

# Create non-root user for running the application
RUN                                                                                 \
    # Create group named "raguser" with GID 1001
    addgroup --system --gid 1001 raguser &&                                         \
    # Create user named "raguser" with UID 1001, member of "raguser" group
    # --home sets HOME so that ~ resolves correctly (default is /nonexistent)
    adduser --system --uid 1001 --ingroup raguser --home /home/raguser raguser && \
    mkdir -p /home/raguser && chown raguser:raguser /home/raguser

ENV HOME=/home/raguser

# Copy the shared venv (with --chown to avoid extra chown layer)
COPY --from=uv-base --chown=1001:1001 $VENV_PATH $VENV_PATH

# Set working directory
WORKDIR $PYSETUP_PATH

# Copy application files with correct ownership
COPY --chown=1001:1001 server/alembic.ini ./
COPY --chown=1001:1001 server/entrypoint.sh ./
COPY --chown=1001:1001 VERSION ./
RUN if [ -n "$SOURCE_VERSION" ]; then printf '%s\n' "$SOURCE_VERSION" > VERSION; fi
COPY --from=embedding-tokenizer /tokenizers /code/tokenizers
ENV EMBED_TOKENIZER_PATH=/code/tokenizers/bge-m3.json
COPY --chown=1001:1001 server/app ./app

# Switch to non-root user
USER 1001

# Entrypoint sets up env vars and exec's the CMD
ENTRYPOINT ["./entrypoint.sh"]

# Production server (no hot-reload, single worker)
CMD [                                                                               \
    "python",                                                                       \
    "main.py",                                                                      \
    "runserver",                                                                    \
    "--host",                                                                       \
    "0.0.0.0",                                                                      \
    "--port",                                                                       \
    "8001"                                                                          \
]

# Container will listen on this port
EXPOSE 8001

# -----------------------------------------------------------------------------------

# Retain the existing build target names for Compose and release automation.
FROM development-cpu AS development-gpu
FROM prod-cpu AS prod-gpu
