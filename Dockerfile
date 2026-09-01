# -----------------------------------------------------------------------------------
# Python-base stage sets up all shared env vars
FROM python:3.11-slim AS python-base

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
    # Source version (passed from docker-compose or build arg)
    SOURCE_VERSION=${SOURCE_VERSION}

# Prepend venv bin to PATH so that python from venv is found first
ENV PATH="/code/.venv/bin:/usr/local/bin:$PATH"

# -----------------------------------------------------------------------------------
# Builder-base stage installs all necessary system deps for building Python extensions
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
    # libgl1 — required by PyMuPDF and PaddleOCR (OpenCV)
    libgl1                                                                          \
    # libglib2.0-0 — required by PaddleOCR (OpenCV)
    libglib2.0-0                                                                    \
    # libgomp1 — required by OpenMP and PaddlePaddle
    libgomp1                                                                        \
    &&                                                                              \
    # Clean up apt cache to reduce layer size
    rm -rf /var/lib/apt/lists/*

# Override default shell from sh to bash for better scripting support
SHELL ["/bin/bash", "-c"]

# -----------------------------------------------------------------------------------
# CPU: uv-base stage installs uv, creates venv and installs CPU dependencies
# This stage uses the CPU-only PyTorch (~300MB instead of ~2.5GB with CUDA)
FROM builder-base AS uv-base-cpu

# Copy uv binary from official image (faster and more reliable than curl install)
COPY --from=ghcr.io/astral-sh/uv:0.9.0 /uv $UV_INSTALL_DIR/uv

# Set working directory to project root
WORKDIR $PYSETUP_PATH

# Copy dependency files first (better Docker layer caching)
# If only pyproject.toml or uv.lock changes, this layer is cached
COPY server/pyproject.toml server/uv.lock ./

# Install dependencies — uv resolves torch from pytorch-cpu index via [tool.uv.sources]
RUN --mount=type=cache,target=/root/.cache/uv                                       \
    uv sync --frozen --no-dev --extra cpu &&                                        \
    # Remove __pycache__ directories (saves ~50-100MB)
    find /code/.venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null      \
        || true &&                                                                  \
    # Remove .pyc compiled bytecode files
    find /code/.venv -name "*.pyc" -delete 2>/dev/null || true &&                   \
    # Remove test directories from installed packages
    find /code/.venv -type d -name "tests" -exec rm -rf {} + 2>/dev/null            \
        || true &&                                                                  \
    find /code/.venv -type d -name "test" -exec rm -rf {} + 2>/dev/null             \
        || true

# -----------------------------------------------------------------------------------
# GPU: uv-base stage installs uv, creates venv and installs GPU dependencies
# This stage uses PyTorch with CUDA (~2.5GB, requires NVIDIA GPU at runtime)
FROM builder-base AS uv-base-gpu

# Copy uv binary from official image
COPY --from=ghcr.io/astral-sh/uv:0.9.0 /uv $UV_INSTALL_DIR/uv

# Set working directory to project root
WORKDIR $PYSETUP_PATH

# Copy dependency files first (better Docker layer caching)
COPY server/pyproject.toml server/uv.lock ./

# Install dependencies with GPU extra (includes torch with CUDA)
RUN --mount=type=cache,target=/root/.cache/uv                                       \
    uv sync --frozen --no-dev --extra gpu

# -----------------------------------------------------------------------------------
# CPU: development stage — live-reload dev image for local development
# docker-compose.yml uses: target: development-cpu
FROM builder-base AS development-cpu

# Set working directory
WORKDIR $PYSETUP_PATH

# Copy uv binary from CPU builder
COPY --from=uv-base-cpu $UV_INSTALL_DIR/uv $UV_INSTALL_DIR/uv

# Copy pre-built venv from CPU builder (saves ~10min of reinstalling deps)
COPY --from=uv-base-cpu $VENV_PATH $VENV_PATH

# Fix Python executable permissions (may be broken after copy)
#RUN chmod +x $VENV_PATH/bin/python $VENV_PATH/bin/python3

# Copy dependency files (needed for uv sync to verify consistency)
COPY server/pyproject.toml server/uv.lock ./

# Re-sync to add dev dependencies (testing, linting, etc.)
RUN --mount=type=cache,target=/root/.cache/uv                                       \
    uv sync --frozen --extra cpu &&                                                 \
    # Clean up __pycache__ and .pyc files
    find /code/.venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null      \
        || true &&                                                                  \
    find /code/.venv -name "*.pyc" -delete 2>/dev/null || true &&                   \
    # Clean up test directories
    find /code/.venv -type d -name "tests" -exec rm -rf {} + 2>/dev/null            \
        || true &&                                                                  \
    find /code/.venv -type d -name "test" -exec rm -rf {} + 2>/dev/null             \
        || true

# Copy Alembic config (for database migrations)
COPY server/alembic.ini ./

# Copy entrypoint script (sets up env vars, then exec's CMD)
COPY server/entrypoint.sh ./

# Copy VERSION file (displayed on startup)
COPY VERSION ./

# Copy application source code
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
# GPU: development stage — live-reload dev image with CUDA support
# docker-compose.yml uses: target: development-gpu
FROM builder-base AS development-gpu

# Set working directory
WORKDIR $PYSETUP_PATH

# Copy uv binary from GPU builder
COPY --from=uv-base-gpu $UV_INSTALL_DIR/uv $UV_INSTALL_DIR/uv

# Copy pre-built venv from GPU builder (includes CUDA torch)
COPY --from=uv-base-gpu $VENV_PATH $VENV_PATH

# Fix Python executable permissions
#RUN chmod +x $VENV_PATH/bin/python $VENV_PATH/bin/python3

# Copy dependency files
COPY server/pyproject.toml server/uv.lock ./

# Re-sync to add dev dependencies
RUN --mount=type=cache,target=/root/.cache/uv                                       \
    uv sync --frozen --extra gpu

# Copy Alembic config
COPY server/alembic.ini ./

# Copy entrypoint script
COPY server/entrypoint.sh ./

# Copy VERSION file
COPY VERSION ./

# Copy application source code
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
# CPU: production stage — minimal standalone image, non-root user, no dev deps
# Uses COPY --chown to avoid a separate chown layer (~2.5GB savings)
FROM python-base AS prod-cpu

# Install runtime dependencies only (no build tools)
RUN                                                                                 \
    # Update package index
    apt-get update &&                                                               \
    # Install packages and clean up cache in one layer
    apt-get install -y --no-install-recommends                                      \
    # Bash — better scripting support than default /bin/sh
    bash                                                                            \
    # Curl — for healthchecks and HTTP requests
    curl                                                                            \
    # libmagic1 — required by python-magic (file type detection)
    libmagic1                                                                       \
    # libgl1 — required by PyMuPDF and PaddleOCR (OpenCV)
    libgl1                                                                          \
    # libglib2.0-0 — required by PaddleOCR (OpenCV)
    libglib2.0-0                                                                    \
    # libgomp1 — required by OpenMP and PaddlePaddle
    libgomp1                                                                        \
    &&                                                                              \
    # Clean up apt cache to reduce layer size
    rm -rf /var/lib/apt/lists/*

# Override default shell from sh to bash for better scripting support
SHELL ["/bin/bash", "-c"]

# Create non-root user for running the application
RUN                                                                                 \
    # Create group named "raguser" with GID 1001
    addgroup --system --gid 1001 raguser &&                                         \
    # Create user named "raguser" with UID 1001, member of "raguser" group
    adduser --system --uid 1001 --ingroup raguser raguser

# Copy pre-built venv from CPU builder (with --chown to avoid extra chown layer)
COPY --from=uv-base-cpu --chown=1001:1001 $VENV_PATH $VENV_PATH

# Set working directory
WORKDIR $PYSETUP_PATH

# Copy application files with correct ownership
COPY --chown=1001:1001 server/alembic.ini ./
COPY --chown=1001:1001 server/entrypoint.sh ./
COPY --chown=1001:1001 VERSION ./
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
# GPU: production stage — standalone image with CUDA deps, non-root user
FROM python-base AS prod-gpu

# Install runtime dependencies only
RUN                                                                                 \
    # Update package index
    apt-get update &&                                                               \
    # Install packages and clean up cache in one layer
    apt-get install -y --no-install-recommends                                      \
    # Bash — better scripting support than default /bin/sh
    bash                                                                            \
    # Curl — for healthchecks and HTTP requests
    curl                                                                            \
    # libmagic1 — required by python-magic (file type detection)
    libmagic1                                                                       \
    # libgl1 — required by PyMuPDF and PaddleOCR (OpenCV)
    libgl1                                                                          \
    # libglib2.0-0 — required by PaddleOCR (OpenCV)
    libglib2.0-0                                                                    \
    # libgomp1 — required by OpenMP and PaddlePaddle
    libgomp1                                                                        \
    &&                                                                              \
    # Clean up apt cache to reduce layer size
    rm -rf /var/lib/apt/lists/*

# Override default shell from sh to bash for better scripting support
SHELL ["/bin/bash", "-c"]

# Create non-root user for running the application
RUN                                                                                 \
    # Create group named "raguser" with GID 1001
    addgroup --system --gid 1001 raguser &&                                         \
    # Create user named "raguser" with UID 1001, member of "raguser" group
    adduser --system --uid 1001 --ingroup raguser raguser

# Copy uv binary from GPU builder (needed for torch runtime)
COPY --from=uv-base-gpu --chown=1001:1001 $UV_INSTALL_DIR/uv $UV_INSTALL_DIR/uv

# Copy pre-built venv from GPU builder (with CUDA torch)
COPY --from=uv-base-gpu --chown=1001:1001 $VENV_PATH $VENV_PATH

# Set working directory
WORKDIR $PYSETUP_PATH

# Copy application files with correct ownership
COPY --chown=1001:1001 server/alembic.ini ./
COPY --chown=1001:1001 server/entrypoint.sh ./
COPY --chown=1001:1001 VERSION ./
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
