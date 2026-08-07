# -----------------------------------------------------------------------------------
# Python-base — shared env vars for all targets
FROM python:3.11-slim AS python-base

LABEL author.email="mitkojenia@gmail.com"
LABEL author.name="Eugene Mitsko"

ARG SOURCE_VERSION
ARG BUILD_DATE

ENV                                                                                 \
    PYTHONUNBUFFERED=1                                                              \
    PYTHONDONTWRITEBYTECODE=1                                                       \
    ROOT_DIR="/code"                                                                \
    UV_INSTALL_DIR="/usr/local/bin"                                                 \
    PYSETUP_PATH="/code/project"                                                    \
    VENV_PATH="/code/.venv"                                                         \
    UV_PROJECT_ENVIRONMENT="/code/.venv"                                            \
    UV_LINK_MODE=copy                                                               \
    UV_COMPILE_BYTECODE=1                                                           \
    SOURCE_VERSION=${SOURCE_VERSION}                                                \
    BUILD_DATE=${BUILD_DATE}

ENV PATH="/code/.venv/bin:$PATH:$UV_INSTALL_DIR"

# -----------------------------------------------------------------------------------
# Builder-base — system deps shared by CPU and GPU builds
# libmagic1     — python-magic
# libgl1        — PyMuPDF, PaddleOCR (cv2)
# libglib2.0-0  — PaddleOCR (cv2)
# libgomp1      — OpenMP, PaddlePaddle
FROM python-base AS builder-base

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    curl \
    libmagic1 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]

# -----------------------------------------------------------------------------------
# CPU: uv-base — install deps from pytorch-cpu index (~300MB torch)
FROM builder-base AS uv-base-cpu

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR $PYSETUP_PATH

COPY server/pyproject.toml server/uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra cpu

# -----------------------------------------------------------------------------------
# GPU: uv-base — install deps from pytorch-gpu index (~2.5GB with CUDA)
FROM builder-base AS uv-base-gpu

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR $PYSETUP_PATH

COPY server/pyproject.toml server/uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra gpu

# -----------------------------------------------------------------------------------
# CPU: development — live-reload dev image
# docker-compose.yml uses: target: development-cpu
FROM builder-base AS development-cpu

WORKDIR $PYSETUP_PATH

COPY --from=uv-base-cpu $UV_INSTALL_DIR/uv $UV_INSTALL_DIR/uv
COPY --from=uv-base-cpu $VENV_PATH $VENV_PATH
COPY server/pyproject.toml server/uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra cpu

COPY server/alembic.ini ./
COPY server/entrypoint.sh ./
COPY VERSION ./
COPY server/app ./app

ENTRYPOINT ["./entrypoint.sh"]
CMD ["python", "main.py", "runserver", "--host", "0.0.0.0", "--port", "8001", "--reload"]

EXPOSE 8001

# -----------------------------------------------------------------------------------
# GPU: development — live-reload dev image with CUDA deps
# docker-compose.yml uses: target: development-gpu
FROM builder-base AS development-gpu

WORKDIR $PYSETUP_PATH

COPY --from=uv-base-gpu $UV_INSTALL_DIR/uv $UV_INSTALL_DIR/uv
COPY --from=uv-base-gpu $VENV_PATH $VENV_PATH
COPY server/pyproject.toml server/uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra gpu

COPY server/alembic.ini ./
COPY server/entrypoint.sh ./
COPY VERSION ./
COPY server/app ./app

ENTRYPOINT ["./entrypoint.sh"]
CMD ["python", "main.py", "runserver", "--host", "0.0.0.0", "--port", "8001", "--reload"]

EXPOSE 8001

# -----------------------------------------------------------------------------------
# CPU: production — standalone image, non-root, no dev deps
FROM python-base AS production-cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    curl \
    libmagic1 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]

COPY --from=uv-base-cpu $UV_INSTALL_DIR/uv $UV_INSTALL_DIR/uv
COPY --from=uv-base-cpu $VENV_PATH $VENV_PATH

RUN                                                                                 \
    addgroup --system --gid 1001 raguser &&                                        \
    adduser --system --uid 1001 --ingroup raguser raguser &&                       \
    chown -R raguser:raguser $ROOT_DIR

USER raguser

WORKDIR $PYSETUP_PATH

COPY --chown=raguser:raguser server/alembic.ini ./
COPY --chown=raguser:raguser server/entrypoint.sh ./
COPY --chown=raguser:raguser VERSION ./
COPY --chown=raguser:raguser server/app ./app

ENTRYPOINT ["./entrypoint.sh"]

CMD ["python", "main.py", "runserver", "--host", "0.0.0.0", "--port", "8001"]

EXPOSE 8001

# -----------------------------------------------------------------------------------
# GPU: production — standalone image with CUDA deps, non-root
FROM python-base AS production-gpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    curl \
    libmagic1 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]

COPY --from=uv-base-gpu $UV_INSTALL_DIR/uv $UV_INSTALL_DIR/uv
COPY --from=uv-base-gpu $VENV_PATH $VENV_PATH

RUN                                                                                 \
    addgroup --system --gid 1001 raguser &&                                        \
    adduser --system --uid 1001 --ingroup raguser raguser &&                       \
    chown -R raguser:raguser $ROOT_DIR

USER raguser

WORKDIR $PYSETUP_PATH

COPY --chown=raguser:raguser server/alembic.ini ./
COPY --chown=raguser:raguser server/entrypoint.sh ./
COPY --chown=raguser:raguser VERSION ./
COPY --chown=raguser:raguser server/app ./app

ENTRYPOINT ["./entrypoint.sh"]

CMD ["python", "main.py", "runserver", "--host", "0.0.0.0", "--port", "8001"]

EXPOSE 8001
