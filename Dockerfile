# -----------------------------------------------------------------------------------
# Python-base stage sets up all shared env vars
FROM python:3.11-slim AS python-base

LABEL author.email="mitkojenia@gmail.com"
LABEL author.name="Eugene Mitsko"

ARG SOURCE_VERSION
ARG BUILD_DATE

ENV                                                                                 \
    # Force stdout and stderr streams to be unbuffered
    PYTHONUNBUFFERED=1                                                              \
    # Prevents python from creating .pyc files
    PYTHONDONTWRITEBYTECODE=1                                                       \
    # Root of the project
    ROOT_DIR="/code"                                                                \
    # Make uv install into this location
    UV_INSTALL_DIR="/usr/local/bin"                                                 \
    # Код приложения — этот путь в dev-режиме бинд-маунтится поверх (docker-compose.yml),
    # поэтому venv НЕ должен лежать внутри него — иначе bind-mount его затрёт.
    PYSETUP_PATH="/code/project"                                                    \
    # venv живёт отдельно от кода — переживает bind-mount кода в dev-режиме
    VENV_PATH="/code/.venv"                                                         \
    UV_PROJECT_ENVIRONMENT="/code/.venv"                                            \
    # uv-specific: копировать пакеты в venv, а не симлинками на кэш
    UV_LINK_MODE=copy                                                               \
    UV_COMPILE_BYTECODE=1                                                           \
    # Source version / build date (см. VERSION)
    SOURCE_VERSION=${SOURCE_VERSION}                                                \
    BUILD_DATE=${BUILD_DATE}

ENV PATH="/code/.venv/bin:$PATH:$UV_INSTALL_DIR"

# -----------------------------------------------------------------------------------
# Builder-base stage installs all necessary system deps for building + running deps
# libmagic1     — определение типа файла (python-magic)
# libgl1        — нужен PyMuPDF и PaddleOCR (cv2)
# libglib2.0-0  — нужен PaddleOCR (cv2)
# libgomp1      — OpenMP, нужен PaddleOCR/PaddlePaddle
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
# uv-base stage installs uv, creates venv and installs project deps
FROM builder-base AS uv-base

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR $PYSETUP_PATH

# Копируем только манифесты зависимостей — слой кэшируется, пока они не меняются
COPY server/pyproject.toml server/uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# -----------------------------------------------------------------------------------
# Development stage — используется docker-compose.yml (target: development) с
# бинд-маунтом ./server/app поверх $PYSETUP_PATH/app для live-reload при разработке.
# venv, entrypoint.sh и манифесты лежат в $PYSETUP_PATH напрямую — bind-mount app/ их не трогает.
FROM builder-base AS development

WORKDIR $PYSETUP_PATH

COPY --from=uv-base $UV_INSTALL_DIR/uv $UV_INSTALL_DIR/uv
COPY --from=uv-base $VENV_PATH $VENV_PATH
# Манифесты нужны ещё раз в $PYSETUP_PATH, чтобы "uv sync" ниже видел зависимости
COPY server/pyproject.toml server/uv.lock ./

# Ставим ещё и dev-зависимости (pytest, ruff) — их нет в --no-dev слое выше
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

COPY server/entrypoint.sh ./
COPY VERSION ./
COPY server/app ./app

EXPOSE 8001

# -----------------------------------------------------------------------------------
# Production stage — самодостаточный образ без dev-зависимостей, non-root пользователь
FROM python-base AS production

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    curl \
    libmagic1 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]

COPY --from=uv-base $UV_INSTALL_DIR/uv $UV_INSTALL_DIR/uv
COPY --from=uv-base $VENV_PATH $VENV_PATH

RUN                                                                                 \
    addgroup --system --gid 1001 raguser &&                                        \
    adduser --system --uid 1001 --ingroup raguser raguser &&                       \
    chown -R raguser:raguser $ROOT_DIR

USER raguser

WORKDIR $PYSETUP_PATH

COPY --chown=raguser:raguser server/entrypoint.sh ./
COPY --chown=raguser:raguser VERSION ./
COPY --chown=raguser:raguser server/app ./app

ENTRYPOINT ["./entrypoint.sh"]

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]

EXPOSE 8001
