# syntax=docker/dockerfile:1

# --- uv: берём готовый статический бинарник, без установки через pip ---
FROM ghcr.io/astral-sh/uv:0.5.11 AS uv

FROM python:3.11-slim AS base

# Системные зависимости
# libmagic1     — определение типа файла (python-magic)
# libgl1        — нужен PyMuPDF и PaddleOCR (cv2)
# libglib2.0-0  — нужен PaddleOCR (cv2)
# libgomp1      — OpenMP, нужен PaddleOCR/PaddlePaddle
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Ставим зависимости отдельным слоем — кэшируется, пока pyproject.toml/uv.lock не меняются.
# tool.uv.package = false в pyproject.toml, поэтому сам проект как пакет не устанавливается —
# один uv sync до копирования кода полностью готовит окружение.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

COPY . .

# Модели PaddleOCR (детекция/распознавание/угол) скачиваются автоматически при первом
# использовании и кэшируются в ~/.paddleocr. Для офлайн-образа можно прогреть кэш здесь:
# RUN .venv/bin/python -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='ru')"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
