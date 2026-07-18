#!/usr/bin/env bash
#
# install.sh — установка RAG-ассистента одной командой.
#
#   curl -fsSL https://raw.githubusercontent.com/yourdrug/RAG-Assistant/main/install.sh | bash
#
# Что делает:
#   1. Проверяет/ставит Docker и Task (go-task), если их ещё нет (Linux).
#   2. Клонирует репозиторий (если ещё не внутри него).
#   3. Генерирует секреты (пароль Postgres, JWT) и создаёт .env / server/.env.
#   4. Спрашивает домен (опционально) и данные admin-аккаунта.
#   5. Собирает образ и поднимает стек.
#
# Безопасно перезапускать: если .env/server/.env уже есть, установка не трогает секреты
# повторно — просто пересобирает и поднимает стек.
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Оформление
# ---------------------------------------------------------------------------

GREEN=$(printf '\033[0;32m'); YELLOW=$(printf '\033[1;33m')
RED=$(printf '\033[0;31m');   BOLD=$(printf '\033[1m'); NC=$(printf '\033[0m')

log()   { printf "%s==>%s %s\n" "$GREEN" "$NC" "$1"; }
warn()  { printf "%s==>%s %s\n" "$YELLOW" "$NC" "$1"; }
err()   { printf "%s==>%s %s\n" "$RED" "$NC" "$1" >&2; }
die()   { err "$1"; exit 1; }

# Прочитать интерактивный ответ. curl | bash отдаёт скрипту stdin, поэтому обычный
# `read` без -u/</dev/tty молча получит EOF вместо ответа пользователя.
ask() {
    local prompt="$1" default="${2:-}" var=""
    if [ ! -t 0 ] && [ -e /dev/tty ]; then
        { read -r -p "$prompt" var < /dev/tty; } 2>/dev/null || true
    else
        read -r -p "$prompt" var || true
    fi
    echo "${var:-$default}"
}

ask_secret() {
    local prompt="$1" var=""
    if [ ! -t 0 ] && [ -e /dev/tty ]; then
        { read -r -s -p "$prompt" var < /dev/tty; } 2>/dev/null || true
    else
        read -r -s -p "$prompt" var || true
    fi
    echo >&2
    echo "${var:-}"
}

REPO_URL="${INSTALL_REPO_URL:-https://github.com/yourdrug/RAG-Assistant.git}"
REPO_DIR_NAME="RAG-Assistant"

# Заменить значение KEY=... в файле. Через awk, а не sed: значение подставляется как
# непрозрачная строка (awk -v), а не как часть sed-паттерна/замены — так пароль с
# символами |, &, \, / (то, чем люди реально пользуются) не сломает и не исказит замену.
set_kv() {
    local file="$1" key="$2" value="$3"
    awk -v key="$key" -v val="$value" '
        BEGIN { FS="=" }
        $1 == key { print key "=" val; next }
        { print }
    ' "$file" > "${file}.tmp" && mv "${file}.tmp" "$file"
}

# ---------------------------------------------------------------------------
# 0. Баннер
# ---------------------------------------------------------------------------

cat <<'BANNER'

  ██████╗  █████╗  ██████╗
  ██╔══██╗██╔══██╗██╔════╝     RAG-ассистент — установка
  ██████╔╝███████║██║  ███╗    Docker + Qdrant + Ollama + Postgres
  ██╔══██╗██╔══██║██║   ██║    полностью локально, без облака
  ██║  ██║██║  ██║╚██████╔╝
  ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝

BANNER

# ---------------------------------------------------------------------------
# 1. Docker
# ---------------------------------------------------------------------------

OS="$(uname -s)"

install_docker_linux() {
    warn "Docker не найден. Ставлю через официальный скрипт (get.docker.com) — попросит sudo."
    curl -fsSL https://get.docker.com | sh
    if ! groups "$USER" | grep -q docker; then
        sudo usermod -aG docker "$USER" || true
        warn "Добавил $USER в группу docker. Может понадобиться перелогиниться (или: newgrp docker)."
    fi
}

if ! command -v docker >/dev/null 2>&1; then
    case "$OS" in
        Linux) install_docker_linux ;;
        Darwin) die "Docker не найден. Поставь Docker Desktop: https://www.docker.com/products/docker-desktop и запусти установку заново." ;;
        *) die "Docker не найден, а автоустановка поддержана только для Linux. Поставь Docker вручную: https://docs.docker.com/get-docker/" ;;
    esac
fi

if ! docker compose version >/dev/null 2>&1; then
    die "Найден Docker, но не Docker Compose v2 (команда 'docker compose'). Обнови Docker: https://docs.docker.com/compose/install/"
fi

log "Docker + Compose — есть."

# ---------------------------------------------------------------------------
# 2. Task (go-task)
# ---------------------------------------------------------------------------

if ! command -v task >/dev/null 2>&1; then
    warn "Task не найден. Ставлю в /usr/local/bin (попросит sudo, если нужны права)."
    if [ -w /usr/local/bin ]; then
        sh -c "$(curl -fsSL https://taskfile.dev/install.sh)" -- -d -b /usr/local/bin
    else
        curl -fsSL https://taskfile.dev/install.sh -o /tmp/task_install.sh
        sudo sh /tmp/task_install.sh -d -b /usr/local/bin
        rm -f /tmp/task_install.sh
    fi
fi

log "Task — есть ($(task --version 2>/dev/null || echo 'ok'))."

# ---------------------------------------------------------------------------
# 3. Репозиторий
# ---------------------------------------------------------------------------

if [ -f "docker-compose.yml" ] && [ -f "Taskfile.yml" ] && [ -d "server" ]; then
    log "Уже внутри репозитория ($(pwd)) — клонировать не нужно."
elif [ -d "$REPO_DIR_NAME" ]; then
    log "Каталог $REPO_DIR_NAME уже существует — использую его."
    cd "$REPO_DIR_NAME"
else
    log "Клонирую $REPO_URL ..."
    command -v git >/dev/null 2>&1 || die "Нужен git (не найден). Поставь: sudo apt install git / brew install git"
    git clone --depth 1 "$REPO_URL" "$REPO_DIR_NAME"
    cd "$REPO_DIR_NAME"
fi

PROJECT_DIR="$(pwd)"
log "Рабочий каталог: $PROJECT_DIR"

# ---------------------------------------------------------------------------
# 3.5 Права на data/
# ---------------------------------------------------------------------------
#
# server (production-стадия — task prod:up / RAG_USE_PREBUILT=1) работает под
# непривилегированным пользователем внутри контейнера. Если data/ создана на хосте
# другим UID (например, git clone от root) — контейнер не сможет туда писать
# (ingestion.log, реестр индексации, кэш моделей). a+rwX — пишем и читаем все,
# заходим в директории все; для остального (моих файлов, кода) permissions не трогаем.
mkdir -p data
chmod -R a+rwX data
log "data/ создана и доступна на запись."

# ---------------------------------------------------------------------------
# 4. Конфигурация
# ---------------------------------------------------------------------------

ALREADY_CONFIGURED=false
if [ -f ".env" ] && [ -f "server/.env" ]; then
    ALREADY_CONFIGURED=true
    warn "Найдены .env и server/.env — использую существующую конфигурацию, секреты не трогаю."
    warn "Для полной переустановки: rm .env server/.env, потом запусти install.sh заново."
fi

if [ "$ALREADY_CONFIGURED" = false ]; then
    echo
    echo "${BOLD}Куда ставим?${NC}"
    echo "  1) Локально — доступ только на этой машине (http://localhost:8001)"
    echo "  2) На свой домен — с автоматическим HTTPS (нужен DNS, указывающий на этот сервер)"
    MODE="$(ask "Выбор [1/2, по умолчанию 1]: " "1")"

    DOMAIN=""
    ACME_EMAIL="postmaster@localhost"
    if [ "$MODE" = "2" ]; then
        DOMAIN="$(ask "Домен (например rag.компания.com): " "")"
        [ -n "$DOMAIN" ] || die "Домен обязателен для варианта 2."
        ACME_EMAIL="$(ask "Email для уведомлений Let's Encrypt (можно пусто): " "$ACME_EMAIL")"
        warn "Проверь заранее: DNS-A-запись $DOMAIN уже указывает на IP этого сервера,"
        warn "и порты 80/443 открыты в файрволе — иначе сертификат не выпустится."
    fi

    echo
    echo "${BOLD}Admin-аккаунт${NC} (заходить в систему и заводить остальных сотрудников)"
    ADMIN_EMAIL="$(ask "Email админа: " "admin@example.com")"
    ADMIN_PASSWORD="$(ask_secret "Пароль админа (пусто — сгенерировать случайный): ")"
    GENERATED_ADMIN_PASSWORD=false
    if [ -z "$ADMIN_PASSWORD" ]; then
        ADMIN_PASSWORD="$(openssl rand -base64 18 | tr -d '=+/' | cut -c1-20)"
        GENERATED_ADMIN_PASSWORD=true
    fi

    echo
    echo "${BOLD}Модель LLM${NC}"
    echo "  1) qwen2.5:14b     — лучшее качество, ~10 GB RAM (по умолчанию)"
    echo "  2) qwen2.5:7b      — быстрее, ~8 GB RAM"
    echo "  3) mistral-nemo:12b — альтернатива, ~9 GB RAM"
    LLM_CHOICE="$(ask "Выбор [1/2/3, по умолчанию 1]: " "1")"
    case "$LLM_CHOICE" in
        2) LLM_MODEL="qwen2.5:7b" ;;
        3) LLM_MODEL="mistral-nemo:12b" ;;
        *) LLM_MODEL="qwen2.5:14b" ;;
    esac

    # --- Секреты ---
    POSTGRES_PASSWORD="$(openssl rand -hex 24)"
    JWT_SECRET_KEY="$(openssl rand -hex 32)"

    # --- .env (корень, читает docker-compose.yml) ---
    cp .env.example .env
    set_kv .env POSTGRES_PASSWORD "$POSTGRES_PASSWORD"
    set_kv .env DOMAIN "$DOMAIN"
    set_kv .env ACME_EMAIL "$ACME_EMAIL"

    # --- server/.env (настройки приложения) ---
    cp server/.env.example server/.env
    ALLOWED_ORIGINS="*"
    [ -n "$DOMAIN" ] && ALLOWED_ORIGINS="https://${DOMAIN}"
    set_kv server/.env DATABASE_URL "postgresql://raguser:${POSTGRES_PASSWORD}@postgres:5432/ragdb"
    set_kv server/.env LLM_MODEL "$LLM_MODEL"
    set_kv server/.env JWT_SECRET_KEY "$JWT_SECRET_KEY"
    set_kv server/.env ADMIN_EMAIL "$ADMIN_EMAIL"
    set_kv server/.env ADMIN_PASSWORD "$ADMIN_PASSWORD"
    set_kv server/.env ALLOWED_ORIGINS "$ALLOWED_ORIGINS"

    log "Секреты сгенерированы, .env и server/.env созданы."
fi

# ---------------------------------------------------------------------------
# 5. Сборка и запуск
# ---------------------------------------------------------------------------

# RAG_USE_PREBUILT=1 — взять готовый образ из GHCR вместо сборки из исходников
# (быстрее в разы, не требует скачивать/компилировать torch/paddleocr локально).
# По умолчанию выключено — собираем из исходников, это всегда работает, даже
# если релиз ещё ни разу не публиковался.
if [ "${RAG_USE_PREBUILT:-0}" = "1" ]; then
    COMPOSE_FILES="-f docker-compose.prod.yml"
    log "RAG_USE_PREBUILT=1 — тяну готовый образ из GHCR вместо сборки ..."
    # shellcheck disable=SC2086
    docker compose $COMPOSE_FILES pull server
else
    COMPOSE_FILES=""
    log "Собираю образ (первый раз — долго, тянет torch/paddleocr, несколько GB)..."
    log "(быстрее: RAG_USE_PREBUILT=1 перед install.sh — готовый образ вместо сборки)"
    task build
fi

if [ "$ALREADY_CONFIGURED" = false ] && [ "${MODE:-1}" = "2" ]; then
    log "Поднимаю стек с Caddy и авто-HTTPS ..."
    # shellcheck disable=SC2086
    docker compose $COMPOSE_FILES --profile domain up -d
else
    log "Поднимаю стек ..."
    # shellcheck disable=SC2086
    docker compose $COMPOSE_FILES up -d
fi
docker compose ps

# ---------------------------------------------------------------------------
# 6. Модель LLM
# ---------------------------------------------------------------------------

if [ "$ALREADY_CONFIGURED" = false ]; then
    PULL="$(ask "Скачать модель ${LLM_MODEL} прямо сейчас? [Y/n]: " "Y")"
    if [ "$PULL" != "n" ] && [ "$PULL" != "N" ]; then
        log "Качаю ${LLM_MODEL} (может занять от нескольких минут до часа в зависимости от канала)..."
        task pull-model -- "$LLM_MODEL"
    else
        warn "Пропущено. Не забудь: task pull-model"
    fi
fi

# ---------------------------------------------------------------------------
# 7. Итог
# ---------------------------------------------------------------------------

if [ -n "${DOMAIN:-}" ]; then
    URL="https://${DOMAIN}"
else
    URL="http://localhost:8001"
fi

echo
echo "${GREEN}${BOLD}Готово!${NC}"
echo
echo "  Адрес:        $URL"
echo "  Swagger UI:   $URL/docs"
if [ "${ALREADY_CONFIGURED:-false}" = false ]; then
    echo "  Admin email:  ${ADMIN_EMAIL:-см. server/.env}"
    if [ "${GENERATED_ADMIN_PASSWORD:-false}" = true ]; then
        echo "  ${YELLOW}Admin пароль: ${ADMIN_PASSWORD} — сохрани прямо сейчас, второй раз не покажу${NC}"
    fi
fi
echo
echo "  Проверить:    task health"
echo "  Залогиниться: task login email=${ADMIN_EMAIL:-admin@example.com} password='...'"
echo "  Документы:    положи файлы в data/docs_sample/, потом  task ingest"
echo
