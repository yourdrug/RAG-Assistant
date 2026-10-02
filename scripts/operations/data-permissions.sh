#!/usr/bin/env bash
# Run explicitly as root on initial setup / maintenance, never during a live backup.
set -euo pipefail
[[ "$EUID" == 0 ]] || { echo "Run with sudo to set container ownership" >&2; exit 1; }
cd "${DEPLOY_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
# postgres:16.4-alpine uses UID/GID 70; redis:7-alpine uses 999/1000.
# Override these when changing images; verify with `docker run --rm --entrypoint id IMAGE USER`.
install -d -m 0750 -o 1001 -g 1001 data
for spec in "postgres:${POSTGRES_DATA_UID:-70}:${POSTGRES_DATA_GID:-70}" "redis:${REDIS_DATA_UID:-999}:${REDIS_DATA_GID:-1000}" "qdrant_storage:0:0" "minio:0:0" "ollama_models:0:0" "tei_cache:0:0" "paddleocr_cache:1001:1001"; do
  IFS=: read -r directory uid gid <<<"$spec"
  install -d -m 0750 -o "$uid" -g "$gid" "data/$directory"
  chown -R "$uid:$gid" "data/$directory"
  chmod -R u+rwX,g+rX,o-rwx "data/$directory"
done
# Config files mounted read-only remain readable by service users.
chmod 0755 data/redis
[[ ! -f data/redis/redis.conf ]] || chmod 0644 data/redis/redis.conf
