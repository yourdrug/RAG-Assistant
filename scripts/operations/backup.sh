#!/usr/bin/env bash
# Consistent cold backup. Requires a maintenance window and permission to read data/.
set -Eeuo pipefail
umask 077
DEPLOY_DIR="${DEPLOY_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
[[ ! -f "$DEPLOY_DIR/.deploy/current.env" ]] || source "$DEPLOY_DIR/.deploy/current.env"
source "$DEPLOY_DIR/scripts/operations/compose.sh"
[[ ! -f .operations.env ]] || source .operations.env
mkdir -p .deploy
exec 9>.deploy/operations.lock
flock -n 9 || { echo "Another deploy/backup is running" >&2; exit 1; }
backup_root="${BACKUP_DIR:-$DEPLOY_DIR/backups}"
backup_dir="$backup_root/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$backup_dir"
mapfile -t running < <(compose ps --services --status running)
[[ ${#running[@]} -gt 0 ]] || { echo "No running stack to back up" >&2; exit 1; }
resume() { compose up -d --no-deps --no-recreate --wait --wait-timeout 600 "${running[@]}"; }
cleanup() {
  local code=$?
  trap - EXIT
  if ! resume; then echo "Stack restart failed after backup" >&2; exit 1; fi
  exit "$code"
}
trap cleanup EXIT
# Stop application writers before the logical dump; stop stores before file copy.
compose stop server worker client
compose exec -T postgres sh -c 'PGPASSWORD="$POSTGRES_PASSWORD" pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc' >"$backup_dir/postgres.dump"
compose stop "${running[@]}"
tar -C data -czf "$backup_dir/data.tar.gz" postgres qdrant_storage minio redis
compose config --images >"$backup_dir/images.txt"
[[ ! -f .deploy/current.env ]] || cp .deploy/current.env "$backup_dir/release.env"
(cd "$backup_dir" && sha256sum postgres.dump data.tar.gz images.txt > SHA256SUMS)
resume
trap - EXIT
# Exercise logical restoration on every backup in a disposable isolated Postgres.
"$DEPLOY_DIR/scripts/operations/restore-check.sh" "$backup_dir"
if [[ -n "${BACKUP_REMOTE:-}" ]]; then
  rsync -a -- "$backup_dir" "$BACKUP_REMOTE"
else
  echo "BACKUP_REMOTE is not configured; backup is LOCAL ONLY" >&2
fi
date -u +%s > .deploy/last-backup-success
printf 'Backup verified: %s\n' "$backup_dir"
