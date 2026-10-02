#!/usr/bin/env bash
# Validate archive integrity and actually restore the logical dump away from production.
set -Eeuo pipefail
backup_dir=$(cd "${1:?Usage: restore-check.sh BACKUP_DIRECTORY}" && pwd)
(cd "$backup_dir" && sha256sum -c SHA256SUMS)
tar -tzf "$backup_dir/data.tar.gz" >/dev/null
container="rag-restore-check-$(date +%s)-$$"
cleanup() { docker rm -f "$container" >/dev/null; }
trap cleanup EXIT
docker run -d --name "$container" --network none \
  -e POSTGRES_HOST_AUTH_METHOD=trust -e POSTGRES_DB=restorecheck \
  --mount "type=bind,source=$backup_dir,target=/backup,readonly" \
  postgres:16.4-alpine >/dev/null
ready=false
for attempt in {1..30}; do
  if docker exec "$container" pg_isready -U postgres -d restorecheck >/dev/null 2>&1; then ready=true; break; fi
  sleep 1
done
[[ "$ready" == true ]] || { echo "Restore-check Postgres did not start" >&2; exit 1; }
docker exec "$container" pg_restore --exit-on-error --no-owner --no-privileges -U postgres -d restorecheck /backup/postgres.dump
docker exec "$container" psql -U postgres -d restorecheck -v ON_ERROR_STOP=1 -Atc 'SELECT version_num FROM alembic_version'
echo "Logical restore succeeded; cold archive integrity verified"
