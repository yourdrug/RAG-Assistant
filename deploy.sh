#!/usr/bin/env bash
# Usage: ./deploy.sh [v]0.6.0 [--gpu]
set -Eeuo pipefail
VERSION="${1:?Usage: ./deploy.sh <version> [--gpu]}"
VERSION="${VERSION#v}"
[[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][a-zA-Z0-9.-]+)?$ ]] || { echo "Invalid release version" >&2; exit 1; }
shift
export VERSION STAGE=production BUILD_TARGET=cpu
for arg in "$@"; do
  case "$arg" in --gpu) export BUILD_TARGET=gpu ;; *) echo "Unknown option: $arg" >&2; exit 1 ;; esac
done
DEPLOY_DIR="${DEPLOY_DIR:-$(cd "$(dirname "$0")" && pwd)}"
source "$DEPLOY_DIR/scripts/operations/compose.sh"
mkdir -p .deploy
exec 9>.deploy/operations.lock
flock -n 9 || { echo "Another deploy/backup is running" >&2; exit 1; }
export SERVER_IMAGE="ghcr.io/yourdrug/rag-assistant:production-${BUILD_TARGET}-${VERSION}"
export CLIENT_IMAGE="ghcr.io/yourdrug/rag-assistant/client:${VERSION}"
compose config --quiet
old_server=$(compose ps -q server)
old_client=$(compose ps -q client)
old_worker=$(compose ps -q worker)
if [[ -n "$old_server" ]]; then old_server=$(docker inspect --format '{{.Image}}' "$old_server"); fi
if [[ -n "$old_client" ]]; then old_client=$(docker inspect --format '{{.Image}}' "$old_client"); fi
if [[ -n "$old_worker" ]]; then old_worker=$(docker inspect --format '{{.Image}}' "$old_worker"); fi
if [[ -n "$old_worker" && "$old_worker" != "$old_server" ]]; then
  echo "Server and worker run different images; resolve before deployment" >&2
  exit 1
fi
schema_revision() {
  compose exec -T postgres sh -c 'PGPASSWORD="$POSTGRES_PASSWORD" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Atc "SELECT version_num FROM alembic_version ORDER BY version_num"'
}
update_started=false
before_schema=""
rollback() {
  local failed_status=$?
  trap - ERR
  if [[ "$update_started" == true && -n "$old_server" && -n "$old_client" ]]; then
    local after_schema
    if after_schema=$(schema_revision) && [[ "$before_schema" == "$after_schema" || "${DEPLOY_SCHEMA_BACKWARD_COMPATIBLE:-false}" == true ]]; then
      echo "Deployment failed; restoring previous application images" >&2
      export SERVER_IMAGE="$old_server" CLIENT_IMAGE="$old_client"
      compose_args+=(-f scripts/operations/rollback.compose.yml)
      if compose up -d --no-deps --force-recreate --pull never --wait --wait-timeout "${DEPLOY_WAIT_TIMEOUT:-600}" server worker client; then
        echo "Previous images restored; deployment remains failed" >&2
      else
        echo "Rollback failed; inspect containers immediately" >&2
      fi
    else
      echo "Schema changed or cannot be checked; automatic image rollback blocked. Follow docs/operations/deployment.md." >&2
    fi
  fi
  exit "$failed_status"
}
trap rollback ERR
# Download every enabled image before touching the application. Preserve old image IDs.
compose pull
infra=(postgres redis qdrant ollama minio)
[[ "$ml_provider" != tei ]] || infra+=(tei-embed tei-rerank)
# Infrastructure upgrades are a separate maintenance operation.
compose up -d --no-recreate --wait --wait-timeout "${DEPLOY_WAIT_TIMEOUT:-600}" "${infra[@]}"
if [[ -n "$old_server" ]]; then before_schema=$(schema_revision); fi
# Trust only the host gateway for real-IP headers, never the whole Docker subnet.
if [[ -z "${TRUSTED_PROXY_CIDR:-}" ]]; then
  export TRUSTED_PROXY_CIDR
  TRUSTED_PROXY_CIDR=$(docker network inspect rag-network --format '{{(index .IPAM.Config 0).Gateway}}')
fi
update_started=true
# Migrations must be backward-compatible with the running release (expand/contract).
timeout "${DEPLOY_MIGRATION_TIMEOUT:-300}" docker compose "${compose_args[@]}" run --rm --no-deps migrate
compose up -d --no-deps --force-recreate --pull never --wait --wait-timeout "${DEPLOY_WAIT_TIMEOUT:-600}" server worker client
# Test the same /api/ proxy path used by external API clients.
curl --fail --silent --show-error --max-time 30 http://127.0.0.1:3001/api/ready >/dev/null
state_tmp=$(mktemp .deploy/current.env.XXXXXX)
printf 'VERSION=%s\nBUILD_TARGET=%s\nSERVER_IMAGE=%s\nCLIENT_IMAGE=%s\n' "$VERSION" "$BUILD_TARGET" "$SERVER_IMAGE" "$CLIENT_IMAGE" >"$state_tmp"
mv "$state_tmp" .deploy/current.env
trap - ERR
echo "Deploy ${VERSION} (${BUILD_TARGET}) complete"
