#!/usr/bin/env bash
# Usage: ./deploy/deploy.sh [v]0.6.0 [--gpu]
set -Eeuo pipefail
VERSION="${1:?Usage: ./deploy/deploy.sh <version> [--gpu]}"
VERSION="${VERSION#v}"
[[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][a-zA-Z0-9.-]+)?$ ]] || { echo "Invalid release version" >&2; exit 1; }
shift
export VERSION STAGE=production BUILD_TARGET=cpu
for arg in "$@"; do
  case "$arg" in --gpu) export BUILD_TARGET=gpu ;; *) echo "Unknown option: $arg" >&2; exit 1 ;; esac
done
DEPLOY_DIR="${DEPLOY_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "$DEPLOY_DIR/scripts/operations/compose.sh"
mkdir -p .deploy
exec 9>.deploy/operations.lock
flock -n 9 || { echo "Another deploy/backup is running" >&2; exit 1; }
# Must match the lowercase ghcr.io/<owner>/<repo> that the Build workflow pushes to.
IMAGE_REPO="${IMAGE_REPO:-ghcr.io/yourdrug/rag-assistant}"
export SERVER_IMAGE="${IMAGE_REPO}:production-${BUILD_TARGET}-${VERSION}"
export CLIENT_IMAGE="${IMAGE_REPO}/client:${VERSION}"
python3 "$DEPLOY_DIR/deploy/metrics-token.py"
compose config --quiet
# One image per service across all replicas; app services must share the server image.
service_image() {
  local images
  images=$(compose ps -q "$1" | xargs -r docker inspect --format '{{.Image}}' | sort -u)
  if [[ "$images" == *$'\n'* ]]; then
    echo "Replicas of $1 run different images; resolve before deployment" >&2
    return 1
  fi
  printf '%s' "$images"
}
old_server=$(service_image server)
old_client=$(service_image client)
old_worker=$(service_image worker)
old_worker_batch=$(service_image worker-batch)
old_maintenance=$(service_image maintenance)
for old_image in "$old_worker" "$old_worker_batch" "$old_maintenance"; do
  if [[ -n "$old_image" && "$old_image" != "$old_server" ]]; then
    echo "Server and workers run different images; resolve before deployment" >&2
    exit 1
  fi
done
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
      # Previous releases may not have /ready; require healthy JSON from the legacy endpoint.
      cat >.deploy/rollback.compose.yml <<'YAML'
services:
  server:
    healthcheck:
      test: ["CMD-SHELL", "curl --silent --fail http://localhost:8001/health | python -c 'import json,sys; sys.exit(json.load(sys.stdin)[\"status\"] != \"healthy\")'"]
YAML
      compose_args+=(-f .deploy/rollback.compose.yml)
      local rollback_services=(server worker client)
      if [[ -n "$old_maintenance" ]]; then
        rollback_services+=(maintenance)
      else
        compose stop maintenance
      fi
      if [[ -n "$old_worker_batch" ]]; then
        rollback_services+=(worker-batch)
      else
        # The previous release predates the documents/batch split: its worker
        # serves every task type and does not know the --role documents flag.
        compose stop worker-batch
        printf 'services:\n  worker:\n    command: ["python", "main.py", "worker"]\n' >.deploy/rollback.legacy-worker.compose.yml
        compose_args+=(-f .deploy/rollback.legacy-worker.compose.yml)
      fi
      if compose up -d --no-deps --force-recreate --pull never --wait --wait-timeout "${DEPLOY_WAIT_TIMEOUT:-600}" "${rollback_services[@]}"; then
        echo "Previous images restored; deployment remains failed" >&2
      else
        echo "Rollback failed; inspect containers immediately" >&2
      fi
    else
      echo "Schema changed or cannot be checked; automatic image rollback blocked. Inspect the schema and restore a verified backup before retrying an incompatible release." >&2
    fi
  fi
  exit "$failed_status"
}
trap rollback ERR
# Add one new-image replica, wait for health, then remove one old replica.

rolling_replace() {
  local service=$1 grace=$2 old replacement id desired count deadline ids expected actual
  local -a originals current before
  ids=$(compose ps -q "$service") || return 1
  if [[ -z "$ids" ]]; then
    compose up -d --no-deps --pull never --wait --wait-timeout "${DEPLOY_WAIT_TIMEOUT:-600}" "$service"
    return
  fi
  mapfile -t originals <<<"$ids"
  if [[ "$service" == client ]]; then
    expected=$(docker image inspect --format '{{.Id}}' "$CLIENT_IMAGE") || return 1
  else
    expected=$(docker image inspect --format '{{.Id}}' "$SERVER_IMAGE") || return 1
  fi
  count=${#originals[@]}
  desired=$(compose config --format json | python3 -c 'import json,sys; print(json.load(sys.stdin)["services"][sys.argv[1]].get("deploy", {}).get("replicas", sys.argv[2]))' "$service" "$count")
  for old in "${originals[@]}"; do
    ids=$(compose ps -q "$service") || return 1
    mapfile -t before <<<"$ids"
    # --no-recreate leaves every existing replica intact while adding one from
    # the new image. Never stop an old owner before its replacement is healthy.
    compose up -d --no-deps --no-recreate --pull never --scale "$service=$((count + 1))" --wait --wait-timeout "${DEPLOY_WAIT_TIMEOUT:-600}" "$service"
    ids=$(compose ps -q "$service") || return 1
    mapfile -t current <<<"$ids"
    replacement=""
    for id in "${current[@]}"; do
      if [[ " ${before[*]} " != *" $id "* ]]; then replacement=$id; fi
    done
    [[ -n "$replacement" ]] || { echo "No replacement found for $service" >&2; return 1; }
    actual=$(docker inspect --format '{{.Image}}' "$replacement") || return 1
    [[ "$actual" == "$expected" ]] || { echo "Replacement for $service has an unexpected image" >&2; return 1; }
    if [[ "$service" == server ]]; then
      deadline=$((SECONDS + ${DEPLOY_WAIT_TIMEOUT:-600}))
      until docker exec "$replacement" curl --fail --silent --max-time 5 http://localhost:8001/ready >/dev/null; do
        ((SECONDS < deadline)) || { echo "Replacement API did not become ready" >&2; return 1; }
        sleep 2
      done
    fi
    if [[ "$service" == server || "$service" == client ]]; then
      sleep "${DEPLOY_DNS_SETTLE_SECONDS:-12}"
    fi
    docker stop --time "$grace" "$old" >/dev/null
    docker rm "$old" >/dev/null
  done
  # Honor an explicitly configured scale change once the old image is gone.
  compose up -d --no-deps --no-recreate --pull never --scale "$service=$desired" --wait --wait-timeout "${DEPLOY_WAIT_TIMEOUT:-600}" "$service"
}
# Download every enabled image before touching the application. Preserve old image IDs.
# Locally built services (ocr-vl profile) have no registry image.
compose pull --ignore-buildable
infra=(postgres redis qdrant minio pgbouncer)
enabled_services=$(compose config --services)
if grep -qx tei-embed <<<"$enabled_services"; then infra+=(tei-embed tei-rerank); fi
# Opt-in GPU services enabled through COMPOSE_PROFILES (vllm, ocr-vl).
for optional in vllm ocr-vl; do
  if grep -qx "$optional" <<<"$enabled_services"; then infra+=("$optional"); fi
done
if grep -qx prometheus <<<"$enabled_services"; then infra+=(prometheus pgbouncer-exporter); fi
# Infrastructure upgrades are a separate maintenance operation.
compose up -d --no-recreate --wait --wait-timeout "${DEPLOY_WAIT_TIMEOUT:-600}" "${infra[@]}"
pg_limit=$(compose exec -T postgres sh -c 'PGPASSWORD="$POSTGRES_PASSWORD" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Atc "SHOW max_connections"')
compose config --format json | python3 "$DEPLOY_DIR/deploy/capacity.py" "$pg_limit"
if [[ -n "$old_server" ]]; then before_schema=$(schema_revision); fi
# Trust only the host gateway for real-IP headers, never the whole Docker subnet.
if [[ -z "${TRUSTED_PROXY_CIDR:-}" ]]; then
  export TRUSTED_PROXY_CIDR
  TRUSTED_PROXY_CIDR=$(docker network inspect rag-network --format '{{(index .IPAM.Config 0).Gateway}}')
fi
legacy_frontend=false
while IFS= read -r frontend_id; do
  [[ -n "$frontend_id" ]] || continue
  if [[ $(docker inspect --format '{{if index .HostConfig.PortBindings "80/tcp"}}legacy{{end}}' "$frontend_id") == legacy ]]; then
    legacy_frontend=true
  fi
done < <(compose ps -q client)
if [[ "$legacy_frontend" == true ]]; then
  # Bring the stable proxy up on a temporary local port so new frontend
  # replicas can already resolve and trust it before the public-port handoff.
  export GATEWAY_HOST_PORT=0
fi
compose up -d --no-deps --no-recreate --pull never --wait --wait-timeout "${DEPLOY_WAIT_TIMEOUT:-600}" gateway
# Migrations must be backward-compatible with the running release (expand/contract).
migrate_container="rag-migrate-$$"
# timeout kills only the compose client; remove the container so it cannot keep the advisory lock and pending DDL.
if ! timeout "${DEPLOY_MIGRATION_TIMEOUT:-300}" docker compose "${compose_args[@]}" run --rm --no-deps --name "$migrate_container" migrate; then
  docker rm -f "$migrate_container" >/dev/null 2>&1 || true
  echo "Migration failed or timed out; application containers left on the previous release" >&2
  exit 1
fi
update_started=true
rolling_replace server 60
rolling_replace worker 120
rolling_replace worker-batch 120
rolling_replace maintenance 120
rolling_replace client 60
if [[ "$legacy_frontend" == true ]]; then
  unset GATEWAY_HOST_PORT
  compose up -d --no-deps --force-recreate --pull never --wait --wait-timeout "${DEPLOY_WAIT_TIMEOUT:-600}" gateway
  while IFS= read -r frontend_id; do
    # set_real_ip_from resolves the stable proxy's address when nginx loads its config.
    docker exec "$frontend_id" nginx -s reload
  done < <(compose ps -q client)
fi
# Test the same /api/ proxy path used by external API clients.
curl --fail --silent --show-error --max-time 30 http://127.0.0.1:3001/api/ready >/dev/null
state_tmp=$(mktemp .deploy/current.env.XXXXXX)
printf 'VERSION=%s\nBUILD_TARGET=%s\nSERVER_IMAGE=%s\nCLIENT_IMAGE=%s\n' "$VERSION" "$BUILD_TARGET" "$SERVER_IMAGE" "$CLIENT_IMAGE" >"$state_tmp"
mv "$state_tmp" .deploy/current.env
trap - ERR
echo "Deploy ${VERSION} (${BUILD_TARGET}) complete"
