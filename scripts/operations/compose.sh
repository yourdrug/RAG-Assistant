#!/usr/bin/env bash
# Shared production Compose arguments. Source from deployment/operations scripts.
DEPLOY_DIR="${DEPLOY_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$DEPLOY_DIR"
export VERSION="${VERSION:-$(cat VERSION)}"
export STAGE="${STAGE:-production}"
export APP_STAGE="${APP_STAGE:-prod}"
export BUILD_TARGET="${BUILD_TARGET:-cpu}"
[[ -z "${SERVER_IMAGE:-}" ]] || export SERVER_IMAGE
[[ -z "${CLIENT_IMAGE:-}" ]] || export CLIENT_IMAGE
compose_args=(--env-file server/.env)
[[ ! -f server/.env.secrets ]] || compose_args+=(--env-file server/.env.secrets)
[[ ! -f client/.env ]] || compose_args+=(--env-file client/.env)
compose_args+=(-f docker-compose.yml)
[[ "$BUILD_TARGET" != gpu ]] || compose_args+=(-f docker-compose.gpu.yml)
compose() { docker compose "${compose_args[@]}" "$@"; }
# Parse only the provider, never print resolved configuration or secrets.
ml_provider=$(compose config --format json | python3 -c 'import json,sys; print(json.load(sys.stdin)["services"]["server"]["environment"]["ML_PROVIDER"])')
case "$ml_provider" in
  tei) compose_args+=(--profile tei) ;;
  deepinfra) ;;
  *) echo "Unsupported ML_PROVIDER: $ml_provider" >&2; exit 1 ;;
esac
