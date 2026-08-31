#!/usr/bin/env bash
# deploy.sh — pull & run a specific version from GHCR on a remote server.
# Usage: ./deploy.sh 0.6.0
#        ./deploy.sh 0.6.0 --gpu
set -euo pipefail

VERSION="${1:?Usage: ./deploy.sh <version> [--gpu]}"
VERSION="${VERSION#v}"
shift
GPU=false
for arg in "$@"; do
  [[ "$arg" == "--gpu" ]] && GPU=true
done

DEPLOY_DIR="${DEPLOY_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$DEPLOY_DIR"

export VERSION
export STAGE=production
export BUILD_TARGET=$( [[ "$GPU" == true ]] && echo "gpu" || echo "cpu" )

COMPOSE_FILES="-f docker-compose.yml -f docker-compose.prod.yml"
[[ "$GPU" == true ]] && COMPOSE_FILES="$COMPOSE_FILES -f docker-compose.gpu.yml"

echo "==> Deploying ${VERSION} (target: ${BUILD_TARGET})"

# Pull from GHCR
docker compose $COMPOSE_FILES pull server worker client

# Stop old containers (avoids GPU reservation conflicts when switching modes)
docker compose $COMPOSE_FILES down --remove-orphans 2>/dev/null || true

# Start (migrations are handled by server entrypoint.sh)
docker compose $COMPOSE_FILES up -d --remove-orphans --wait

# Cleanup
docker image prune -f

echo "==> Deploy ${VERSION} complete"
