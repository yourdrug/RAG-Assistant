#!/usr/bin/env bash
# deploy.sh — pull & run a specific version from GHCR on a remote server.
# Usage: ./deploy.sh v0.6.0
#        ./deploy.sh v0.6.0 --gpu
#        DEPLOY_DIR=/opt/rag-app ./deploy.sh v0.6.0
set -euo pipefail

VERSION="${1:?Usage: ./deploy.sh <version> [--gpu]}"
shift
GPU=false
for arg in "$@"; do
  [[ "$arg" == "--gpu" ]] && GPU=true
done

DEPLOY_DIR="${DEPLOY_DIR:-/opt/rag-app}"
cd "$DEPLOY_DIR"

export VERSION
export STAGE=production
export BUILD_TARGET=$( [[ "$GPU" == true ]] && echo "gpu" || echo "cpu" )

echo "==> Deploying ${VERSION} (target: ${BUILD_TARGET})"

docker compose \
  -f docker-compose.yml \
  -f docker-compose.prod.yml \
  ${GPU:+-f docker-compose.gpu.yml} \
  pull server worker

docker compose \
  -f docker-compose.yml \
  -f docker-compose.prod.yml \
  ${GPU:+-f docker-compose.gpu.yml} \
  up -d --remove-orphans

docker compose exec -T server alembic upgrade head
docker image prune -f

echo "==> Deploy ${VERSION} complete"
