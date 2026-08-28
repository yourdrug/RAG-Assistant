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

echo "==> Deploying ${VERSION} (target: ${BUILD_TARGET})"

# Pull from GHCR
docker compose \
  -f docker-compose.yml \
  -f docker-compose.prod.yml \
  ${GPU:+-f docker-compose.gpu.yml} \
  pull server worker

# Tag so compose doesn't try to rebuild (build: is in base compose)
docker tag ghcr.io/yourdrug/rag-assistant:${VERSION} rag-server:${STAGE}-${BUILD_TARGET}-${VERSION}
docker tag ghcr.io/yourdrug/rag-assistant:${VERSION} rag-server:${STAGE}-${BUILD_TARGET}-latest
docker tag ghcr.io/yourdrug/rag-assistant:${VERSION} rag-worker:${STAGE}-${BUILD_TARGET}-${VERSION}
docker tag ghcr.io/yourdrug/rag-assistant:${VERSION} rag-worker:${STAGE}-${BUILD_TARGET}-latest

# Start
docker compose \
  -f docker-compose.yml \
  -f docker-compose.prod.yml \
  ${GPU:+-f docker-compose.gpu.yml} \
  up -d --remove-orphans

# Migrations
docker compose exec -T server alembic upgrade head 2>/dev/null || true

# Cleanup
docker image prune -f

echo "==> Deploy ${VERSION} complete"
