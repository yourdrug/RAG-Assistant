#!/usr/bin/env bash
# Pre-commit: warn when benchmark questions in the running stack differ from the committed
# snapshots (questions.json of each pack). Never blocks a commit; skipped when the stack is down.
set -uo pipefail

project=$(printf '%s' "${COMPOSE_PROJECT_NAME:-$(basename "$PWD")}" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9_-')
container=$(docker ps -q \
  --filter "label=com.docker.compose.project=$project" \
  --filter "label=com.docker.compose.service=server" 2>/dev/null | head -n 1)

[[ -n "$container" ]] || exit 0

output=$(timeout 60 docker exec --user "$(id -u):$(id -g)" "$container" \
  bash -c 'cd app && python -m composition.cli seed check' 2>&1)
status=$?

if [[ $status -ne 0 ]]; then
  echo "WARNING: benchmark questions in the database differ from the committed seed:"
  # Drop timestamped log lines, keep the command's own report.
  grep -v '^\[20' <<<"$output" | sed 's/^/  /'
fi

exit 0
