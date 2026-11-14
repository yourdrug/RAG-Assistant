#!/usr/bin/env bash
# Sourced by deploy.sh; compose() and the operation lock belong to the caller.

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
