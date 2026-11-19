#!/usr/bin/env bash
# Local copy of .github/workflows/lint.yml: lint, test (PostgreSQL + Redis in Docker),
# offline-gate and client-test. Runs against a clean checkout of the given commit in a
# temporary worktree, so uncommitted changes neither leak into nor hide CI failures.
#
#   scripts/ci/local.sh [ref]            # default ref: HEAD
#   CI_JOBS="lint test" scripts/ci/local.sh
#   SKIP_LOCAL_CI=1 git push             # skip the pre-push run
#
# Keep the steps in sync with lint.yml when the workflow changes.
set -uo pipefail

if [ "${SKIP_LOCAL_CI:-}" = "1" ]; then
    echo "local CI: skipped (SKIP_LOCAL_CI=1)"
    exit 0
fi

ref="${1:-HEAD}"
jobs="${CI_JOBS:-lint test offline-gate client-test}"
repo="$(git rev-parse --show-toplevel)"
commit="$(git -C "$repo" rev-parse --verify "$ref^{commit}")" || exit 1
work="$(mktemp -d "${TMPDIR:-/tmp}/rag-local-ci.XXXXXX")"
tree="$work/tree"
suffix="$$"
pg="rag-local-ci-pg-$suffix"
redis="rag-local-ci-redis-$suffix"

cleanup() {
    docker rm -f "$pg" "$redis" >/dev/null 2>&1 || true
    git -C "$repo" worktree remove --force "$tree" >/dev/null 2>&1 || true
    rm -rf "$work"
}
trap cleanup EXIT INT TERM

git -C "$repo" worktree add --detach --quiet "$tree" "$commit" || exit 1
echo "local CI: $(git -C "$repo" log -1 --format='%h %s' "$commit")"

failed=()
run_job() {
    local name="$1"
    shift
    echo
    echo "======== $name"
    if ( "$@" ); then
        echo "======== $name: OK"
    else
        echo "======== $name: FAILED"
        failed+=("$name")
    fi
}

sync_server() {
    (cd "$tree/server" && uv sync --frozen --quiet --group development --group testing)
}

job_lint() {
    sync_server || return 1
    cd "$tree/server" || return 1
    local status=0
    uv run python tools/blank_lines.py --check app tests tools || status=1
    uv run ruff check app tests tools typings || status=1
    uv run ruff format --check app tests tools typings || status=1
    uv run mypy || status=1
    PYTHONPATH=app uv run lint-imports || status=1
    uv run deptry . || status=1
    uv run vulture || status=1
    return "$status"
}

wait_healthy() {
    local container="$1" check="$2" i
    for i in $(seq 1 60); do
        # shellcheck disable=SC2086
        docker exec "$container" $check >/dev/null 2>&1 && return 0
        sleep 1
    done
    echo "$container did not become ready" >&2
    return 1
}

job_test() {
    command -v docker >/dev/null || { echo "docker is required for the test job" >&2; return 1; }
    sync_server || return 1
    docker run -d --rm --name "$pg" -p 127.0.0.1::5432 \
        -e POSTGRES_USER=test -e POSTGRES_PASSWORD=test -e POSTGRES_DB=rag_test \
        postgres:16.4-alpine >/dev/null || return 1
    docker run -d --rm --name "$redis" -p 127.0.0.1::6379 redis:7-alpine >/dev/null || return 1
    wait_healthy "$pg" "pg_isready -U test -d rag_test" || return 1
    wait_healthy "$redis" "redis-cli ping" || return 1
    local pg_port redis_port
    pg_port="$(docker port "$pg" 5432/tcp | head -1 | sed 's/.*://')"
    redis_port="$(docker port "$redis" 6379/tcp | head -1 | sed 's/.*://')"

    cd "$tree/server" || return 1
    export DATA_DIR="$work/rag_test"
    export TEST_POSTGRES_URL="postgresql+asyncpg://test:test@localhost:$pg_port/rag_test"
    export TEST_WORKER_REDIS_URL="redis://localhost:$redis_port/15"
    export RUN_INTEGRATION_TESTS=1
    # CI reaches this test through the default redis://localhost:6379/15; here the port is random.
    export RATE_LIMIT_TEST_REDIS_URL="redis://localhost:$redis_port/15"

    echo "-- migration baseline"
    uv run pytest tests/integration/test_migration_schema.py -q || return 1
    echo "-- worker reliability"
    TEST_REDIS_URL="redis://localhost:$redis_port/14" TEST_REDIS_CONTAINER="$redis" \
        uv run pytest -q \
        tests/integration/test_postgres_transactions.py \
        tests/integration/test_worker_queues.py \
        tests/integration/test_configuration_corpus_postgres.py || return 1
    echo "-- tests"
    uv run pytest -q \
        --ignore=tests/integration/test_migration_schema.py \
        --ignore=tests/integration/test_worker_queues.py \
        --ignore=tests/integration/test_postgres_transactions.py \
        --ignore=tests/integration/test_configuration_corpus_postgres.py \
        --cov=domain.knowledge.access_control \
        --cov=infrastructure.repositories.vector.acl \
        --cov=infrastructure.ml.cache.keys \
        --cov-report=term-missing:skip-covered --cov-fail-under=85
}

job_offline_gate() {
    local cache="${XDG_CACHE_HOME:-$HOME/.cache}/rag-local-ci"
    local tokenizer="$cache/bge-m3-tokenizer.json"
    local sha="21106b6d7dab2952c1d496fb21d5dc9db75c28ed361a05f5020bbba27810dd08"
    mkdir -p "$cache"
    if ! echo "$sha  $tokenizer" | sha256sum -c --quiet - 2>/dev/null; then
        curl -fsSL -o "$tokenizer" \
            https://huggingface.co/BAAI/bge-m3/resolve/5617a9f61b028005a4858fdac845db406aefb181/tokenizer.json \
            || return 1
        echo "$sha  $tokenizer" | sha256sum -c --quiet - || return 1
    fi
    sync_server || return 1
    cd "$tree/server" || return 1
    env PYTHONPATH=app DATA_DIR="$work/rag_offline_gate" \
        DB_HOST=localhost DB_PORT=5432 DB_USER=offline DB_PASSWORD=offline DB_NAME=offline \
        ALLOWED_ORIGINS='*' JWT_SECRET_KEY=offline-gate-placeholder-secret-0000 \
        REDIS_PASSWORD=offline-gate-redis-password \
        S3_ACCESS_KEY=offline-gate-access S3_SECRET_KEY=offline-gate-secret-value \
        bash -c '
            status=0
            for pack in support internal belarus_law; do
                echo "== $pack"
                thresholds=()
                spec="app/infrastructure/domain_packs/$pack/benchmarks/acceptance.json"
                [ -f "$spec" ] && thresholds=(--thresholds "server/$spec")
                uv run python ../scripts/benchmark/chunking_report.py --pack "$pack" \
                    --tokenizer "$0" "${thresholds[@]}" \
                    --baseline "docs/quality/artifacts/offline-gate/$pack.json" --gate 0.02 || status=1
            done
            exit "$status"
        ' "$tokenizer"
}

job_client_test() {
    cd "$tree/client" || return 1
    npm ci --prefer-offline --no-audit --no-fund --loglevel=error || return 1
    npm test || return 1
    npm run check
}

for job in $jobs; do
    case "$job" in
        lint) run_job lint job_lint ;;
        test) run_job test job_test ;;
        offline-gate) run_job offline-gate job_offline_gate ;;
        client-test) run_job client-test job_client_test ;;
        *) echo "unknown job: $job" >&2; failed+=("$job") ;;
    esac
done

echo
if [ "${#failed[@]}" -gt 0 ]; then
    echo "local CI FAILED: ${failed[*]}  (skip once: SKIP_LOCAL_CI=1 git push)"
    exit 1
fi
echo "local CI passed: $jobs"
