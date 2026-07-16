#!/bin/bash

set -e

GREEN="\033[0;32m"
NC="\033[0m"

log_info() {
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    printf "${GREEN}[${timestamp}] INFO: %s${NC}\n" "$1"
}

main() {
    export SERVICE_START_DATETIME
    SERVICE_START_DATETIME=$(date --utc '+%Y-%m-%d %H:%M:%S%z')

    log_info "rag-server starting (version: $(cat VERSION 2>/dev/null || echo 'dev'))"

    cd app
    log_info "Current working directory: '$(pwd)'"

    if [ $# -eq 0 ]; then
        log_info "No command given — starting uvicorn with defaults."
        exec uvicorn main:app --host 0.0.0.0 --port 8001
    fi

    log_info "Command: '$*'"
    exec "$@"
}

main "$@"
