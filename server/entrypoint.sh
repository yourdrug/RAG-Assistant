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

    # Run Alembic migrations from project root (alembic.ini is here)
    log_info "Running Alembic migrations..."
    alembic upgrade head
    log_info "Alembic migrations completed."

    cd app
    log_info "Current working directory: '$(pwd)'"

    # Show current prompt
    export PS1='\[\033[01;32m\]rag\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '

    if [ $# -eq 0 ]; then
        log_info "No command given — use: python main.py runserver"
        exit 1
    fi

    log_info "Command: '$*'"
    exec "$@"
}

main "$@"
