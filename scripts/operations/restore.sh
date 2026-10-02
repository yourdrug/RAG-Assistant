#!/usr/bin/env bash
# Extract a verified cold backup into a NEW directory, never overwrite live data.
set -euo pipefail
umask 077
backup_dir=$(cd "${1:?Usage: restore.sh BACKUP_DIRECTORY NEW_DATA_DIRECTORY}" && pwd)
target="${2:?Specify a NEW data directory}"
[[ ! -e "$target" ]] || { echo "Target exists; refusing to overwrite" >&2; exit 1; }
(cd "$backup_dir" && sha256sum -c SHA256SUMS)
mkdir -m 0750 "$target"
tar --same-owner -xzf "$backup_dir/data.tar.gz" -C "$target"
echo "Data extracted to $target. Follow docs/operations/deployment.md before switching production."
