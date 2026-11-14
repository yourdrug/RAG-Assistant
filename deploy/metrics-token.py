"""Prepare a stable local scrape credential without printing it."""

import os
from pathlib import Path
import secrets


directory = Path(".deploy")
directory.mkdir(mode=0o700, exist_ok=True)
directory.chmod(0o700)
token_file = directory / "metrics.token"
if not token_file.exists():
    descriptor = os.open(token_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(secrets.token_urlsafe(48))
token = token_file.read_text().strip()
if len(token) < 32 or not all(char.isalnum() or char in "_-" for char in token):
    raise SystemExit("Invalid local metrics credential; replace .deploy/metrics.token")
# The enclosing directory is private; the bound file must be readable by the
# unprivileged Prometheus user inside its container.
token_file.chmod(0o644)
environment = directory / "metrics.env"
descriptor = os.open(environment, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(descriptor, "w") as stream:
    stream.write(f"METRICS_SCRAPE_TOKEN={token}\n")
environment.chmod(0o600)
