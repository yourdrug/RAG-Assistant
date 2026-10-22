#!/usr/bin/env python3
"""Periodic readiness, free-space and backup-age checks; optional webhook notification."""

import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit
from pathlib import Path

root = Path(__file__).resolve().parents[2]
errors = []
url = os.environ.get("OPERATIONS_READY_URL", "http://127.0.0.1:3001/api/ready")
try:
    if urlsplit(url).scheme not in {"http", "https"}:
        raise ValueError("Readiness URL must use HTTP or HTTPS")
    with urllib.request.urlopen(url, timeout=20) as response:  # noqa: S310 -- scheme validated above
        if json.load(response).get("status") != "healthy":
            errors.append("API reports degraded readiness")
except (urllib.error.URLError, TimeoutError, ValueError) as exc:
    errors.append(f"Readiness failed: {exc}")
data_dir = root / "data"
if not data_dir.is_dir():
    errors.append("Data directory is missing")
usage = shutil.disk_usage(data_dir if data_dir.is_dir() else root)
if usage.free / usage.total < float(os.environ.get("OPERATIONS_MIN_FREE_RATIO", "0.15")):
    errors.append("Data filesystem has less than configured free space")
backup_time = root / ".deploy/last-backup-success"
try:
    age = time.time() - int(backup_time.read_text().strip())
    if age > int(os.environ.get("OPERATIONS_BACKUP_MAX_AGE_SECONDS", "172800")):
        errors.append("Last verified backup is too old")
except (OSError, ValueError):
    errors.append("No successful verified backup recorded")
if errors:
    message = "; ".join(errors)
    print(message, file=sys.stderr)
    webhook = os.environ.get("OPERATIONS_ALERT_WEBHOOK", "")
    if webhook and urlsplit(webhook).scheme not in {"http", "https"}:
        print("Alert webhook must use HTTP or HTTPS", file=sys.stderr)
        sys.exit(1)
    if webhook:
        request = urllib.request.Request(  # noqa: S310 -- scheme validated above
            webhook, data=json.dumps({"text": message}).encode(), headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310 -- scheme validated above
                response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"Alert delivery failed: {exc}", file=sys.stderr)
    sys.exit(1)
print("Readiness, disk space and backup age OK")
