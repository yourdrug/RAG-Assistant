"""Stable identities for benchmark recovery."""

import hashlib
import json


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()
