"""Document versioning -- pure domain logic for conflict resolution strategy.

Decides whether a document conflict should be resolved as a new version
or as a replacement.  No I/O, no infrastructure dependencies.
"""

from __future__ import annotations

from typing import Literal


def decide_resolution_strategy(
    is_versioned: bool,
) -> Literal["version", "replace"]:
    """Decide how to resolve a conflict between old and new document versions.

    Returns ``"version"`` for versioned domains (legal, decree) -- the old
    document stays as historical, a new ActVersion is created.
    Returns ``"replace"`` for legacy/general domains -- the old document
    is deleted, the new one takes its canonical filename slot.
    """
    return "version" if is_versioned else "replace"
