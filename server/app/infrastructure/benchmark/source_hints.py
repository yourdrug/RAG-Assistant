"""Shared interpretation of benchmark source hints.

Semicolon-separated hints identify relevant sources. Hit Rate and MRR use
the first retrieved source matching any hint; blank hints provide no labels.
"""

from domain.services.benchmark_source_hints import (
    matches_source_hints as matches_source_hints,
    parse_source_hints as parse_source_hints,
)
