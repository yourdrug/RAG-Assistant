"""Composition root — DI container for the RAG application."""

# Must precede every import that loads ``config``: it installs the startup hook
# that reports ``ConfigurationError`` raised while the settings are built.
import composition.startup_errors


# isort: split
from composition.container import Container


__all__ = ["Container"]
