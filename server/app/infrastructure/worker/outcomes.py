"""Expected resumable job outcomes, distinct from unexpected worker exceptions."""


class IncompleteEvaluation(ValueError):
    """All work finished, but some judge scores remain unavailable."""


def job_error_message(exc: Exception) -> str:
    """Keep failures visible even when an exception has no message."""
    return str(exc).strip() or type(exc).__name__
