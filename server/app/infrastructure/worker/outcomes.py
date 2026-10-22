"""Expected resumable job outcomes, distinct from unexpected worker exceptions."""


def job_error_message(exc: Exception) -> str:
    """Keep failures visible even when an exception has no message."""
    return str(exc).strip() or type(exc).__name__
