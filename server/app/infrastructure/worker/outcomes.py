"""Expected resumable job outcomes, distinct from unexpected worker exceptions."""

from sqlalchemy.exc import DBAPIError


SQL_PREVIEW_CHARS = 200
DIAG_FIELDS = (
    ("detail", "message_detail"),
    ("hint", "message_hint"),
    ("table", "table_name"),
    ("column", "column_name"),
    ("constraint", "constraint_name"),
)


def db_error_message(exc: DBAPIError) -> str:
    """Summarise a driver error: type, server message, diagnostics and SQL head, not the full statement."""
    orig = exc.orig

    if orig is None:
        return type(exc).__name__

    diag = getattr(orig, "diag", None)
    text = str(orig).strip()
    primary = getattr(diag, "message_primary", None) or (text.splitlines()[0] if text else "")
    lines = [f"{type(orig).__name__}: {primary}".rstrip(": ")]

    for label, attr in DIAG_FIELDS:
        value = getattr(diag, attr, None)

        if value:
            lines.append(f"{label}: {value}")

    if exc.statement:
        sql = " ".join(str.__str__(exc.statement).split())

        if len(sql) > SQL_PREVIEW_CHARS:
            sql = sql[:SQL_PREVIEW_CHARS] + "…"

        lines.append(f"SQL: {sql}")

    return "\n".join(lines)


def job_error_message(exc: Exception) -> str:
    """Keep failures visible even when an exception has no message."""
    if isinstance(exc, DBAPIError):
        return db_error_message(exc)

    return str(exc).strip() or type(exc).__name__
