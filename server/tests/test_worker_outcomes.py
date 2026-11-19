from types import SimpleNamespace

from sqlalchemy.exc import DBAPIError
from sqlalchemy.sql.util import _long_statement

from infrastructure.worker.outcomes import SQL_PREVIEW_CHARS, job_error_message


class StringDataRightTruncation(Exception):  # noqa: N818 - mirrors the psycopg name
    pass


def test_db_error_message_is_short_and_keeps_the_cause() -> None:
    statement = "INSERT INTO chunks (document_id, content) VALUES " + ", ".join(  # noqa: S608
        f"(p{i}, q{i})" for i in range(500)
    )
    orig = StringDataRightTruncation("value too long for type character varying(500)\n")
    orig.diag = SimpleNamespace(  # type: ignore[attr-defined]
        message_primary="value too long for type character varying(500)",
        message_detail=None,
        message_hint=None,
        table_name="chunks",
        column_name=None,
        constraint_name=None,
    )
    exc = DBAPIError(_long_statement(statement), {"p0": "secret chunk text"}, orig)

    message = job_error_message(exc)

    first_line = "StringDataRightTruncation: value too long for type character varying(500)"
    assert message.splitlines()[0] == first_line
    assert "table: chunks" in message
    assert "SQL: INSERT INTO chunks (document_id, content) VALUES" in message
    assert "secret chunk text" not in message
    assert len(message) < SQL_PREVIEW_CHARS + 300


def test_db_error_message_without_diag() -> None:
    exc = DBAPIError("SELECT 1", None, RuntimeError("connection reset"))

    assert job_error_message(exc) == "RuntimeError: connection reset\nSQL: SELECT 1"


def test_job_error_message_falls_back_to_type_name() -> None:
    assert job_error_message(RuntimeError()) == "RuntimeError"
