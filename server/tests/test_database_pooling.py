"""Session features must bypass transaction pooling and keep database timeouts."""

from pydantic import SecretStr

from config import DatabaseSection
from infrastructure.database.database import configured_database_url, connection_arguments


def test_direct_database_url_is_distinct_from_transaction_endpoint() -> None:
    settings = DatabaseSection(
        db_host="pgbouncer",
        db_port="5432",
        db_user="user",
        db_password=SecretStr("disposable-test-value"),
        db_name="rag",
        db_direct_host="postgres",
        db_direct_port="5432",
        db_use_pgbouncer=True,
    )
    assert configured_database_url(settings, direct=True).url.host == "postgres"
    assert configured_database_url(settings).url.host == "pgbouncer"
    assert "options" not in connection_arguments(settings, pooled=True)
    assert "statement_timeout" in str(connection_arguments(settings)["options"])


def test_transaction_pooling_requires_a_direct_endpoint() -> None:
    settings = DatabaseSection(
        db_host="pgbouncer",
        db_port="5432",
        db_user="user",
        db_password=SecretStr("disposable-test-value"),
        db_name="rag",
        db_use_pgbouncer=True,
    )
    assert settings.check_database()
