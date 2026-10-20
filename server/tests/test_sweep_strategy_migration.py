"""The former strategy alias is normalized without changing sweep state."""

from importlib import import_module

from alembic.migration import MigrationContext
from alembic.operations import Operations
import pytest
from sqlalchemy import create_engine, text

from domain.benchmark.sweep import BenchmarkStrategy
from presentation.api.schemas.benchmark import SweepCreateRequest


@pytest.mark.parametrize("legacy_count", [0, 1, 3])
def test_strategy_migration_preserves_rows_and_supports_replay(legacy_count: int) -> None:
    migration = import_module(
        "infrastructure.database.migrations.versions.u1k2l3m4n5o6_normalize_sweep_strategy"
    )
    strategies = [strategy.value for strategy in BenchmarkStrategy] + ["successive_halving"] * legacy_count
    engine = create_engine("sqlite://")
    try:
        with engine.begin() as connection:
            connection.execute(text("CREATE TABLE benchmark_sweeps (id INTEGER, strategy TEXT, status TEXT)"))
            connection.execute(
                text("INSERT INTO benchmark_sweeps VALUES (:id, :strategy, :status)"),
                [{"id": i, "strategy": strategy, "status": "done"} for i, strategy in enumerate(strategies)],
            )
            expected = [
                (i, "optuna_tpe" if s == "successive_halving" else s, "done")
                for i, s in enumerate(strategies)
            ]
            with Operations.context(MigrationContext.configure(connection)):
                for operation in (migration.upgrade, migration.upgrade, migration.downgrade):
                    operation()
                    rows = connection.execute(text("SELECT * FROM benchmark_sweeps ORDER BY id"))
                    assert [tuple(row) for row in rows] == expected
    finally:
        engine.dispose()


def test_openapi_strategy_values_exclude_legacy_alias() -> None:
    schema = SweepCreateRequest.model_json_schema()
    assert schema["$defs"]["BenchmarkStrategy"]["enum"] == ["grid", "random", "optuna_tpe"]
