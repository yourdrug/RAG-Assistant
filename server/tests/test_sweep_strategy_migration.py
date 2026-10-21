"""The API exposes only supported benchmark strategies."""

from presentation.api.schemas.benchmark import SweepCreateRequest


def test_openapi_strategy_values_exclude_legacy_alias() -> None:
    schema = SweepCreateRequest.model_json_schema()
    assert schema["$defs"]["BenchmarkStrategy"]["enum"] == ["grid", "random", "optuna_tpe"]
