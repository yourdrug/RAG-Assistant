"""Dynamic declarations use enum categories without changing storage/API strings."""

import json
from typing import Any

import pytest

from domain.runtime.dynamic_params import DYNAMIC_PARAMS, DynamicParam, DynamicParamCategory
from infrastructure.initialization import build_defaults
from presentation.api.schemas.admin_config import ConfigParamResponse


@pytest.mark.parametrize("value", [*DynamicParamCategory, *(item.value for item in DynamicParamCategory)])
def test_category_normalizes_enum_and_string(value: Any) -> None:
    param = DynamicParam("example", str, value, "example")
    assert param.category is DynamicParamCategory(value)
    assert json.loads(json.dumps({"category": param.category}))["category"] == str(value)


@pytest.mark.parametrize("value", ["unknown", "", None])
def test_category_rejects_unknown_values(value: Any) -> None:
    with pytest.raises(ValueError, match="is not a valid DynamicParamCategory"):
        DynamicParam("example", str, value, "example")


def test_registry_and_seed_preserve_category_strings() -> None:
    expected = {
        "corpus",
        "cost",
        "hybrid",
        "ingestion",
        "jobs",
        "llm",
        "ml",
        "ocr",
        "openrouter",
        "rag",
        "rate_limit",
        "reranker",
        "toggles",
    }
    assert {category.value for category in DynamicParamCategory} == expected
    assert {param.category for param in DYNAMIC_PARAMS} == expected
    rows = {(row.key, row.domain_key): row for row in build_defaults()}

    for param in DYNAMIC_PARAMS:
        assert isinstance(param.category, DynamicParamCategory)
        row = rows[(param.key, param.domain_key)]
        assert row.category == param.category.value
        response = ConfigParamResponse(
            key=row.key, value=row.value, value_type=row.value_type, category=row.category
        )
        assert response.model_dump(mode="json")["category"] == param.category.value
