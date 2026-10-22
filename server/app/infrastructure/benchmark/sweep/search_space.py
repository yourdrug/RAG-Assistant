"""Search space."""

from __future__ import annotations
import itertools
from decimal import Decimal
from domain.benchmark.evaluation import (
    has_retrieval_labels as has_retrieval_labels,
)


def generate_grid_points(search_space: dict) -> list[dict]:
    """Generate all parameter combinations (cartesian product)."""
    param_lists = {}

    for param, spec in search_space.items():
        if param.startswith("_"):
            continue

        if "values" in spec:
            param_lists[param] = spec["values"]
        elif "min" in spec and "max" in spec:
            param_lists[param] = grid_range_values(spec)
        else:
            param_lists[param] = [spec.get("default", 0)]

    if not param_lists:
        return [{}]

    keys = list(param_lists.keys())
    value_combos = list(itertools.product(*[param_lists[k] for k in keys]))
    return [dict(zip(keys, combo, strict=False)) for combo in value_combos]


def grid_range_values(spec: dict) -> list[int | float]:
    """Enumerate inclusive ranges without truncating floating rerank thresholds."""
    start, stop, step = (Decimal(str(value)) for value in (spec["min"], spec["max"], spec.get("step", 1)))

    if not all(value.is_finite() for value in (start, stop, step)) or step <= 0 or stop < start:
        raise ValueError("Grid ranges require finite bounds, min <= max and a positive step")

    is_float = any(isinstance(spec.get(key), float) for key in ("min", "max", "step"))
    count = int((stop - start) // step) + 1
    convert = float if is_float else int
    return [convert(start + step * index) for index in range(count)]


def generate_random_points(search_space: dict, n: int) -> list[dict]:
    """Generate N random parameter combinations."""
    import random as _random

    points = []

    for _ in range(n):
        point = {}

        for param, spec in search_space.items():
            if param.startswith("_"):
                continue

            if "values" in spec:
                point[param] = _random.choice(spec["values"])  # noqa: S311
            elif "min" in spec and "max" in spec:
                step = spec.get("step")
                is_float = (
                    isinstance(spec.get("min"), float)
                    or isinstance(spec.get("max"), float)
                    or (step and isinstance(step, float))
                )

                if is_float:
                    point[param] = _random.uniform(spec["min"], spec["max"])  # noqa: S311
                else:
                    point[param] = _random.randint(int(spec["min"]), int(spec["max"]))  # noqa: S311
            else:
                point[param] = spec.get("default", 0)

        points.append(point)

    return points
