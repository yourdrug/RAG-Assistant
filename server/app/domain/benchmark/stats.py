"""Statistics for benchmark metrics."""


def percentile(sorted_data: list[float], p: float) -> float:
    """Compute percentile from pre-sorted data using linear interpolation."""
    if not sorted_data:
        return 0.0

    if len(sorted_data) == 1:
        return sorted_data[0]

    k = (len(sorted_data) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_data) - 1)
    return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)
