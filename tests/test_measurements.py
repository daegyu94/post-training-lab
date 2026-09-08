import pytest

from profiling_lab.measurements import summarize_steps


def test_warmup_is_excluded_and_percentiles_use_nearest_rank():
    result = summarize_steps([100, 4, 1, 3, 2], 1)
    assert result == {"count": 4, "mean_seconds": 2.5, "p50_seconds": 2, "p95_seconds": 4}


@pytest.mark.parametrize("values,warmup", [([], 0), ([1], 1), ([0], 0), ([float('nan')], 0)])
def test_invalid_measurements(values, warmup):
    with pytest.raises(ValueError):
        summarize_steps(values, warmup)
