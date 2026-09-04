from pathlib import Path

from profiling_lab.schema import load_schema


def test_metric_schema_is_valid_and_unique() -> None:
    schema = load_schema(Path("config/metrics.json"))
    names = [metric["name"] for metric in schema["metrics"]]
    assert len(names) == len(set(names))
    assert len(names) >= 30


def test_metric_schema_covers_required_categories() -> None:
    schema = load_schema(Path("config/metrics.json"))
    categories = {metric["category"] for metric in schema["metrics"]}
    assert {
        "training",
        "rollout",
        "agent",
        "orchestration",
        "gpu",
        "host",
        "network",
        "storage",
        "checkpoint",
    } <= categories
