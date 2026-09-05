import json
from pathlib import Path

from profiling_lab.schema import load_schema


def test_metric_schema_is_valid_and_unique() -> None:
    schema = load_schema(Path("config/metrics.json"))
    names = [metric["name"] for metric in schema["metrics"]]
    assert len(names) == len(set(names))
    assert len(names) >= 80


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
        "container",
        "network",
        "data_movement",
        "storage",
        "checkpoint",
    } <= categories


def test_metric_schema_covers_phase_and_data_path_signals() -> None:
    schema = load_schema(Path("config/metrics.json"))
    names = {metric["name"] for metric in schema["metrics"]}
    assert {
        "training_dataset_load_time_seconds",
        "training_model_load_time_seconds",
        "gpu_memory_peak_bytes",
        "host_pinned_memory_used_bytes",
        "network_communication_bytes_total",
        "data_movement_bytes_total",
        "data_movement_effective_bandwidth_bytes_per_second",
        "storage_read_bytes_total",
        "storage_write_bytes_total",
        "checkpoint_read_throughput_bytes_per_second",
        "checkpoint_write_throughput_bytes_per_second",
        "checkpoint_frequency_steps",
    } <= names
    assert {
        "dataset_loading",
        "model_loading",
        "training_input",
        "forward_backward",
        "optimizer_step",
        "checkpoint_save",
        "checkpoint_restore",
        "evaluation",
        "rollout",
        "tool_interaction",
        "reward",
        "weight_sync",
    } <= set(schema["phase_vocabulary"])


def test_json_schema_documents_the_runtime_contract() -> None:
    contract = json.loads(Path("config/metrics.schema.json").read_text(encoding="utf-8"))
    assert contract["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert contract["properties"]["schema_version"]["const"] == 1
    assert set(contract["properties"]["metrics"]["items"]["required"]) == {
        "name",
        "category",
        "unit",
        "scope",
        "source",
        "policy",
    }
