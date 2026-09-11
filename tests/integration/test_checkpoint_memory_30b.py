import json
from pathlib import Path

from experiments import checkpoint_memory_30b


def test_aggregate_probe_sums_ranks_and_uses_slowest_time() -> None:
    def rank(seconds, logical, physical):
        read = {"seconds": seconds, "logical_bytes": logical, "physical_read_bytes": physical, "classification": "valid"}
        return {
            "checkpoint_io": {"inventory": {"logical_bytes": logical, "allocated_bytes": logical},
                              "reads": {name: dict(read) for name in ("warm_after_write", "cold_buffered", "warm_buffered", "direct")}},
            "resources": {"mem_available_min_bytes": 100, "host_memory_pressure_bytes": 20, "device_write_bytes_delta": 30},
        }

    result = checkpoint_memory_30b.aggregate_probe({"0": rank(2, 100, 90), "1": rank(4, 200, 180)})

    assert result["logical_checkpoint_bytes"] == 300
    assert result["stage_device_write_bytes"] == 60
    assert result["reads"]["cold_buffered"]["logical_bytes_per_second"] == 75


def test_resource_summary_reports_peak_pressure() -> None:
    lines = [
        json.dumps({"monotonic_seconds": 1, "device_major_minor": "1:2", "memavailable_bytes": 100, "memtotal_bytes": 200,
                    "swaptotal_bytes": 20, "swapfree_bytes": 20, "read_bytes": 1, "write_bytes": 2,
                    "read_operations": 1, "write_operations": 2, "read_time_ms": 3, "write_time_ms": 4,
                    "in_flight": 0, "busy_time_ms": 3}),
        json.dumps({"monotonic_seconds": 3, "device_major_minor": "1:2", "memavailable_bytes": 60, "memtotal_bytes": 200,
                    "swaptotal_bytes": 20, "swapfree_bytes": 10, "read_bytes": 11, "write_bytes": 22,
                    "read_operations": 3, "write_operations": 6, "read_time_ms": 8, "write_time_ms": 10,
                    "in_flight": 2, "busy_time_ms": 13}),
    ]

    result = checkpoint_memory_30b.summarize_resources(lines)

    assert result["host_memory_pressure_bytes"] == 40
    assert result["device_write_bytes_delta"] == 20
    assert result["device_write_bytes_per_second"] == 10
    assert result["device_in_flight_max"] == 2
    assert result["swap_used_max_bytes"] == 10


def test_relative_mad_drives_extension_rule() -> None:
    assert checkpoint_memory_30b.relative_mad([10, 10, 11, 9, 10]) == 0
    assert checkpoint_memory_30b.relative_mad([5, 7, 10, 13, 20]) == 0.3


def test_memory_matrix_matches_fixed_length_and_backend_contract() -> None:
    conditions = {item["name"]: item for item in checkpoint_memory_30b.memory_experiments()}

    assert set(conditions) == {"len-1024", "len-4096", "trl-ddp", "trl-fsdp2", "trl-zero3-nvme"}
    assert conditions["len-4096"]["experiment"]["env"]["PAD_TO_MAX_LENGTH"] is True
    assert conditions["trl-ddp"]["experiment"]["env"]["PAD_TO_MAX_LENGTH"] is True
    zero = conditions["trl-zero3-nvme"]["experiment"]["env"]
    assert zero["FINETUNING_MODE"] == "full"
    assert zero["OPTIMIZER"] == "sgd"
    assert zero["DEEPSPEED_CONFIG"].endswith("deepspeed-zero3-nvme.json")


def test_memory_estimates_include_non_runnable_full_adam() -> None:
    setup = {"nodes": [{
        "host": "spark@spark1", "checkout": "/repo",
        "python": {"megatron": "/venv/bin/python"},
        "model_dirs": {checkpoint_memory_30b.MODEL_ID: "/models/qwen"},
    }]}

    records = checkpoint_memory_30b.collect_memory_estimates(setup, execute=False)

    assert {item["name"] for item in records} >= {"meg-full-sgd", "meg-full-adam"}
    assert all("--json" in item["command"] for item in records)


def test_prepare_cohorts_parses_pretty_printed_cached_manifest() -> None:
    # The cache-hit branch `cat`s manifest.json, which is written with indent=2
    # (multi-line); a same-selection, same-content result from both nodes must
    # not raise even though it is not a single compact JSON line.
    manifest = {
        "dataset": checkpoint_memory_30b.DATASET_ID,
        "dataset_revision": checkpoint_memory_30b.DATASET_REVISION,
        "model_revision": checkpoint_memory_30b.MODEL_REVISION,
        "max_length": 2048, "train_count": 32, "eval_count": 8,
        "source_selection_sha256": "abc",
        "files": {"training": {"sha256": "t"}, "validation": {"sha256": "v"}},
    }

    class Result:
        returncode = 0
        stdout = json.dumps(manifest, indent=2) + "\n"

    node = {
        "host": "spark@spark1", "checkout": "/repo", "python": {"megatron": "/venv/bin/python"},
        "model_dirs": {checkpoint_memory_30b.MODEL_ID: "/models/qwen"},
        "data_dir": "/data/ultrachat", "output_root": {"megatron": "/mnt/megatron"},
    }
    setup = {"nodes": [node, {**node, "host": "spark@spark2"}]}

    updated, planned = checkpoint_memory_30b.prepare_cohorts(setup, execute=True, remote_run=lambda *a, **k: Result())

    assert len(planned) == 2
    assert updated["nodes"][0]["data_dir"] == checkpoint_memory_30b.cohort_path(node)


def test_cleanup_checkpoints_removes_each_ranks_checkpoint_dir() -> None:
    seen = []

    class Result:
        returncode = 0
        stderr = ""

    def fake_run(command, **kwargs):
        seen.append(command)
        return Result()

    plan = {"ranks": [
        {"rank": 0, "host": "spark@spark1", "env": {"CHECKPOINT_DIR": "/mnt/a/checkpoints"}},
        {"rank": 1, "host": "spark@spark2", "env": {"CHECKPOINT_DIR": "/mnt/b/checkpoints"}},
    ]}

    checkpoint_memory_30b.cleanup_checkpoints(plan, Path("/unused"), {}, remote_run=fake_run)

    assert len(seen) == 2
    assert "rm -rf -- /mnt/a/checkpoints" in seen[0][-1]
    assert "rm -rf -- /mnt/b/checkpoints" in seen[1][-1]
