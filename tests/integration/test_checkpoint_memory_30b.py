import json
from pathlib import Path
from types import SimpleNamespace

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


def test_checkpoint_summary_derives_variant_names_from_records() -> None:
    def record(variant: str, size: int, save_seconds: float) -> dict:
        return {
            "status": "passed", "warmup": False, "variant": variant,
            "metrics": {"save_call_host_seconds_max_across_ranks": save_seconds},
            "post_run": {"aggregate": {
                "logical_checkpoint_bytes": size,
                "reads": {"cold_buffered": {"logical_bytes_per_second": 1e9}},
            }},
        }

    records = [
        record("ratio-0.1pct", 1_000_000, 1.0),
        record("ratio-0.5pct", 5_000_000, 1.2),
        record("ratio-1pct", 10_000_000, 1.5),
    ]

    summary = checkpoint_memory_30b.checkpoint_summary(records)

    assert set(summary) == {"ratio-0.1pct", "ratio-0.5pct", "ratio-1pct"}
    assert summary["ratio-1pct"]["logical_checkpoint_bytes_median"] == 10_000_000
    assert summary["ratio-1pct"]["run_count"] == 1


def test_memory_matrix_matches_fixed_length_and_backend_contract() -> None:
    conditions = {item["name"]: item for item in checkpoint_memory_30b.memory_experiments()}

    assert set(conditions) == {"len-4096", "len-8192", "trl-ddp", "trl-fsdp2", "trl-zero3-nvme"}
    assert conditions["len-4096"]["experiment"]["env"]["PAD_TO_MAX_LENGTH"] is True
    assert conditions["trl-ddp"]["experiment"]["env"]["PAD_TO_MAX_LENGTH"] is True
    zero = conditions["trl-zero3-nvme"]["experiment"]["env"]
    assert zero["FINETUNING_MODE"] == "full"
    # DeepSpeed substitutes DeepSpeedCPUAdam whenever optimizer state is offloaded,
    # so this condition must record adamw to match what actually runs.
    assert zero["OPTIMIZER"] == "adamw"
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


def test_prior_terminal_status_ignores_a_run_stuck_running(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"

    assert checkpoint_memory_30b._prior_terminal_status(True, manifest) is None  # no prior attempt

    manifest.write_text(json.dumps({"status": "running"}), encoding="utf-8")
    assert checkpoint_memory_30b._prior_terminal_status(True, manifest) is None  # interrupted mid-run

    manifest.write_text(json.dumps({"status": "passed"}), encoding="utf-8")
    assert checkpoint_memory_30b._prior_terminal_status(True, manifest) == "passed"

    manifest.write_text(json.dumps({"status": "failed"}), encoding="utf-8")
    assert checkpoint_memory_30b._prior_terminal_status(True, manifest) == "failed"

    assert checkpoint_memory_30b._prior_terminal_status(False, manifest) is None  # resume disabled


def test_reclaim_remote_kills_and_clears_each_ranks_output(tmp_path: Path) -> None:
    seen = []

    class Result:
        returncode = 0
        stderr = ""

    def fake_run(command, **kwargs):
        seen.append(command)
        return Result()

    plan = {"ranks": [
        {"rank": 0, "host": "spark@spark1", "output": "/mnt/a/run"},
        {"rank": 1, "host": "spark@spark2", "output": "/mnt/b/run"},
    ]}

    checkpoint_memory_30b.reclaim_remote(plan, remote_run=fake_run)

    assert len(seen) == 2
    assert "pkill -9 -f -- /mnt/a/run" in seen[0][-1] and "rm -rf -- /mnt/a/run" in seen[0][-1]
    assert "pkill -9 -f -- /mnt/b/run" in seen[1][-1] and "rm -rf -- /mnt/b/run" in seen[1][-1]


def test_local_file_run_replays_files_in_call_order(tmp_path: Path) -> None:
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    first.write_text("first\n", encoding="utf-8")
    second.write_text("second\n", encoding="utf-8")

    run_fn = checkpoint_memory_30b._local_file_run([first, second])

    assert run_fn(["ssh", "whatever"]).stdout == "first\n"
    assert run_fn(["ssh", "whatever"]).stdout == "second\n"


def test_local_file_run_falls_back_when_a_file_is_missing(tmp_path: Path) -> None:
    # A run can be interrupted mid-collection: one rank's file already fetched,
    # the other not. The missing one must fall back to a real fetch, not crash.
    present = tmp_path / "present.jsonl"
    present.write_text("cached\n", encoding="utf-8")
    missing = tmp_path / "missing.jsonl"
    seen = []

    def fallback(*args, **kwargs):
        seen.append(args)
        class Result:
            stdout = "fetched\n"
        return Result()

    run_fn = checkpoint_memory_30b._local_file_run([present, missing], fallback=fallback)

    assert run_fn(["ssh", "a"]).stdout == "cached\n"
    assert run_fn(["ssh", "b"]).stdout == "fetched\n"
    assert seen == [(["ssh", "b"],)]


def _two_node_setup() -> dict:
    node = {
        "host": "spark@spark1", "checkout": "/repo",
        "python": {"trl": "/venv/bin/python", "megatron": "/venv/bin/python"},
        "model_dirs": {checkpoint_memory_30b.MODEL_ID: "/models/qwen"},
        "output_root": {"trl": "/mnt/trl", "megatron": "/mnt/megatron"},
        "env": {}, "data_dir": "/data/ultrachat",
    }
    return {"master_addr": "10.0.0.1", "master_port": 29500, "env": {}, "nodes": [node, {**node, "host": "spark@spark2"}]}


def test_run_memory_matrix_condition_filter_selects_one_condition(tmp_path: Path) -> None:
    setup = _two_node_setup()
    setup_path = tmp_path / "setup.json"
    setup_path.write_text(json.dumps(setup), encoding="utf-8")

    result = checkpoint_memory_30b.run_memory_matrix(
        setup, setup_path, tmp_path / "output",
        execute=False, timeout=60, conditions=["trl-zero3-nvme"],
    )

    assert {item["condition"] for item in result["records"]} == {"trl-zero3-nvme"}


def test_run_memory_matrix_condition_filter_rejects_unknown_name(tmp_path: Path) -> None:
    setup = _two_node_setup()
    setup_path = tmp_path / "setup.json"
    setup_path.write_text(json.dumps(setup), encoding="utf-8")

    try:
        checkpoint_memory_30b.run_memory_matrix(
            setup, setup_path, tmp_path / "output",
            execute=False, timeout=60, conditions=["not-a-real-condition"],
        )
        raise AssertionError("expected ConfigError")
    except checkpoint_memory_30b.run.ConfigError as exc:
        assert "not-a-real-condition" in str(exc)


def test_model_registry_gives_each_model_its_own_cohort_and_megatron_env() -> None:
    """Row selection depends on the tokenizer, so Qwen and GLM must not share a
    cohort directory; GLM additionally needs the Transformer Engine path."""
    node = {"output_root": {"megatron": "/mnt/megatron"}}

    qwen = checkpoint_memory_30b.cohort_path(node, "qwen")
    glm = checkpoint_memory_30b.cohort_path(node, "glm")

    assert qwen != glm
    assert checkpoint_memory_30b.MODELS["qwen"]["megatron_env"] == {}
    assert checkpoint_memory_30b.MODELS["glm"]["megatron_env"] == {"TRANSFORMER_IMPL": "auto"}


def test_memory_experiments_carry_the_selected_model_and_its_megatron_env() -> None:
    glm = {item["name"]: item for item in checkpoint_memory_30b.memory_experiments("glm")}

    megatron_env = glm["len-4096"]["experiment"]["env"]
    assert megatron_env["MODEL_ID"] == "zai-org/GLM-4.7-Flash"
    assert megatron_env["TRANSFORMER_IMPL"] == "auto"
    # TRL conditions take the same model but never the Megatron-only switch.
    trl_env = glm["trl-ddp"]["experiment"]["env"]
    assert trl_env["MODEL_ID"] == "zai-org/GLM-4.7-Flash"
    assert "TRANSFORMER_IMPL" not in trl_env
