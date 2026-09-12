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


def test_resource_window_isolates_bytes_inside_the_window() -> None:
    def sample(wall_ns, read_bytes, write_bytes):
        return json.dumps({"wall_time_ns": wall_ns, "read_bytes": read_bytes, "write_bytes": write_bytes})

    lines = [
        sample(1_000, 0, 0),
        sample(2_000, 100, 200),
        sample(3_000, 150, 260),
        sample(4_000, 999, 999),
    ]

    result = checkpoint_memory_30b.summarize_resources_window(lines, 2_000, 3_000)

    assert result["sample_count"] == 2
    assert result["insufficient_resolution"] is False
    assert result["device_read_bytes_delta"] == 50
    assert result["device_write_bytes_delta"] == 60


def test_resource_window_flags_insufficient_resolution_below_two_samples() -> None:
    lines = [json.dumps({"wall_time_ns": 2_000, "read_bytes": 100, "write_bytes": 200})]

    result = checkpoint_memory_30b.summarize_resources_window(lines, 1_000, 3_000)

    assert result["sample_count"] == 1
    assert result["insufficient_resolution"] is True
    assert result["device_read_bytes_delta"] is None


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

    assert set(conditions) == {
        "len-4096", "len-8192", "trl-ddp", "trl-fsdp2", "trl-zero3-nvme", "trl-zero3-nvme-checkpoint-io",
    }
    assert conditions["len-4096"]["experiment"]["env"]["PAD_TO_MAX_LENGTH"] is True
    assert conditions["trl-ddp"]["experiment"]["env"]["PAD_TO_MAX_LENGTH"] is True
    zero = conditions["trl-zero3-nvme"]["experiment"]["env"]
    assert zero["FINETUNING_MODE"] == "full"
    assert zero["OPTIMIZER"] == "sgd"
    assert zero["DEEPSPEED_CONFIG"].endswith("deepspeed-zero3-nvme.json")
    checkpoint_io = conditions["trl-zero3-nvme-checkpoint-io"]["experiment"]["env"]
    assert checkpoint_io["STAGE"] == "all"
    assert checkpoint_io["FINETUNING_MODE"] == "full"


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


def test_collect_deepspeed_checkpoint_io_fetches_timing_and_probe_and_windows_bytes(tmp_path: Path) -> None:
    run_output = tmp_path
    (run_output / "measurements").mkdir()
    resource_lines = "\n".join([
        json.dumps({"wall_time_ns": 1_000, "read_bytes": 0, "write_bytes": 0}),
        json.dumps({"wall_time_ns": 2_000, "read_bytes": 500, "write_bytes": 700}),
    ]) + "\n"
    (run_output / "measurements" / "resources-node-0.jsonl").write_text(resource_lines)

    save_timing = {
        "event": "save", "rank": 0, "host_seconds": 1.0,
        "start_wall_ns": 1_000, "end_wall_ns": 2_000, "path": "/mnt/a/run/model",
    }
    probe_result = {
        "inventory": {"groups": {"metadata": {"logical_bytes": 10}, "offloaded_tensors": {"logical_bytes": 20}}},
        "reads": {},
    }

    def fake_run(command, **kwargs):
        tail = command[-1]
        if "checkpoint-save-timing-rank-0.json" in tail:
            return SimpleNamespace(returncode=0, stdout=json.dumps(save_timing) + "\n", stderr="")
        if "checkpoint-restore-timing-rank-0.json" in tail:
            return SimpleNamespace(returncode=1, stdout="", stderr="")
        if "checkpoint_io_probe.py" in tail:
            return SimpleNamespace(returncode=0, stdout=json.dumps(probe_result) + "\n", stderr="")
        raise AssertionError(f"unexpected command: {tail}")

    plan = {"ranks": [
        {"rank": 0, "host": "spark@spark1", "checkout": "/repo", "output": "/mnt/a/run",
         "env": {"PYTHON": "/venv/bin/python"}},
    ]}

    result = checkpoint_memory_30b.collect_deepspeed_checkpoint_io(plan, run_output, remote_run=fake_run)

    rank0 = result["0"]
    assert rank0["timing"]["save"]["host_seconds"] == 1.0
    assert rank0["timing"]["restore"] is None
    assert rank0["checkpoint_io"]["inventory"]["groups"]["offloaded_tensors"]["logical_bytes"] == 20
    assert rank0["windows"]["save"]["device_read_bytes_delta"] == 500
    assert rank0["windows"]["restore"] is None
    assert (run_output / "measurements" / "deepspeed-checkpoint-io-rank-0.json").exists()


def test_cleanup_trl_checkpoint_removes_each_ranks_model_dir() -> None:
    seen = []

    def fake_run(command, **kwargs):
        seen.append(command)
        return SimpleNamespace(returncode=0, stderr="")

    plan = {"ranks": [
        {"rank": 0, "host": "spark@spark1", "output": "/mnt/a/run"},
        {"rank": 1, "host": "spark@spark2", "output": "/mnt/b/run"},
    ]}

    checkpoint_memory_30b.cleanup_trl_checkpoint(plan, Path("/unused"), {}, remote_run=fake_run)

    assert len(seen) == 2
    assert "rm -rf -- /mnt/a/run/model" in seen[0][-1]
    assert "rm -rf -- /mnt/b/run/model" in seen[1][-1]


def test_deepspeed_checkpoint_summary_reports_medians_and_missing_restore() -> None:
    def record(save_seconds, restore_seconds, metadata_bytes, offloaded_bytes):
        per_rank = {
            "0": {
                "timing": {
                    "save": {"host_seconds": save_seconds},
                    "restore": {"host_seconds": restore_seconds} if restore_seconds is not None else None,
                },
                "checkpoint_io": {"inventory": {"groups": {
                    "metadata": {"logical_bytes": metadata_bytes},
                    "offloaded_tensors": {"logical_bytes": offloaded_bytes},
                }}},
            },
        }
        return {"status": "passed", "warmup": False, "deepspeed_checkpoint_io": per_rank}

    records = [
        record(1.0, None, 100, 0),
        record(1.2, 0.5, 100, 0),
        record(1.1, 0.6, 100, 0),
    ]

    summary = checkpoint_memory_30b.deepspeed_checkpoint_summary(records)

    assert summary["run_count"] == 3
    assert summary["metadata_bytes_median"] == 100
    assert summary["offloaded_tensors_bytes_median"] == 0
    assert summary["save_call_seconds_median_max_across_ranks"] == 1.1
    assert summary["restore_call_seconds_median_max_across_ranks"] == 0.55


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
        execute=False, timeout=60, conditions=["trl-zero3-nvme-checkpoint-io"],
    )

    assert {item["condition"] for item in result["records"]} == {"trl-zero3-nvme-checkpoint-io"}


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
