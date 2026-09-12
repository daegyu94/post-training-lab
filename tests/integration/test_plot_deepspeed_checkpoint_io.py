from experiments import plot_deepspeed_checkpoint_io as plot_mod


def _record(save_seconds, restore_seconds, metadata_bytes, offloaded_bytes, save_window_write, whole_run_write):
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
            "windows": {
                "save": {"insufficient_resolution": False, "device_write_bytes_delta": save_window_write},
                "restore": None,
            },
        },
    }
    return {
        "status": "passed", "warmup": False, "run": 0,
        "deepspeed_checkpoint_io": per_rank,
        "resources": {"0": {"device_write_bytes_delta": whole_run_write}},
    }


def test_measured_records_excludes_warmup_and_failed() -> None:
    passed = _record(1.0, 0.5, 10, 20, 5, 50)
    warmup = {**_record(1.0, 0.5, 10, 20, 5, 50), "warmup": True}
    failed = {**_record(1.0, 0.5, 10, 20, 5, 50), "status": "failed"}

    manifest = {"records": [passed, warmup, failed]}

    assert plot_mod.measured_records(manifest) == [passed]


def test_per_run_bytes_sums_across_ranks() -> None:
    record = _record(1.0, 0.5, 10, 20, 5, 50)

    assert plot_mod.per_run_bytes(record) == {"metadata": 10, "offloaded_tensors": 20}


def test_per_run_seconds_returns_none_when_event_absent() -> None:
    record = _record(1.0, None, 10, 20, 5, 50)

    assert plot_mod.per_run_seconds(record, "save") == 1.0
    assert plot_mod.per_run_seconds(record, "restore") is None


def test_per_run_window_bytes_skips_insufficient_resolution() -> None:
    record = _record(1.0, 0.5, 10, 20, 5, 50)
    record["deepspeed_checkpoint_io"]["0"]["windows"]["save"]["insufficient_resolution"] = True

    assert plot_mod.per_run_window_bytes(record, "save", "write") is None
