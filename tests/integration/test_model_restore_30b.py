from experiments import model_restore_30b


def _rank(seconds: float, logical: int, storage: int) -> dict:
    return {"metrics": {
        "load_call_host_seconds_sum_per_rank": seconds,
        "load_call_count_per_rank": 1,
        "load_process_logical_read_bytes_sum_per_rank": logical,
        "load_process_storage_read_bytes_sum_per_rank": storage,
        "model_ready_seconds_per_rank": seconds + 10,
    }}


def test_restore_summary_uses_cluster_critical_path() -> None:
    result = model_restore_30b.summarize_restore({
        "0": _rank(2, 100, 80),
        "1": _rank(4, 200, 160),
    })

    assert result["restore_seconds_max_across_ranks"] == 4
    assert result["effective_logical_read_bytes_per_second"] == 75
    assert result["observed_storage_read_bytes_per_second"] == 60
    assert result["model_ready_seconds_max_across_ranks"] == 14


def test_restore_schedule_pairs_cold_and_warm_after_one_warmup() -> None:
    schedule = model_restore_30b.restore_schedule(2)

    assert [(item["cache_state"], item["warmup"]) for item in schedule] == [
        ("cold", True), ("warm", True),
        ("cold", False), ("warm", False),
        ("cold", False), ("warm", False),
    ]


def test_restore_experiment_loads_model_only_from_explicit_checkpoint() -> None:
    value = model_restore_30b.experiment("qwen", "tuned", "/mnt/checkpoint")

    assert value["env"]["STAGE"] == "tuned"
    assert value["env"]["LOAD_CHECKPOINT"] == "/mnt/checkpoint"
    assert value["env"]["SAVE_OPTIMIZER"] is False
    assert value["env"]["LOAD_OPTIMIZER"] is False
