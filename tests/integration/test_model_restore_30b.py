from experiments import model_restore_30b


def _summary(seconds: float, storage: int) -> dict:
    return {
        "model_restore_seconds": seconds,
        "restore_input_logical_bytes": 100,
        "restore_process_io": {"read_bytes": storage},
    }


def test_restore_summary_uses_three_run_means() -> None:
    records = [
        {
            "train": {
                "checkpoint_save_seconds": value,
                "checkpoint_logical_bytes": value * 100,
                "trainable_parameter_fraction": value / 100,
            },
            "cold": _summary(value, int(value * 10)),
            "warm": _summary(value * 2, int(value)),
        }
        for value in (1.0, 2.0, 3.0)
    ]

    result = model_restore_30b.summarize_records(records)

    assert result["run_count"] == 3
    assert result["train"]["checkpoint_save_seconds_mean"] == 2
    assert result["cold"]["model_restore_seconds_mean"] == 2
    assert result["warm"]["process_storage_read_bytes_mean"] == 2


def test_single_node_restore_uses_explicit_trl_checkpoint() -> None:
    value = model_restore_30b.experiment("qwen", "lora-r64", "tuned", "/mnt/checkpoint")

    assert value["backend"] == "trl"
    assert value["nnodes"] == 1
    assert value["env"]["STAGE"] == "tuned"
    assert value["env"]["LOAD_DIR"] == "/mnt/checkpoint"
    assert value["env"]["RESTORE_ONLY"] is True
    assert value["env"]["LORA_R"] == 64


def test_full_variant_uses_single_node_zero3_nvme() -> None:
    value = model_restore_30b.experiment("glm", "zero3-full", "train")

    assert value["env"]["FINETUNING_MODE"] == "full"
    assert value["env"]["DISTRIBUTED_BACKEND"] == "deepspeed"
    assert value["env"]["MAX_STEPS"] == 1
    assert value["env"]["MAX_LENGTH"] == 512
