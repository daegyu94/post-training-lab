"""The estimator is only worth anything if it lands near the runs we actually measured.

Reference measurements come from docs/experiments.md's 30B GPU Results, all
on two Spark nodes, one process per node, BF16, sequence length 2048.
"""

import json
import struct
from pathlib import Path

import pytest

from experiments.estimate_memory import (
    GIB,
    ModelWeights,
    estimate_memory,
    read_safetensors_weights,
)

# Read off the real Qwen3-30B-A3B snapshot on spark1 by read_safetensors_weights
# itself, so these constants are ground truth rather than a reconstruction.
QWEN3_30B = ModelWeights(expert_bytes=57_982_058_496, dense_bytes=3_082_186_752, tensor_count=18867)
QWEN3_HIDDEN, QWEN3_LAYERS = 2048, 48

# docs/experiments.md, 30B GPU Results.
MEASURED_GIB = {"megatron_lora_ep2": 34.759, "trl_ddp_lora": 58.825, "trl_fsdp2_lora": 32.147}
# The estimate omits CUDA context, fragmentation and framework workspaces, so it
# reads low; anything beyond this band means the model of a backend is wrong.
TOLERANCE = 0.15


def _gib(value: int) -> float:
    return value / GIB


def _assert_near_measured(estimate, key: str) -> None:
    measured = MEASURED_GIB[key]
    predicted = _gib(estimate.total_bytes)
    assert abs(predicted - measured) / measured < TOLERANCE, f"{key}: predicted {predicted:.1f} vs measured {measured}"


def test_megatron_lora_ep2_lands_near_the_measured_34_8_gib():
    """Measured: peak allocated 34.759 GiB (TP=1, PP=1, EP=2, DP=2, LoRA)."""
    estimate = estimate_memory(
        QWEN3_30B,
        backend="megatron",
        finetuning_mode="lora",
        optimizer="adam",
        world_size=2,
        tensor_parallel=1,
        pipeline_parallel=1,
        expert_parallel=2,
        micro_batch_size=1,
        seq_len=2048,
        hidden_size=QWEN3_HIDDEN,
        layers=QWEN3_LAYERS,
        recompute="full",
    )
    _assert_near_measured(estimate, "megatron_lora_ep2")


def test_trl_ddp_lora_lands_near_the_measured_58_8_gib():
    """Measured: peak allocated 58.825 GiB. DDP replicates every parameter."""
    estimate = estimate_memory(
        QWEN3_30B,
        backend="trl",
        distributed_backend="ddp",
        finetuning_mode="lora",
        optimizer="adam",
        world_size=2,
        micro_batch_size=1,
        seq_len=2048,
        hidden_size=QWEN3_HIDDEN,
        layers=QWEN3_LAYERS,
        recompute="full",
    )
    _assert_near_measured(estimate, "trl_ddp_lora")


def test_trl_fsdp2_lora_lands_near_the_measured_32_1_gib():
    """Measured: peak allocated 32.147 GiB. FSDP2 shards parameters across both ranks."""
    estimate = estimate_memory(
        QWEN3_30B,
        backend="trl",
        distributed_backend="fsdp2",
        finetuning_mode="lora",
        optimizer="adam",
        world_size=2,
        micro_batch_size=1,
        seq_len=2048,
        hidden_size=QWEN3_HIDDEN,
        layers=QWEN3_LAYERS,
        recompute="full",
    )
    _assert_near_measured(estimate, "trl_fsdp2_lora")


def test_full_parameter_30b_is_predicted_not_to_fit_this_cluster():
    """The question item 1 spent real GPU time failing to answer: 119 GiB per node."""
    estimate = estimate_memory(
        QWEN3_30B,
        backend="megatron",
        finetuning_mode="full",
        optimizer="adam",
        world_size=2,
        expert_parallel=2,
        micro_batch_size=1,
        seq_len=2048,
        hidden_size=QWEN3_HIDDEN,
        layers=QWEN3_LAYERS,
        recompute="full",
    )
    assert _gib(estimate.total_bytes) > 119, _gib(estimate.total_bytes)


def test_offload_moves_optimizer_off_device_and_says_where_it_went():
    on_device = estimate_memory(
        QWEN3_30B, backend="trl", distributed_backend="deepspeed", zero_stage=3,
        finetuning_mode="full", optimizer="adam", world_size=2, offload="nvme",
        hidden_size=QWEN3_HIDDEN, layers=QWEN3_LAYERS,
    )
    assert on_device.optimizer_bytes == 0 and on_device.gradients_bytes == 0
    assert any("nvme" in note for note in on_device.notes)
    assert any("GiB of state must fit" in note for note in on_device.notes)


def test_zero_stages_shard_progressively_more():
    common = dict(
        backend="trl", distributed_backend="deepspeed", finetuning_mode="full",
        optimizer="adam", world_size=2, hidden_size=QWEN3_HIDDEN, layers=QWEN3_LAYERS,
    )
    totals = [estimate_memory(QWEN3_30B, zero_stage=stage, **common).total_bytes for stage in (0, 1, 2, 3)]
    assert totals == sorted(totals, reverse=True), totals


def test_rejects_unknown_optimizer_and_recompute():
    with pytest.raises(ValueError, match="optimizer"):
        estimate_memory(QWEN3_30B, backend="trl", finetuning_mode="full", optimizer="lion", world_size=1)
    with pytest.raises(ValueError, match="recompute"):
        estimate_memory(
            QWEN3_30B, backend="trl", finetuning_mode="full", optimizer="adam", world_size=1,
            hidden_size=QWEN3_HIDDEN, layers=QWEN3_LAYERS, recompute="partial",
        )


def test_reads_real_tensor_sizes_and_separates_expert_weights(tmp_path: Path):
    header = {
        "model.layers.0.self_attn.q_proj.weight": {"dtype": "BF16", "shape": [4, 4], "data_offsets": [0, 32]},
        "model.layers.0.mlp.experts.0.up_proj.weight": {"dtype": "BF16", "shape": [8, 4], "data_offsets": [32, 96]},
        "__metadata__": {"format": "pt"},
    }
    blob = json.dumps(header).encode()
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * 96)

    weights = read_safetensors_weights(tmp_path)

    assert weights.tensor_count == 2
    assert weights.dense_bytes == 32
    assert weights.expert_bytes == 64
