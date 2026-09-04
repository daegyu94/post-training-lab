import pytest

from megatron_lab.topology import build_layout


def test_tensor_parallel_layout_changes_tp_rank() -> None:
    layout = build_layout(world_size=2, tensor_parallel=2)
    assert layout.data_parallel == 1
    assert [rank.tensor_parallel_rank for rank in layout.ranks] == [0, 1]
    assert [rank.data_parallel_rank for rank in layout.ranks] == [0, 0]


def test_data_parallel_layout_changes_dp_rank() -> None:
    layout = build_layout(world_size=2, tensor_parallel=1)
    assert layout.data_parallel == 2
    assert [rank.tensor_parallel_rank for rank in layout.ranks] == [0, 0]
    assert [rank.data_parallel_rank for rank in layout.ranks] == [0, 1]


def test_layout_rejects_non_divisible_parallelism() -> None:
    with pytest.raises(ValueError, match="divisible"):
        build_layout(world_size=3, tensor_parallel=2)
