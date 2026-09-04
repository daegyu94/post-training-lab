import pytest

from megatron_lab.topology import build_topology, rank_coordinates


def test_build_topology_derives_data_parallel_size() -> None:
    topology = build_topology(2, 8, 2, 2, 1, 1)
    assert topology.world_size == 16
    assert topology.data_parallel == 4


def test_rank_coordinates_follow_default_megatron_order() -> None:
    topology = build_topology(2, 4, 2, 2, 1, 1)
    assert rank_coordinates(topology, 0)["tp"] == 0
    assert rank_coordinates(topology, 1)["tp"] == 1
    assert rank_coordinates(topology, 2)["dp"] == 1
    assert rank_coordinates(topology, 4)["pp"] == 1


def test_invalid_parallel_product_is_rejected() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        build_topology(1, 2, 2, 2, 1, 1)
