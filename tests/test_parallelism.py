import pytest

from megatron_lab.parallelism import build_layout


def test_build_layout_includes_context_parallel_dimension() -> None:
    layout = build_layout(
        world_size=16,
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        context_parallel_size=2,
    )

    assert layout["data_parallel_size"] == 2
    assert len(layout["groups"]["context_parallel"]) == 8
    assert all(len(group) == 2 for group in layout["groups"]["context_parallel"])


def test_layout_rejects_incompatible_world_size() -> None:
    with pytest.raises(ValueError, match="must be divisible"):
        build_layout(
            world_size=8,
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            context_parallel_size=2,
        )
