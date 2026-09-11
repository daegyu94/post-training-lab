from pathlib import Path

from observability import resource_sampler


def test_memory_values_are_bytes() -> None:
    values = resource_sampler._memory()

    assert values["memtotal_bytes"] > values["memavailable_bytes"] > 0
    assert values["swaptotal_bytes"] >= values["swapfree_bytes"] >= 0


def test_disk_parser_uses_kernel_sector_units(tmp_path: Path) -> None:
    path = tmp_path / "stat"
    path.write_text("1 2 3 4 5 6 7 8 9 10 11\n")

    result = resource_sampler._disk(path)

    assert result["read_bytes"] == 3 * 512
    assert result["write_bytes"] == 7 * 512
    assert result["busy_time_ms"] == 10
