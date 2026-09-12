from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments import checkpoint_io_probe


def test_latest_iteration_and_shards(tmp_path: Path) -> None:
    (tmp_path / "iter_0000002").mkdir()
    latest = tmp_path / "iter_0000008"
    latest.mkdir()
    (latest / "__0_0.distcp").write_bytes(b"data")

    assert checkpoint_io_probe.latest_iteration(tmp_path) == latest
    assert checkpoint_io_probe.shard_files(latest) == [latest / "__0_0.distcp"]


def test_latest_global_step_picks_highest_numbered_step(tmp_path: Path) -> None:
    (tmp_path / "global_step1").mkdir()
    latest = tmp_path / "global_step10"
    latest.mkdir()

    assert checkpoint_io_probe.latest_global_step(tmp_path) == latest


def test_deepspeed_checkpoint_groups_separates_metadata_from_offloaded_tensors(tmp_path: Path) -> None:
    step_dir = tmp_path / "global_step1"
    step_dir.mkdir()
    (step_dir / "zero_pp_rank_0_mp_rank_00_model_states.pt").write_bytes(b"m" * 10)
    (step_dir / "bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt").write_bytes(b"o" * 20)
    offloaded_dir = step_dir / "offloaded_tensors" / "rank0"
    offloaded_dir.mkdir(parents=True)
    (offloaded_dir / "param.tensor.swp").write_bytes(b"p" * 30)

    groups = checkpoint_io_probe.deepspeed_checkpoint_groups(step_dir)

    assert {path.name for path in groups["metadata"]} == {
        "zero_pp_rank_0_mp_rank_00_model_states.pt", "bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt",
    }
    assert [path.name for path in groups["offloaded_tensors"]] == ["param.tensor.swp"]


def test_deepspeed_checkpoint_groups_tolerates_missing_offloaded_tensors(tmp_path: Path) -> None:
    """offload_param has no checkpoint-copy step (unlike offload_optimizer), and a
    plain SGD run may swap no optimizer state at all -- offloaded_tensors/ can
    legitimately not exist. That must report zero, not fail the probe."""
    step_dir = tmp_path / "global_step1"
    step_dir.mkdir()
    (step_dir / "zero_pp_rank_0_mp_rank_00_model_states.pt").write_bytes(b"m" * 10)

    groups = checkpoint_io_probe.deepspeed_checkpoint_groups(step_dir)

    assert len(groups["metadata"]) == 1
    assert groups["offloaded_tensors"] == []


def test_deepspeed_checkpoint_groups_raises_when_both_empty(tmp_path: Path) -> None:
    step_dir = tmp_path / "global_step1"
    step_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        checkpoint_io_probe.deepspeed_checkpoint_groups(step_dir)


def test_run_probe_deepspeed_layout_reports_grouped_inventory(tmp_path: Path, monkeypatch) -> None:
    checkpoint_dir = tmp_path
    step_dir = checkpoint_dir / "global_step1"
    step_dir.mkdir()
    (step_dir / "zero_pp_rank_0_mp_rank_00_model_states.pt").write_bytes(b"m" * 100)
    offloaded_dir = step_dir / "offloaded_tensors" / "rank0"
    offloaded_dir.mkdir(parents=True)
    (offloaded_dir / "param.tensor.swp").write_bytes(b"p" * 300)

    monkeypatch.setattr(checkpoint_io_probe, "_device_reader", lambda _: ("1:2", lambda: {
        "read_operations": 0, "read_bytes": 0, "write_operations": 0, "write_bytes": 0, "busy_time_ms": 0,
    }))

    result = checkpoint_io_probe.run_probe(checkpoint_dir, include_direct=False, layout="deepspeed")

    assert result["inventory"]["layout"] == "deepspeed"
    assert result["inventory"]["iteration"] == 1
    assert result["inventory"]["logical_bytes"] == 400
    assert result["inventory"]["groups"]["metadata"]["logical_bytes"] == 100
    assert result["inventory"]["groups"]["offloaded_tensors"]["logical_bytes"] == 300
    assert result["scope"] == "raw checkpoint-file read; not deepspeed_load_checkpoint() restore"
    assert result["reads"]["direct"] is None


def test_buffered_read_reports_physical_delta(tmp_path: Path) -> None:
    path = tmp_path / "shard.distcp"
    path.write_bytes(b"x" * 1024)
    samples = iter([
        {"read_operations": 1, "read_bytes": 512, "write_operations": 0, "write_bytes": 0, "busy_time_ms": 2},
        {"read_operations": 3, "read_bytes": 1536, "write_operations": 0, "write_bytes": 0, "busy_time_ms": 5},
    ])

    result = checkpoint_io_probe.buffered_read([path], lambda: next(samples))

    assert result["logical_bytes"] == 1024
    assert result["physical_read_bytes"] == 1024
    assert result["physical_to_logical_ratio"] == 1


def test_direct_read_invokes_dd_with_direct_flag(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "shard.distcp"
    path.write_bytes(b"x" * 512)
    monkeypatch.setattr(checkpoint_io_probe.shutil, "which", lambda _: "/bin/dd")
    seen = []
    samples = iter([
        {"read_operations": 1, "read_bytes": 0, "write_operations": 0, "write_bytes": 0, "busy_time_ms": 0},
        {"read_operations": 2, "read_bytes": 512, "write_operations": 0, "write_bytes": 0, "busy_time_ms": 1},
    ])

    def fake_run(command, **kwargs):
        seen.append(command)
        return SimpleNamespace(returncode=0, stderr="")

    result = checkpoint_io_probe.direct_read([path], lambda: next(samples), run=fake_run)

    assert "iflag=direct" in seen[0]
    assert result["physical_read_bytes"] == 512
