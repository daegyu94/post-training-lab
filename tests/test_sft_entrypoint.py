"""Exercise the cluster-only CLI and stage dispatch without a GPU runtime."""

import json
import sys
from types import ModuleType

import pytest

from megatron_lab import sft
from megatron_lab.config import ModelSpec


def stage_argv(tmp_path, stage="train"):
    return [
        "sft", "--stage", stage,
        "--model-id", "Qwen/Qwen2.5-0.5B-Instruct",
        "--model-revision", "a" * 40,
        "--dataset-revision", "b" * 40,
        "--model-dir", str(tmp_path / "model"),
        "--train-data", str(tmp_path / "training.jsonl"),
        "--eval-data", str(tmp_path / "validation.jsonl"),
        "--output-dir", str(tmp_path / "output"),
        "--ep", "1",
    ]


def test_removed_single_setup_is_rejected_before_runtime_import(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sys, "argv", stage_argv(tmp_path) + ["--setup", "single"])
    with pytest.raises(SystemExit) as error:
        sft.main()
    assert error.value.code == 2
    assert "invalid choice: 'single'" in capsys.readouterr().err


@pytest.mark.parametrize("flag", ["--model-id", "--model-revision", "--dataset-revision"])
def test_cluster_identity_is_required(monkeypatch, tmp_path, capsys, flag):
    argv = stage_argv(tmp_path)
    index = argv.index(flag)
    del argv[index:index + 2]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as error:
        sft.main()
    assert error.value.code == 2
    assert flag in capsys.readouterr().err


@pytest.mark.parametrize("stage", ["base", "train", "resume", "tuned"])
def test_default_cli_dispatches_cluster_stage_with_metadata(monkeypatch, tmp_path, stage):
    monkeypatch.setattr(sys, "argv", stage_argv(tmp_path, stage))
    monkeypatch.delenv("RANK_LOG_DIR", raising=False)
    for key, value in {"RANK": "0", "NNODES": "2", "NODE_RANK": "0", "NPROC_PER_NODE": "1"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(sft, "validate_snapshot", lambda *args: {"verified": True})
    config = object()
    calls = []

    def build(args):
        assert args.setup == "spark-cluster"
        assert args.stage == stage
        assert args.topology.world_size == 2
        assert args.topology.data_parallel_size == 2
        return config, ModelSpec("qwen2", "qwen2", args.model_id)

    monkeypatch.setattr(sft, "build_cluster_config", build)
    training = ModuleType("megatron.bridge.training.finetune")
    training.finetune = lambda **kwargs: calls.append(kwargs)
    step = ModuleType("megatron.bridge.training.gpt_step")
    step.forward_step = object()
    monkeypatch.setitem(sys.modules, training.__name__, training)
    monkeypatch.setitem(sys.modules, step.__name__, step)

    sft.main()

    assert calls == [{"config": config, "forward_step_func": step.forward_step}]
    metadata = json.loads((tmp_path / "output" / f"run-metadata-{stage}.json").read_text())
    assert metadata["setup"] == "spark-cluster"
    assert metadata["model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert metadata["model_revision"] == "a" * 40
    assert metadata["dataset_revision"] == "b" * 40
    assert metadata["topology"]["world_size"] == 2
    expected = str(tmp_path / "output" / "checkpoints") if stage in {"resume", "tuned"} else None
    assert metadata["load_checkpoint"] == expected
