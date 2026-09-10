from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments import build, run


MODEL_REVISION = "a" * 40
ULTRACHAT_REVISION = "8049631c405ae6576f93f445c6b8166f76f5505a"


def setup_file(tmp_path: Path, *, data_dir: Path, backend: str = "trl") -> Path:
    setup = {
        "setup": "spark",
        "master_addr": "spark1",
        "master_port": 29670,
        "env": {"NCCL_DEBUG": "WARN"},
        "nodes": [
            {
                "host": f"spark{index + 1}",
                "checkout": "/srv/post-training-unified",
                "python": {"trl": "/env/trl/bin/python", "megatron": "/env/megatron/bin/python"},
                "env": {},
                "model_dirs": {"Qwen/Qwen2.5-0.5B-Instruct": "/models/qwen"},
                "data_dir": str(data_dir),
                "output_root": f"/results/controller/spark{index + 1}",
            }
            for index in range(2)
        ],
    }
    path = tmp_path / "setup.json"
    path.write_text(json.dumps(setup), encoding="utf-8")
    return path


def write_manifest(data_dir: Path, *, dataset: str, dataset_revision: str, train_count: int = 100) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": dataset, "dataset_revision": dataset_revision, "train_count": train_count, "eval_count": 10}
    (data_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_dry_run_maps_knobs_to_experiment_and_delegates_to_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data_dir = tmp_path / "data"
    write_manifest(data_dir, dataset="HuggingFaceH4/ultrachat_200k", dataset_revision=ULTRACHAT_REVISION)
    setup_path = setup_file(tmp_path, data_dir=data_dir)

    exit_code = build.main([
        "--backend", "trl", "--dataset", "ultrachat",
        "--model-id", "Qwen/Qwen2.5-0.5B-Instruct", "--model-revision", MODEL_REVISION,
        "--setup", str(setup_path), "--output", str(tmp_path / "out"),
    ])

    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["mode"] == "dry-run"
    assert printed["backend"] == "trl"
    assert printed["ranks"][0]["env"]["DATASET_ID"] == "HuggingFaceH4/ultrachat_200k"
    assert printed["ranks"][0]["env"]["NNODES"] == "2"


def test_epochs_and_max_steps_are_mutually_exclusive(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        build.parse_args([
            "--backend", "trl", "--dataset", "ultrachat",
            "--model-id", "m", "--model-revision", MODEL_REVISION,
            "--setup", str(tmp_path / "setup.json"), "--output", str(tmp_path / "out"),
            "--epochs", "1", "--max-steps", "5",
        ])


def test_offload_rejected_for_megatron_backend(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    write_manifest(data_dir, dataset="HuggingFaceH4/ultrachat_200k", dataset_revision=ULTRACHAT_REVISION)
    setup_path = setup_file(tmp_path, data_dir=data_dir)
    args = build.parse_args([
        "--backend", "megatron", "--dataset", "ultrachat",
        "--model-id", "m", "--model-revision", MODEL_REVISION,
        "--setup", str(setup_path), "--output", str(tmp_path / "out"),
        "--offload", "nvme",
    ])
    setup = run.load_setup(setup_path)

    with pytest.raises(run.ConfigError, match="Megatron has no offload"):
        build.build_experiment(args, setup)


def test_offload_selects_deepspeed_config_and_forces_backend(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    write_manifest(data_dir, dataset="HuggingFaceH4/ultrachat_200k", dataset_revision=ULTRACHAT_REVISION)
    setup_path = setup_file(tmp_path, data_dir=data_dir)
    args = build.parse_args([
        "--backend", "trl", "--dataset", "ultrachat",
        "--model-id", "m", "--model-revision", MODEL_REVISION,
        "--setup", str(setup_path), "--output", str(tmp_path / "out"),
        "--offload", "nvme", "--finetuning-mode", "full",
    ])
    setup = run.load_setup(setup_path)

    experiment = build.build_experiment(args, setup)

    assert experiment["env"]["DISTRIBUTED_BACKEND"] == "deepspeed"
    assert experiment["env"]["DEEPSPEED_CONFIG"] == "configs/deepspeed-zero3-nvme.json"


def test_nvme_offload_rejects_lora(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    write_manifest(data_dir, dataset="HuggingFaceH4/ultrachat_200k", dataset_revision=ULTRACHAT_REVISION)
    setup_path = setup_file(tmp_path, data_dir=data_dir)
    args = build.parse_args([
        "--backend", "trl", "--dataset", "ultrachat",
        "--model-id", "m", "--model-revision", MODEL_REVISION,
        "--setup", str(setup_path), "--output", str(tmp_path / "out"),
        "--offload", "nvme",
    ])

    with pytest.raises(run.ConfigError, match="requires --finetuning-mode full"):
        build.build_experiment(args, run.load_setup(setup_path))


def test_dataset_choices_exclude_reference_only_presets(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    write_manifest(data_dir, dataset="anything", dataset_revision="c" * 40)
    setup_path = setup_file(tmp_path, data_dir=data_dir)
    args = build.parse_args([
        "--backend", "trl", "--dataset", "swe_trajectories",
        "--model-id", "m", "--model-revision", MODEL_REVISION,
        "--setup", str(setup_path), "--output", str(tmp_path / "out"),
    ])
    setup = run.load_setup(setup_path)

    with pytest.raises(run.ConfigError, match="--dataset must be one of"):
        build.build_experiment(args, setup)


def test_missing_manifest_is_skipped_gracefully_for_trl(tmp_path: Path) -> None:
    # data_dir may be unreadable from controller (e.g. it's the Spark-side NFS
    # path, not controller's) - TRL doesn't need train_count locally, since
    # spark_train.py resolves --epochs itself at run time on the node.
    data_dir = tmp_path / "data"  # never created; unreadable from here
    setup_path = setup_file(tmp_path, data_dir=data_dir)
    args = build.parse_args([
        "--backend", "trl", "--dataset", "ultrachat",
        "--model-id", "m", "--model-revision", MODEL_REVISION,
        "--setup", str(setup_path), "--output", str(tmp_path / "out"),
        "--epochs", "1",
    ])
    setup = run.load_setup(setup_path)

    experiment = build.build_experiment(args, setup)

    assert experiment["env"]["NUM_TRAIN_EPOCHS"] == 1


def test_missing_manifest_blocks_megatron_epoch_conversion(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"  # never created; unreadable from here
    setup_path = setup_file(tmp_path, data_dir=data_dir, backend="megatron")
    args = build.parse_args([
        "--backend", "megatron", "--dataset", "ultrachat",
        "--model-id", "m", "--model-revision", MODEL_REVISION,
        "--setup", str(setup_path), "--output", str(tmp_path / "out"),
        "--epochs", "1",
    ])
    setup = run.load_setup(setup_path)

    with pytest.raises(run.ConfigError, match="cannot convert --epochs"):
        build.build_experiment(args, setup)


def test_manifest_mismatch_fails_fast(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    write_manifest(data_dir, dataset="HuggingFaceH4/no_robots", dataset_revision="c" * 40)
    setup_path = setup_file(tmp_path, data_dir=data_dir)
    args = build.parse_args([
        "--backend", "trl", "--dataset", "ultrachat",
        "--model-id", "m", "--model-revision", MODEL_REVISION,
        "--setup", str(setup_path), "--output", str(tmp_path / "out"),
    ])
    setup = run.load_setup(setup_path)

    with pytest.raises(run.ConfigError, match="Re-run prepare_public_data.py"):
        build.build_experiment(args, setup)


def test_megatron_epochs_convert_to_steps_from_manifest_train_count(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    write_manifest(data_dir, dataset="HuggingFaceH4/ultrachat_200k", dataset_revision=ULTRACHAT_REVISION, train_count=100)
    setup_path = setup_file(tmp_path, data_dir=data_dir, backend="megatron")
    args = build.parse_args([
        "--backend", "megatron", "--dataset", "ultrachat",
        "--model-id", "m", "--model-revision", MODEL_REVISION,
        "--setup", str(setup_path), "--output", str(tmp_path / "out"),
        "--epochs", "2", "--global-batch-size", "8",
    ])
    setup = run.load_setup(setup_path)

    experiment = build.build_experiment(args, setup)

    assert experiment["env"]["MAX_STEPS"] == 25  # ceil(2 * 100 / 8)
    assert experiment["env"]["SCHEDULE_STEPS"] == 25
    assert experiment["env"]["NUM_TRAIN_EPOCHS"] == 2


def test_megatron_defaults_match_verified_30b_presets(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    write_manifest(data_dir, dataset="HuggingFaceH4/ultrachat_200k", dataset_revision=ULTRACHAT_REVISION)
    setup_path = setup_file(tmp_path, data_dir=data_dir, backend="megatron")
    args = build.parse_args([
        "--backend", "megatron", "--dataset", "ultrachat",
        "--model-id", "m", "--model-revision", MODEL_REVISION,
        "--setup", str(setup_path), "--output", str(tmp_path / "out"),
    ])
    setup = run.load_setup(setup_path)

    experiment = build.build_experiment(args, setup)

    assert (experiment["env"]["TP"], experiment["env"]["PP"], experiment["env"]["EP"]) == (1, 1, 2)
