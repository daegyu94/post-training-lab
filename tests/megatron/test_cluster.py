import argparse
import json
import os
import subprocess
import sys
import types
from functools import partial
from argparse import Namespace
from pathlib import Path

import pytest

from megatron_lab.cluster import ClusterTopology
from megatron_lab.config import (
    DATASET_ID,
    ModelSpec,
    _resolve_cluster_load_checkpoint,
    _set_transformer_layer_spec,
    detect_model_config,
    select_transformer_impl,
    validate_reshardable_checkpoint_options,
    validate_dataset_manifest,
    validate_model_identity,
)
from megatron_lab.sft import parse_args, write_run_metadata
from megatron_lab.sft import positive_int
from megatron_lab.feature_lab import make_feature_run, summarize_timings


def test_cluster_topology_separates_dense_and_expert_data_parallelism() -> None:
    topology = ClusterTopology(
        nnodes=2,
        nproc_per_node=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        expert_parallel_size=2,
        micro_batch_size=1,
        global_batch_size=8,
    ).validate()

    assert topology.world_size == 2
    assert topology.data_parallel_size == 2
    assert topology.expert_data_parallel_size == 1
    assert topology.gradient_accumulation_steps == 4


def test_cluster_topology_rejects_nondivisible_global_batch() -> None:
    with pytest.raises(ValueError, match="global_batch_size"):
        ClusterTopology(global_batch_size=7).validate()


def test_cluster_topology_allows_tp_and_ep_to_overlap_on_two_ranks() -> None:
    topology = ClusterTopology(
        tensor_parallel_size=2,
        expert_parallel_size=2,
        global_batch_size=8,
    ).validate()

    assert topology.data_parallel_size == 1
    assert topology.expert_data_parallel_size == 1


def test_model_config_detection_uses_model_type(tmp_path: Path) -> None:
    model_dir = tmp_path / "glm"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "glm4_moe_lite", "_name_or_path": "local"}),
        encoding="utf-8",
    )

    spec = detect_model_config(model_dir)

    assert spec.family == "glm4_moe_lite"
    assert spec.model_type == "glm4_moe_lite"


def test_model_identity_and_dataset_manifest_are_checked(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3_moe"}), encoding="utf-8"
    )
    spec = detect_model_config(model_dir)
    validate_model_identity("Qwen/Qwen3-30B-A3B", spec)
    with pytest.raises(ValueError, match="disagrees"):
        validate_model_identity("zai-org/GLM-4.7-Flash", spec)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    revision = "a" * 40
    (data_dir / "manifest.json").write_text(
        json.dumps({"dataset": DATASET_ID, "dataset_revision": revision}),
        encoding="utf-8",
    )
    assert validate_dataset_manifest(data_dir / "training.jsonl", revision)[
        "dataset"
    ] == DATASET_ID
    with pytest.raises(ValueError, match="revision"):
        validate_dataset_manifest(data_dir / "training.jsonl", "b" * 40)


def test_transformer_backend_defaults_and_explicit_constraints() -> None:
    qwen = ModelSpec("qwen3_moe", "qwen3_moe", None)
    glm = ModelSpec("glm4_moe_lite", "glm4_moe_lite", None)

    assert select_transformer_impl(qwen) == "local"
    assert select_transformer_impl(glm) == "transformer_engine"
    assert select_transformer_impl(qwen, "transformer_engine") == "transformer_engine"
    with pytest.raises(ValueError, match="GLM-4.7-Flash requires"):
        select_transformer_impl(glm, "local")
    with pytest.raises(ValueError, match="sequence parallel requires"):
        select_transformer_impl(qwen, "local", sequence_parallel=True)


def test_fully_reshardable_checkpoint_options_are_explicit_and_stage_scoped() -> None:
    base = Namespace(
        setup="spark-cluster",
        stage="train",
        dist_ckpt_optim_fully_reshardable=True,
        distributed_optimizer=True,
        save_optimizer=True,
    )
    validate_reshardable_checkpoint_options(base)

    base.distributed_optimizer = False
    with pytest.raises(ValueError, match="require distributed_optimizer"):
        validate_reshardable_checkpoint_options(base)
    base.distributed_optimizer = True
    base.save_optimizer = False
    with pytest.raises(ValueError, match="requires save_optimizer"):
        validate_reshardable_checkpoint_options(base)

    base.stage = "resume"
    validate_reshardable_checkpoint_options(base)

    defaults = Namespace(
        setup="spark-cluster",
        stage="train",
        dist_ckpt_optim_fully_reshardable=False,
        distributed_optimizer=False,
        save_optimizer=False,
    )
    validate_reshardable_checkpoint_options(defaults)


def test_qwen_transformer_engine_uses_official_layer_spec(monkeypatch) -> None:
    marker = object()
    module = types.ModuleType("megatron.bridge.models.gpt_provider")
    module.local_layer_spec = object()
    module.transformer_engine_layer_spec = marker
    monkeypatch.setitem(sys.modules, "megatron.bridge.models.gpt_provider", module)
    model = types.SimpleNamespace(transformer_layer_spec=None)

    _set_transformer_layer_spec(
        model, ModelSpec("qwen3_moe", "qwen3_moe", None), "transformer_engine"
    )

    assert model.transformer_layer_spec is marker


def test_sequence_parallel_feature_selects_transformer_engine() -> None:
    script = (Path(__file__).parents[2] / "backends" / "megatron" / "scripts/run_feature_lab.sh").read_text()
    assert "sequence-parallel:off) tp=2; transformer_impl=transformer_engine" in script
    assert "sequence-parallel:on) tp=2; sequence_parallel=true; transformer_impl=transformer_engine" in script
    assert 'TRANSFORMER_IMPL="$transformer_impl"' in script


def test_feature_variants_are_explicit_and_warmup_is_excluded() -> None:
    run = make_feature_run("checkpoint", "async", 1, warmup=True)
    assert run.model_scope == "small-dense-feature-model"
    summary = summarize_timings(
        [
            {"variant": "sync", "warmup": True, "elapsed_seconds": 2.0},
            {"variant": "sync", "warmup": False, "elapsed_seconds": 3.0},
            {"variant": "sync", "warmup": False, "elapsed_seconds": 4.0},
        ]
    )
    assert summary["measured_repeats"]["sync"]["count"] == 2
    assert summary["measured_repeats"]["sync"]["median_seconds"] == 3.5


def test_run_metadata_records_actual_input_files_and_dataset(tmp_path: Path) -> None:
    args = Namespace(
        setup="spark-cluster",
        stage="train",
        model_id="Qwen/Qwen3-30B-A3B",
        model_revision="model-rev",
        dataset_id="HuggingFaceH4/no_robots",
        dataset_revision="dataset-rev",
        train_data=tmp_path / "training.jsonl",
        eval_data=tmp_path / "validation.jsonl",
        output_dir=tmp_path / "out",
        seed=42,
        finetuning_mode="lora",
        max_length=512,
        max_steps=5,
        schedule_steps=10,
        optimizer="adam",
        checkpoint_mode="sync",
        fully_parallel_save=True,
        fully_parallel_load=True,
        overlap_grad_reduce=False,
    )
    topology = ClusterTopology(global_batch_size=8, micro_batch_size=1)
    write_run_metadata(args, types.SimpleNamespace(model_type="qwen3_moe"), topology)
    metadata = json.loads(
        (args.output_dir / "run-metadata-train.json").read_text(encoding="utf-8")
    )
    assert metadata["dataset"] == "HuggingFaceH4/no_robots"
    assert metadata["train_split"] == "training.jsonl"
    assert metadata["eval_split"] == "validation.jsonl"
    assert metadata["input_files"]["train"].endswith("training.jsonl")
    assert metadata["configuration"]["save_interval"] == 5
    assert metadata["configuration"]["dist_ckpt_optim_fully_reshardable"] is False
    assert metadata["configuration"]["pad_to_max_length"] is False
    source = tmp_path / "iter_0000004"
    source.mkdir()
    args.stage = "resume"
    args.load_checkpoint = source
    args.dist_ckpt_optim_fully_reshardable = True
    args.pad_to_max_length = True
    write_run_metadata(args, types.SimpleNamespace(model_type="qwen3_moe"), topology)
    resumed_metadata = json.loads(
        (args.output_dir / "run-metadata-resume.json").read_text(encoding="utf-8")
    )
    assert resumed_metadata["load_checkpoint"] == str(source)
    assert resumed_metadata["configuration"]["dist_ckpt_optim_fully_reshardable"] is True
    assert resumed_metadata["configuration"]["pad_to_max_length"] is True


def test_pad_to_max_length_cli_defaults_off_and_accepts_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    argv = ["sft", "--stage", "train", "--model-id", "m", "--model-revision", "a" * 40,
            "--dataset-revision", "b" * 40, "--model-dir", "model", "--train-data", "train",
            "--eval-data", "eval", "--output-dir", "out"]
    monkeypatch.setattr(sys, "argv", argv)
    assert parse_args().pad_to_max_length is False
    monkeypatch.setattr(sys, "argv", argv + ["--pad-to-max-length"])
    assert parse_args().pad_to_max_length is True


def test_save_interval_cli_requires_positive_value() -> None:
    assert positive_int("4") == 4
    with pytest.raises(argparse.ArgumentTypeError, match="greater than zero"):
        positive_int("0")


def test_load_checkpoint_is_existing_source_and_stage_scoped(tmp_path: Path) -> None:
    source = tmp_path / "iter_0000004"
    source.mkdir()
    args = Namespace(stage="resume", load_checkpoint=source)
    assert _resolve_cluster_load_checkpoint(args, tmp_path / "new" / "checkpoints") == source
    default = tmp_path / "new" / "checkpoints"
    default.mkdir(parents=True)
    assert _resolve_cluster_load_checkpoint(
        Namespace(stage="tuned", load_checkpoint=None), default
    ) == default

    with pytest.raises(ValueError, match="does not exist"):
        _resolve_cluster_load_checkpoint(
            Namespace(stage="resume", load_checkpoint=tmp_path / "missing"),
            tmp_path / "new" / "checkpoints",
        )
    with pytest.raises(ValueError, match="only for setup2"):
        _resolve_cluster_load_checkpoint(
            Namespace(stage="train", load_checkpoint=source),
            tmp_path / "new" / "checkpoints",
        )


def test_glm_local_layer_spec_is_rejected_without_transformer_engine():
    model = types.SimpleNamespace(
        transformer_layer_spec=partial(
            lambda **kwargs: kwargs, use_transformer_engine=False
        )
    )
    with pytest.raises(RuntimeError, match="requires a Transformer Engine"):
        _set_transformer_layer_spec(
            model, ModelSpec("glm4_moe_lite", "glm4_moe_lite", None)
        )


def test_qwen_layer_spec_stays_core_local(monkeypatch):
    marker = object()
    megatron = types.ModuleType("megatron")
    bridge = types.ModuleType("megatron.bridge")
    models = types.ModuleType("megatron.bridge.models")
    megatron.__path__ = []
    bridge.__path__ = []
    models.__path__ = []
    bridge.models = models
    megatron.bridge = bridge
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.bridge", bridge)
    monkeypatch.setitem(sys.modules, "megatron.bridge.models", models)
    module = types.ModuleType("megatron.bridge.models.gpt_provider")
    module.local_layer_spec = marker
    monkeypatch.setitem(sys.modules, "megatron.bridge.models.gpt_provider", module)
    model = types.SimpleNamespace(transformer_layer_spec=None)
    _set_transformer_layer_spec(model, ModelSpec("qwen3_moe", "qwen3_moe", None))
    assert model.transformer_layer_spec is marker


def test_glm_config_uses_local_provider_and_known_options_only(tmp_path: Path, monkeypatch) -> None:
    """A strict fake catches remote recipe calls and misspelled provider fields."""

    model_dir = tmp_path / "glm"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "glm4_moe_lite"}), encoding="utf-8"
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "manifest.json").write_text(
        json.dumps({"dataset": DATASET_ID, "dataset_revision": "d" * 40}),
        encoding="utf-8",
    )
    canonical = {
        "prompt_id": "train-1",
        "messages": [
            {"role": "user", "content": "Say hello."},
            {"role": "assistant", "content": "Hello."},
        ],
        "provenance": {"source": "test"},
    }
    (data_dir / "training.jsonl").write_text(json.dumps(canonical) + "\n", encoding="utf-8")
    canonical["prompt_id"] = "eval-1"
    (data_dir / "validation.jsonl").write_text(json.dumps(canonical) + "\n", encoding="utf-8")

    class StrictProvider:
        __slots__ = (
            "tensor_model_parallel_size", "pipeline_model_parallel_size",
            "expert_model_parallel_size", "expert_tensor_parallel_size",
            "context_parallel_size", "sequence_parallel", "seq_length",
            "transformer_impl", "transformer_layer_spec", "bf16", "cross_entropy_loss_fusion",
            "cross_entropy_fusion_impl", "attention_backend", "apply_rope_fusion",
            "persist_layer_norm", "bias_activation_fusion", "bias_dropout_fusion",
            "moe_router_fusion", "mtp_num_layers", "cuda_graph_impl",
            "moe_token_dispatcher_type", "moe_flex_dispatcher_backend",
            "moe_shared_expert_overlap", "moe_grouped_gemm",
            "moe_permute_fusion", "recompute_granularity", "recompute_method",
            "recompute_num_layers", "overlap_moe_expert_parallel_comm",
        )

        def __init__(self):
            for field in self.__slots__:
                setattr(self, field, None)
            self.transformer_layer_spec = partial(
                lambda **kwargs: kwargs, use_transformer_engine=True
            )

    class Component:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.tokenizer = types.SimpleNamespace()
            self.validation = types.SimpleNamespace()
            self.ddp = types.SimpleNamespace()
            self.rng = types.SimpleNamespace()
            self.peft = None

    class FakeAutoBridge:
        path = None

        @classmethod
        def from_hf_pretrained(cls, path):
            cls.path = path
            return cls()

        def to_megatron_provider(self, load_weights=False):
            assert load_weights is False
            return StrictProvider()

    class FakeLoRA:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    def install(name: str, **values):
        module = types.ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    install("megatron", bridge=None)
    bridge = install("megatron.bridge", AutoBridge=FakeAutoBridge)
    install("megatron.bridge.data")
    install(
        "megatron.bridge.data.builders",
        ChatSFTPreprocessingConfig=Component,
        PromptCompletionSFTPreprocessingConfig=Component,
        DirectHFSFTDatasetConfig=Component,
        HFDatasetSourceConfig=Component,
    )
    install("megatron.bridge.training")
    install(
        "megatron.bridge.training.config",
        CheckpointConfig=Component,
        ConfigContainer=FakeConfig,
        LoggerConfig=Component,
        OptimizerConfig=Component,
        SchedulerConfig=Component,
        TrainingConfig=Component,
    )
    install("megatron.bridge.training.tokenizers")
    install(
        "megatron.bridge.training.tokenizers.config",
        TokenizerConfig=Component,
    )
    install("megatron.bridge.peft")
    install("megatron.bridge.peft.lora", LoRA=FakeLoRA)
    install("megatron.core")
    install("megatron.core.transformer")
    install(
        "megatron.core.transformer.enums",
        AttnBackend=types.SimpleNamespace(local="local", auto="auto"),
    )
    install("megatron.bridge.models")
    install(
        "megatron.bridge.models.gpt_provider",
        local_layer_spec=lambda config: ("local", config),
    )

    class FakeTokenizer:
        eos_token = "<eos>"
        eos_token_id = 999

        def apply_chat_template(self, messages, **kwargs):
            return "<prompt>"

        def __call__(self, text, **kwargs):
            if text == "<eos>":
                return {"input_ids": [999]}
            if text == "<prompt>":
                return {"input_ids": [1]}
            if text.startswith("<prompt>"):
                return {"input_ids": [1, 2, 999]}
            if text.endswith("<eos>"):
                return {"input_ids": [2, 999]}
            return {"input_ids": [1, 2, 999]}

    from megatron_lab import config as config_module

    dataset_builds = 0
    original_build_cluster_dataset = config_module._build_cluster_dataset

    def count_dataset_builds(args):
        nonlocal dataset_builds
        dataset_builds += 1
        return original_build_cluster_dataset(args)

    monkeypatch.setattr(
        config_module, "_build_cluster_dataset", count_dataset_builds
    )
    build_cluster_config = config_module.build_cluster_config

    args = Namespace(
        setup="spark-cluster",
        model_dir=model_dir,
        model_id="zai-org/GLM-4.7-Flash",
        model_revision="c" * 40,
        dataset_revision="d" * 40,
        train_data=data_dir / "training.jsonl",
        eval_data=data_dir / "validation.jsonl",
        output_dir=tmp_path / "out",
        topology=ClusterTopology(expert_parallel_size=2),
        max_length=512,
        max_steps=5,
        schedule_steps=10,
        eval_iters=2,
        seed=42,
        finetuning_mode="lora",
        lora_dim=8,
        lora_alpha=16,
        lora_dropout=0.0,
        optimizer="adam",
        distributed_optimizer=True,
        overlap_grad_reduce=False,
        sequence_parallel=False,
        recompute="none",
        checkpoint_mode="sync",
        fully_parallel_save=True,
        fully_parallel_load=True,
        dist_ckpt_optim_fully_reshardable=True,
        save_optimizer=True,
        load_optimizer=True,
        stage="train",
        cluster_tokenizer=FakeTokenizer(),
    )
    cfg, spec = build_cluster_config(args)

    assert FakeAutoBridge.path == str(model_dir)
    assert spec.family == "glm4_moe_lite"
    assert cfg.model.transformer_impl == "transformer_engine"
    assert cfg.peft.target_modules[:2] == ["linear_q_down_proj", "linear_q_up_proj"]
    assert cfg.model.mtp_num_layers == 0
    assert cfg.model.transformer_layer_spec.keywords["use_transformer_engine"] is True
    assert cfg.model.attention_backend == "auto"
    assert cfg.model.overlap_moe_expert_parallel_comm is False
    assert cfg.model.recompute_granularity is None
    assert cfg.optimizer.use_distributed_optimizer is True
    assert cfg.scheduler.max_steps == 10
    assert cfg.checkpoint.save_interval == 5
    assert cfg.checkpoint.dist_ckpt_optim_fully_reshardable is True
    assert cfg.dataset.preprocessing.loss_mode == "completion"
    assert cfg.dataset.preprocessing.add_eos is False
    assert cfg.dataset.pad_to_max_length is False
    args.pad_to_max_length = True
    padded_cfg, _ = build_cluster_config(args)
    assert padded_cfg.dataset.pad_to_max_length is True
    args.save_interval = 3
    custom_cfg, _ = build_cluster_config(args)
    assert custom_cfg.checkpoint.save_interval == 3
    assert dataset_builds == 3


def test_cluster_launcher_passes_explicit_ranks_and_resume_horizon(tmp_path: Path) -> None:
    calls = tmp_path / "calls.jsonl"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        f"open({str(calls)!r}, 'a').write(json.dumps({{'argv': sys.argv[1:], 'env': {{k: os.environ.get(k) for k in ('NNODES', 'NPROC_PER_NODE', 'NODE_RANK', 'MASTER_ADDR')}}}}) + '\\n')\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    output_dir = tmp_path / "output"
    env = {
        **os.environ,
        "PYTHON": str(fake_python),
        "NODE_RANK": "0",
        "NNODES": "2",
        "NPROC_PER_NODE": "1",
        "MASTER_ADDR": "spark1",
        "OUTPUT_DIR": str(output_dir),
        "MODEL_ID": "Qwen/Qwen3-30B-A3B",
        "MODEL_REVISION": "model-rev",
        "DATASET_ID": "dataset/id",
        "DATASET_REVISION": "data-rev",
        "TRANSFORMER_IMPL": "transformer_engine",
        "RESUME_AFTER_TRAIN": "true",
        "RESUME_MAX_STEPS": "10",
    }
    subprocess.run(
        ["bash", "scripts/run_spark_cluster.sh"],
        cwd=Path(__file__).parents[2] / "backends" / "megatron",
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    initial_calls = [json.loads(line) for line in calls.read_text().splitlines()]
    launch_calls = [
        call for call in initial_calls if "torch.distributed.run" in call["argv"]
    ]
    assert len(launch_calls) == 4
    assert all(call["env"]["NODE_RANK"] == "0" for call in launch_calls)
    assert all("--standalone" not in call["argv"] for call in launch_calls)
    assert all(
        call["argv"][call["argv"].index("--transformer-impl") + 1]
        == "transformer_engine"
        for call in launch_calls
    )
    assert all(
        "--dist-ckpt-optim-fully-reshardable" not in call["argv"]
        for call in launch_calls
    )
    assert all("--save-interval" not in call["argv"] for call in launch_calls)
    resume_call = next(
        call for call in launch_calls if "--stage" in call["argv"] and call["argv"][call["argv"].index("--stage") + 1] == "resume"
    )
    max_step_values = [
        resume_call["argv"][index + 1]
        for index, value in enumerate(resume_call["argv"])
        if value == "--max-steps"
    ]
    assert max_step_values[-1] == "10"

    custom_env = {
        **env,
        "OUTPUT_DIR": str(tmp_path / "custom-output"),
        "SAVE_INTERVAL": "4",
        "DIST_CKPT_OPTIM_FULLY_RESHARDABLE": "true",
    }
    source_checkpoint = tmp_path / "source-checkpoint" / "iter_0000004"
    source_checkpoint.mkdir(parents=True)
    custom_env["LOAD_CHECKPOINT"] = str(source_checkpoint)
    subprocess.run(
        ["bash", "scripts/run_spark_cluster.sh"],
        cwd=Path(__file__).parents[2] / "backends" / "megatron",
        env=custom_env,
        check=True,
        capture_output=True,
        text=True,
    )
    all_calls = [json.loads(line) for line in calls.read_text().splitlines()]
    custom_launch_calls = [
        call for call in all_calls
        if "torch.distributed.run" in call["argv"]
        and str(tmp_path / "custom-output") in call["argv"]
    ]
    assert len(custom_launch_calls) == 4
    assert all(
        call["argv"][call["argv"].index("--save-interval") + 1] == "4"
        for call in custom_launch_calls
    )
    assert all(
        "--dist-ckpt-optim-fully-reshardable" in call["argv"]
        for call in custom_launch_calls
    )
    custom_resume = next(
        call for call in custom_launch_calls
        if call["argv"][call["argv"].index("--stage") + 1] == "resume"
    )
    assert custom_resume["argv"][custom_resume["argv"].index("--load-checkpoint") + 1] == str(source_checkpoint)

    invalid = subprocess.run(
        ["bash", "scripts/run_spark_cluster.sh"],
        cwd=Path(__file__).parents[2] / "backends" / "megatron",
        env={**env, "OUTPUT_DIR": str(tmp_path / "invalid-output"), "SAVE_INTERVAL": "0"},
        capture_output=True,
        text=True,
    )
    assert invalid.returncode == 2
    assert "positive integer" in invalid.stderr

    invalid_reshardable = subprocess.run(
        ["bash", "scripts/run_spark_cluster.sh"],
        cwd=Path(__file__).parents[2] / "backends" / "megatron",
        env={
            **env,
            "OUTPUT_DIR": str(tmp_path / "invalid-reshardable-output"),
            "DIST_CKPT_OPTIM_FULLY_RESHARDABLE": "maybe",
        },
        capture_output=True,
        text=True,
    )
    assert invalid_reshardable.returncode == 2
    assert "must be true or false" in invalid_reshardable.stderr

    invalid_reshardable_save = subprocess.run(
        ["bash", "scripts/run_spark_cluster.sh"],
        cwd=Path(__file__).parents[2] / "backends" / "megatron",
        env={
            **env,
            "OUTPUT_DIR": str(tmp_path / "invalid-reshardable-save-output"),
            "DIST_CKPT_OPTIM_FULLY_RESHARDABLE": "true",
            "SAVE_OPTIMIZER": "false",
        },
        capture_output=True,
        text=True,
    )
    assert invalid_reshardable_save.returncode == 2
    assert "requires SAVE_OPTIMIZER=true" in invalid_reshardable_save.stderr

    missing_checkpoint = subprocess.run(
        ["bash", "scripts/run_spark_cluster.sh"],
        cwd=Path(__file__).parents[2] / "backends" / "megatron",
        env={**env, "OUTPUT_DIR": str(tmp_path / "missing-output"), "LOAD_CHECKPOINT": str(tmp_path / "no-such-checkpoint")},
        capture_output=True,
        text=True,
    )
    assert missing_checkpoint.returncode == 2
    assert "checkpoint directory" in missing_checkpoint.stderr
