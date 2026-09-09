import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import types

import pytest

import trl_lab.spark_train as spark_train


def test_process_group_cleanup_runs_on_normal_and_exception_exit(monkeypatch) -> None:
    calls = []

    class FakeDistributed:
        def is_available(self):
            return True

        def is_initialized(self):
            return True

        def destroy_process_group(self):
            calls.append("destroy")

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(distributed=FakeDistributed()))
    monkeypatch.delenv("RANK_LOG_DIR", raising=False)
    args = argparse.Namespace(stage="train")

    with spark_train._write_log(args):
        pass
    assert calls == ["destroy"]

    with pytest.raises(RuntimeError, match="sentinel"):
        with spark_train._write_log(args):
            raise RuntimeError("sentinel")
    assert calls == ["destroy", "destroy"]


def test_process_group_cleanup_is_noop_without_initialized_group(monkeypatch) -> None:
    class FakeDistributed:
        def is_available(self):
            return True

        def is_initialized(self):
            return False

        def destroy_process_group(self):
            raise AssertionError("must not destroy an uninitialized group")

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(distributed=FakeDistributed()))
    spark_train._destroy_process_group_if_initialized()


def test_rank_log_preserves_stdout_and_stderr_file_descriptors(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RANK_LOG_DIR", str(tmp_path))
    args = argparse.Namespace(stage="train")

    with spark_train._write_log(args):
        assert sys.stdout.fileno() == sys.__stdout__.fileno()
        assert sys.stderr.fileno() == sys.__stderr__.fileno()


class FakeTensor:
    dtype = "torch.bfloat16"
    requires_grad = True

    def __init__(self, value=0.0):
        self.value = value

    def numel(self):
        return 1

    def element_size(self):
        return 2

    def detach(self):
        return self

    def reshape(self, *_):
        return self

    def __getitem__(self, _):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def clone(self):
        return FakeTensor(self.value)

    def __sub__(self, other):
        return FakeTensor(self.value - other.value)

    def abs(self):
        return FakeTensor(abs(self.value))

    def max(self):
        return self

    def item(self):
        return self.value


class FakeModel:
    def __init__(self):
        self.parameter = FakeTensor()
        self.config = types.SimpleNamespace(use_cache=True)
        self.checkpointing_kwargs = None

    def parameters(self):
        return [self.parameter]

    def named_parameters(self):
        return [("layers.0.lora_B.weight", self.parameter)]

    def to(self, _):
        raise AssertionError("Trainer/Accelerate must own device placement")

    def gradient_checkpointing_enable(self, **kwargs):
        self.checkpointing_kwargs = kwargs

    def requires_grad_(self, value):
        self.parameter.requires_grad = value


def test_deepspeed_model_loads_safetensor_shards_one_at_a_time(monkeypatch, tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "first.safetensors", "b": "second.safetensors"}}),
        encoding="utf-8",
    )
    loaded = []
    model = FakeModel()
    fake_transformers = types.SimpleNamespace(
        AutoConfig=types.SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: object()),
        AutoModelForCausalLM=types.SimpleNamespace(from_config=lambda *_args, **_kwargs: model),
    )
    safetensors = types.ModuleType("safetensors")
    safetensors_torch = types.ModuleType("safetensors.torch")
    safetensors_torch.load_file = lambda path, **_kwargs: {Path(path).name: FakeTensor()}
    integrations = types.ModuleType("transformers.integrations")
    deepspeed = types.ModuleType("transformers.integrations.deepspeed")
    deepspeed._load_state_dict_into_zero3_model = lambda _model, state: (loaded.append(next(iter(state))) or [], set())
    monkeypatch.setitem(sys.modules, "safetensors", safetensors)
    monkeypatch.setitem(sys.modules, "safetensors.torch", safetensors_torch)
    monkeypatch.setitem(sys.modules, "transformers.integrations", integrations)
    monkeypatch.setitem(sys.modules, "transformers.integrations.deepspeed", deepspeed)
    config = types.SimpleNamespace(
        distributed_backend="deepspeed", model_dir=model_dir, output_dir=tmp_path / "out",
        finetuning_mode="full", model_revision="a" * 40,
    )

    assert spark_train._load_model(config, types.SimpleNamespace(bfloat16="bf16"), fake_transformers, object()) is model
    assert loaded == ["first.safetensors", "second.safetensors"]


def test_deepspeed_tuned_full_model_loads_as_skeleton_without_state_dict(tmp_path: Path) -> None:
    model = FakeModel()
    config_calls = []
    fake_transformers = types.SimpleNamespace(
        AutoConfig=types.SimpleNamespace(from_pretrained=lambda path, **_kwargs: config_calls.append(path) or object()),
        AutoModelForCausalLM=types.SimpleNamespace(from_config=lambda *_args, **_kwargs: model),
    )
    config = types.SimpleNamespace(
        distributed_backend="deepspeed", model_dir=tmp_path / "model", output_dir=tmp_path / "out",
        finetuning_mode="full", model_revision="a" * 40,
    )

    loaded_model = spark_train._load_model(
        config, types.SimpleNamespace(bfloat16="bf16"), fake_transformers, object(), load_tuned=True,
    )

    assert loaded_model is model
    assert config_calls == [str(tmp_path / "model")]


def test_deepspeed_nvme_profile_is_detected(tmp_path: Path) -> None:
    profile = tmp_path / "deepspeed.json"
    profile.write_text(json.dumps({"zero_optimization": {"offload_param": {"device": "nvme"}}}))
    config = types.SimpleNamespace(distributed_backend="deepspeed", deepspeed_config=profile)

    assert spark_train._uses_deepspeed_nvme(config)


def test_parameter_sampling_keeps_representative_update_evidence() -> None:
    parameter = FakeTensor()
    selected = spark_train._sample_trainable_parameters([
        ("layers.0.lora_B.weight", parameter),
        ("layers.1.lora_B.weight", parameter),
        ("model.norm.weight", parameter),
        ("lm_head.weight", parameter),
    ], limit=4)
    assert [name for name, _ in selected] == [
        "layers.0.lora_B.weight", "model.norm.weight", "lm_head.weight", "layers.1.lora_B.weight",
    ]


def test_main_passes_cli_optimizer_and_sft_config_to_trainer(monkeypatch, tmp_path: Path) -> None:
    captured = {}
    args = argparse.Namespace(
        model_id="Qwen/Qwen3-30B-A3B", model_dir=tmp_path / "model", model_revision="a" * 40,
        dataset_id="public/data", dataset_revision="b" * 40, data_dir=tmp_path / "data", output_dir=tmp_path / "out",
        stage="train", finetuning_mode="full", optimizer="adamw", learning_rate=0.0003,
        max_steps=1, epochs=None, max_length=32, gradient_accumulation_steps=2, seed=42,
        distributed_backend="ddp", deepspeed_config=None,
        train_samples=None, eval_samples=None, lora_r=8, lora_alpha=16,
    )
    model = FakeModel()

    class FakeOptimizer:
        def __init__(self, parameters, **kwargs):
            self.parameters = parameters
            self.kwargs = kwargs
            self.state = {}

    class FakeSFTConfig:
        def __init__(self, **kwargs):
            captured["sft_config"] = kwargs
            self.gradient_checkpointing = kwargs["gradient_checkpointing"]

    class FakeTrainer:
        def __init__(self, **kwargs):
            captured["trainer"] = kwargs
            self.model = kwargs["model"]
            self.optimizer = FakeOptimizer(self.model.parameters(), lr=args.learning_rate)

        def train(self):
            captured.setdefault("calls", []).append("train")
            self.model.parameter.value += 1.0
            self.optimizer.state = {self.model.parameter: {"exp_avg": FakeTensor(), "exp_avg_sq": FakeTensor()}}
            return types.SimpleNamespace(metrics={"train_loss": 1.0}, global_step=1)

        def evaluate(self):
            captured.setdefault("calls", []).append("evaluate")
            return {"eval_loss": 1.0}

        def save_model(self, _):
            captured.setdefault("calls", []).append("save")
            return None

    def fake_load_dataset(*args, **kwargs):
        captured["load_dataset"] = {"args": args, "kwargs": kwargs}
        return {"train": [{"prompt": "p", "completion": "c"}], "validation": [{"prompt": "p", "completion": "c"}]}

    fake_torch = types.SimpleNamespace(
        bfloat16="bf16",
        manual_seed=lambda _: None,
        device=lambda *_: "cuda:0",
        is_tensor=lambda value: isinstance(value, FakeTensor),
        optim=types.SimpleNamespace(AdamW=FakeOptimizer, SGD=FakeOptimizer),
        cuda=types.SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True, device_count=lambda: 1, set_device=lambda _: None, max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0),
    )
    fake_peft = types.SimpleNamespace(LoraConfig=object, get_peft_model=lambda model, _: model, PeftModel=types.SimpleNamespace(from_pretrained=lambda model, *_args, **_kwargs: model))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "trl", types.SimpleNamespace(SFTConfig=FakeSFTConfig, SFTTrainer=FakeTrainer))
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset))
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    monkeypatch.setattr(spark_train, "parse_args", lambda: args)
    monkeypatch.setattr(spark_train, "validate_config", lambda _: "qwen3_moe")
    monkeypatch.setattr(spark_train, "_write_log", lambda _: nullcontext())
    monkeypatch.setattr(spark_train, "_load_tokenizer", lambda *_: types.SimpleNamespace(pad_token="<pad>", eos_token="<eos>"))
    def load_model(*_args, **_kwargs):
        assert "sft_config" in captured
        return model

    monkeypatch.setattr(spark_train, "_load_model", load_model)
    monkeypatch.setattr(spark_train, "model_snapshot_evidence", lambda *_: {})
    import trl_lab.spark_data as spark_data
    monkeypatch.setattr(
        spark_data,
        "prepare_prompt_completion_data",
        lambda *_args: (
            {"train": tmp_path / "training.jsonl", "validation": tmp_path / "validation.jsonl"},
            {},
        ),
    )

    spark_train.main()

    assert "optimizers" not in captured["trainer"]
    assert captured["sft_config"]["learning_rate"] == args.learning_rate
    assert captured["sft_config"]["optim"] == "adamw_torch"
    assert captured["sft_config"]["ddp_find_unused_parameters"] is False
    assert captured["sft_config"]["gradient_checkpointing"] is True
    assert captured["sft_config"]["gradient_checkpointing_kwargs"] == {"use_reentrant": False}
    assert captured["sft_config"]["save_strategy"] == "no"
    assert captured["calls"] == ["train", "evaluate", "save"]
    assert captured["load_dataset"]["kwargs"]["cache_dir"].endswith("hf-cache")
    assert model.checkpointing_kwargs == {"gradient_checkpointing_kwargs": {"use_reentrant": False}}
    summary = json.loads((args.output_dir / "summary-train.json").read_text(encoding="utf-8"))
    assert summary["memory_components_scope"] == "replicated model view"


def test_tuned_stage_reloads_deepspeed_checkpoint_before_evaluate(monkeypatch, tmp_path: Path) -> None:
    captured = {}
    args = argparse.Namespace(
        model_id="Qwen/Qwen3-30B-A3B", model_dir=tmp_path / "model", model_revision="a" * 40,
        dataset_id="public/data", dataset_revision="b" * 40, data_dir=tmp_path / "data", output_dir=tmp_path / "out",
        stage="tuned", finetuning_mode="full", optimizer="adamw", learning_rate=0.0003,
        max_steps=1, epochs=None, max_length=32, gradient_accumulation_steps=2, seed=42,
        distributed_backend="deepspeed", deepspeed_config=tmp_path / "ds.json",
        train_samples=None, eval_samples=None, lora_r=8, lora_alpha=16,
    )
    model = FakeModel()

    class FakeOptimizer:
        def __init__(self, parameters, **kwargs):
            self.parameters = parameters
            self.kwargs = kwargs
            self.state = {}

    class FakeSFTConfig:
        def __init__(self, **kwargs):
            captured["sft_config"] = kwargs
            self.gradient_checkpointing = kwargs["gradient_checkpointing"]

    class FakeTrainer:
        def __init__(self, **kwargs):
            captured["trainer"] = kwargs
            self.model = kwargs["model"]
            self.model_wrapped = self.model
            self.optimizer = FakeOptimizer(self.model.parameters(), lr=args.learning_rate)

        def get_train_dataloader(self):
            captured.setdefault("calls", []).append("get_train_dataloader")
            return "fake-dataloader"

        def _prepare_for_training(self, max_steps, train_dataloader, resume_from_checkpoint):
            captured.setdefault("calls", []).append("prepare_for_training")
            captured["prepare_kwargs"] = {
                "max_steps": max_steps, "train_dataloader": train_dataloader,
                "resume_from_checkpoint": resume_from_checkpoint,
            }

        def evaluate(self):
            captured.setdefault("calls", []).append("evaluate")
            return {"eval_loss": 1.0}

    def fake_load_dataset(*args, **kwargs):
        captured["load_dataset"] = {"args": args, "kwargs": kwargs}
        return {"train": [{"prompt": "p", "completion": "c"}], "validation": [{"prompt": "p", "completion": "c"}]}

    fake_torch = types.SimpleNamespace(
        bfloat16="bf16",
        manual_seed=lambda _: None,
        device=lambda *_: "cuda:0",
        is_tensor=lambda value: isinstance(value, FakeTensor),
        optim=types.SimpleNamespace(AdamW=FakeOptimizer, SGD=FakeOptimizer),
        cuda=types.SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True, device_count=lambda: 1, set_device=lambda _: None, max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0),
    )
    fake_peft = types.SimpleNamespace(LoraConfig=object, get_peft_model=lambda model, _: model, PeftModel=types.SimpleNamespace(from_pretrained=lambda model, *_args, **_kwargs: model))

    def fake_deepspeed_load_checkpoint(model_wrapped, checkpoint_dir, load_module_strict=True):
        captured.setdefault("calls", []).append("deepspeed_load_checkpoint")
        captured["load_checkpoint_args"] = {
            "model_wrapped": model_wrapped, "checkpoint_dir": checkpoint_dir, "load_module_strict": load_module_strict,
        }

    integrations = types.ModuleType("transformers.integrations")
    deepspeed_integration = types.ModuleType("transformers.integrations.deepspeed")
    deepspeed_integration.deepspeed_load_checkpoint = fake_deepspeed_load_checkpoint
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "transformers.integrations", integrations)
    monkeypatch.setitem(sys.modules, "transformers.integrations.deepspeed", deepspeed_integration)
    monkeypatch.setitem(sys.modules, "trl", types.SimpleNamespace(SFTConfig=FakeSFTConfig, SFTTrainer=FakeTrainer))
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset))
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    monkeypatch.setattr(spark_train, "parse_args", lambda: args)
    monkeypatch.setattr(spark_train, "validate_config", lambda _: "qwen3_moe")
    monkeypatch.setattr(spark_train, "_write_log", lambda _: nullcontext())
    monkeypatch.setattr(spark_train, "_load_tokenizer", lambda *_: types.SimpleNamespace(pad_token="<pad>", eos_token="<eos>"))
    monkeypatch.setattr(spark_train, "_load_model", lambda *_args, **_kwargs: model)
    monkeypatch.setattr(spark_train, "model_snapshot_evidence", lambda *_: {})
    import trl_lab.spark_data as spark_data
    monkeypatch.setattr(
        spark_data,
        "prepare_prompt_completion_data",
        lambda *_args: (
            {"train": tmp_path / "training.jsonl", "validation": tmp_path / "validation.jsonl"},
            {},
        ),
    )

    (args.output_dir / "model" / "global_step1").mkdir(parents=True)

    spark_train.main()

    assert captured["calls"] == ["get_train_dataloader", "prepare_for_training", "deepspeed_load_checkpoint", "evaluate"]
    assert captured["prepare_kwargs"] == {
        "max_steps": 1, "train_dataloader": "fake-dataloader", "resume_from_checkpoint": None,
    }
    assert captured["load_checkpoint_args"] == {
        "model_wrapped": model, "checkpoint_dir": str(args.output_dir / "model"), "load_module_strict": True,
    }
    assert (args.output_dir / "model" / "latest").read_text(encoding="utf-8") == "global_step1"
    summary = json.loads((args.output_dir / "summary-tuned.json").read_text(encoding="utf-8"))
    assert summary["evaluation"] == {"eval_loss": 1.0}


def test_sharded_backend_training_arguments_are_explicit(tmp_path: Path) -> None:
    base = argparse.Namespace(
        max_steps=3,
        epochs=None,
        learning_rate=2e-5,
        gradient_accumulation_steps=4,
        optimizer="adamw",
        seed=42,
    )
    config = types.SimpleNamespace(
        output_dir=tmp_path,
        max_length=512,
        distributed_backend="fsdp2",
        deepspeed_config=None,
    )
    fsdp = spark_train._sft_config_kwargs(config, base)
    assert fsdp["fsdp"] is True
    assert fsdp["fsdp_config"] == {
        "version": 2,
        "reshard_after_forward": True,
        "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
        "activation_checkpointing": True,
        "cpu_ram_efficient_loading": True,
        "state_dict_type": "SHARDED_STATE_DICT",
    }
    assert fsdp["gradient_checkpointing"] is False

    profile = tmp_path / "zero3.json"
    profile.write_text('{"zero_optimization": {}}', encoding="utf-8")
    config.distributed_backend = "deepspeed"
    config.deepspeed_config = profile
    deepspeed = spark_train._sft_config_kwargs(config, base)
    assert deepspeed["deepspeed"] == str(profile)
    assert deepspeed["gradient_checkpointing"] is True
    assert "fsdp" not in deepspeed


def test_sft_config_kwargs_epochs_sets_max_steps_sentinel(tmp_path: Path) -> None:
    args = argparse.Namespace(
        max_steps=None, epochs=2.5, learning_rate=2e-5, gradient_accumulation_steps=4,
        optimizer="adamw", seed=42,
    )
    config = types.SimpleNamespace(
        output_dir=tmp_path, max_length=512, distributed_backend="ddp", deepspeed_config=None,
    )

    kwargs = spark_train._sft_config_kwargs(config, args)

    assert kwargs["max_steps"] == -1
    assert kwargs["num_train_epochs"] == 2.5


def test_sft_config_kwargs_steps_mode_ignores_epochs_field(tmp_path: Path) -> None:
    args = argparse.Namespace(
        max_steps=7, epochs=None, learning_rate=2e-5, gradient_accumulation_steps=4,
        optimizer="adamw", seed=42,
    )
    config = types.SimpleNamespace(
        output_dir=tmp_path, max_length=512, distributed_backend="ddp", deepspeed_config=None,
    )

    kwargs = spark_train._sft_config_kwargs(config, args)

    assert kwargs["max_steps"] == 7
    assert kwargs["num_train_epochs"] == 1


def test_parse_args_rejects_epochs_and_max_steps_together(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", [
        "spark_train", "--model-id", "m", "--model-dir", "/m", "--model-revision", "a" * 40,
        "--dataset-id", "d", "--dataset-revision", "b" * 40, "--data-dir", "/d", "--output-dir", "/o",
        "--stage", "train", "--max-steps", "3", "--epochs", "1",
    ])

    with pytest.raises(SystemExit):
        spark_train.parse_args()


def test_parse_args_defaults_to_five_steps_when_neither_given(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", [
        "spark_train", "--model-id", "m", "--model-dir", "/m", "--model-revision", "a" * 40,
        "--dataset-id", "d", "--dataset-revision", "b" * 40, "--data-dir", "/d", "--output-dir", "/o",
        "--stage", "train",
    ])

    args = spark_train.parse_args()

    assert args.max_steps == 5
    assert args.epochs is None


def test_parse_args_accepts_epochs_alone(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", [
        "spark_train", "--model-id", "m", "--model-dir", "/m", "--model-revision", "a" * 40,
        "--dataset-id", "d", "--dataset-revision", "b" * 40, "--data-dir", "/d", "--output-dir", "/o",
        "--stage", "train", "--epochs", "3",
    ])

    args = spark_train.parse_args()

    assert args.max_steps is None
    assert args.epochs == 3.0
