import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import types

import pytest

import sft_lab.spark_train as spark_train


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
        max_steps=1, max_length=32, gradient_accumulation_steps=2, seed=42,
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

    class FakeTrainer:
        def __init__(self, **kwargs):
            captured["trainer"] = kwargs
            self.model = kwargs["model"]
            self.optimizer = FakeOptimizer(self.model.parameters(), lr=args.learning_rate)

        def train(self):
            self.model.parameter.value += 1.0
            self.optimizer.state = {self.model.parameter: {"exp_avg": FakeTensor(), "exp_avg_sq": FakeTensor()}}
            return types.SimpleNamespace(metrics={"train_loss": 1.0}, global_step=1)

        def evaluate(self):
            return {"eval_loss": 1.0}

        def save_model(self, _):
            return None

    class FakeDataset:
        @classmethod
        def from_list(cls, rows):
            return rows

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
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(Dataset=FakeDataset))
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
    import sft_lab.spark_data as spark_data
    monkeypatch.setattr(spark_data, "load_prompt_completion_data", lambda *_args: ([{"prompt": "p", "completion": "c"}], [{"prompt": "p", "completion": "c"}], {}))

    spark_train.main()

    assert "optimizers" not in captured["trainer"]
    assert captured["sft_config"]["learning_rate"] == args.learning_rate
    assert captured["sft_config"]["optim"] == "adamw_torch"
    assert captured["sft_config"]["ddp_find_unused_parameters"] is False
    assert captured["sft_config"]["gradient_checkpointing"] is True
    assert captured["sft_config"]["gradient_checkpointing_kwargs"] == {"use_reentrant": False}
    assert model.checkpointing_kwargs == {"gradient_checkpointing_kwargs": {"use_reentrant": False}}
    summary = json.loads((args.output_dir / "summary-train.json").read_text(encoding="utf-8"))
    assert summary["memory_components_scope"] == "replicated model view"


def test_sharded_backend_training_arguments_are_explicit(tmp_path: Path) -> None:
    base = argparse.Namespace(
        max_steps=3,
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
    config.distributed_backend = "deepspeed"
    config.deepspeed_config = profile
    deepspeed = spark_train._sft_config_kwargs(config, base)
    assert deepspeed["deepspeed"] == str(profile)
    assert deepspeed["gradient_checkpointing"] is True
    assert "fsdp" not in deepspeed
