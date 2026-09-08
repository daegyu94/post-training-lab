"""Build the Spark-cluster Megatron Bridge configuration."""

from __future__ import annotations

from argparse import Namespace
import json
import os
from dataclasses import dataclass
from pathlib import Path


DATASET_ID = "HuggingFaceH4/ultrachat_200k"
QWEN3_30B_MODEL_ID = "Qwen/Qwen3-30B-A3B"
GLM47_FLASH_MODEL_ID = "zai-org/GLM-4.7-Flash"


@dataclass(frozen=True)
class ModelSpec:
    """Model architecture information read from a local HF config.json."""

    model_type: str
    family: str
    model_id: str | None


MODEL_TYPES = {
    "qwen2": "qwen2",
    "qwen3_moe": "qwen3_moe",
    "glm4_moe_lite": "glm4_moe_lite",
}
TRANSFORMER_IMPL_CHOICES = ("auto", "local", "transformer_engine")


def detect_model_config(model_dir: Path) -> ModelSpec:
    """Detect a supported architecture without inferring it from a model name."""

    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise ValueError(f"model config is missing: {config_path}")
    try:
        model_config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid model config: {config_path}") from exc
    model_type = str(model_config.get("model_type", ""))
    family = MODEL_TYPES.get(model_type)
    if family is None:
        architectures = ", ".join(model_config.get("architectures", []))
        raise ValueError(
            "unsupported model_type; expected qwen2, qwen3_moe or glm4_moe_lite, "
            f"found {model_type!r} (architectures={architectures or 'unknown'})"
        )
    model_id = model_config.get("_name_or_path")
    return ModelSpec(model_type=model_type, family=family, model_id=model_id)


def validate_model_identity(model_id: str, spec: ModelSpec) -> None:
    """Reject a command whose advertised model disagrees with config.json."""

    expected = {
        QWEN3_30B_MODEL_ID: "qwen3_moe",
        GLM47_FLASH_MODEL_ID: "glm4_moe_lite",
    }.get(model_id)
    if expected and spec.family != expected:
        raise ValueError(
            f"model id {model_id!r} disagrees with config model_type {spec.model_type!r}"
        )


def select_transformer_impl(
    spec: ModelSpec,
    requested: str = "auto",
    *,
    sequence_parallel: bool = False,
) -> str:
    """Resolve a backend without silently changing a setup2 A/B variant."""

    if requested not in TRANSFORMER_IMPL_CHOICES:
        choices = ", ".join(TRANSFORMER_IMPL_CHOICES)
        raise ValueError(f"transformer_impl must be one of {choices}")
    family = getattr(spec, "family", MODEL_TYPES.get(getattr(spec, "model_type", "")))
    if family == "glm4_moe_lite" and requested == "local":
        raise ValueError(
            "GLM-4.7-Flash requires transformer_engine; --transformer-impl local "
            "is unsupported for its MLA provider"
        )
    selected = (
        "transformer_engine"
        if requested == "auto" and family == "glm4_moe_lite"
        else "local"
        if requested == "auto"
        else requested
    )
    if sequence_parallel and selected == "local":
        raise ValueError(
            "sequence parallel requires --transformer-impl transformer_engine; "
            "the local backend does not support this setup2 path"
        )
    return selected


def validate_reshardable_checkpoint_options(args: Namespace) -> None:
    """Validate the optimizer checkpoint format and stage requirements."""

    if not getattr(args, "dist_ckpt_optim_fully_reshardable", False):
        return
    if not args.distributed_optimizer:
        raise ValueError(
            "fully reshardable optimizer checkpoints require distributed_optimizer"
        )
    if args.stage == "train" and not args.save_optimizer:
        raise ValueError(
            "fully reshardable checkpoint creation requires save_optimizer during train"
        )


def validate_dataset_manifest(train_data: Path, dataset_revision: str, dataset_id: str = DATASET_ID) -> dict[str, object]:
    """Require the prepared split manifest to match the requested revision."""

    from megatron_lab.model_cache import require_immutable_revision

    require_immutable_revision(dataset_revision, name="dataset revision")

    manifest_path = train_data.parent / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"dataset manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != dataset_id:
        raise ValueError("dataset manifest has an unexpected dataset")
    if manifest.get("dataset_revision") != dataset_revision:
        raise ValueError("dataset revision does not match manifest")
    return manifest


def _set_model_option(model: object, name: str, value: object) -> None:
    """Set an explicit Bridge provider option while keeping test doubles simple."""

    if not hasattr(model, name):
        raise AttributeError(f"Bridge provider lacks required option: {name}")
    setattr(model, name, value)


def _set_optional_model_option(model: object, name: str, value: object) -> None:
    """Apply a known provider option only when this architecture exposes it."""

    if hasattr(model, name):
        setattr(model, name, value)


def _resolve_cluster_load_checkpoint(args: Namespace, checkpoint_dir: Path) -> Path | None:
    """Resolve an immutable source checkpoint for setup2 load-only stages."""

    requested = getattr(args, "load_checkpoint", None)
    if requested is not None and args.stage not in {"resume", "tuned"}:
        raise ValueError("--load-checkpoint is valid only for setup2 resume or tuned stages")
    if args.stage not in {"resume", "tuned"}:
        return None
    source = Path(requested) if requested is not None else checkpoint_dir
    if not source.is_dir():
        raise ValueError(f"checkpoint source directory does not exist: {source}")
    return source


def _set_transformer_layer_spec(
    model: object, spec: ModelSpec, implementation: str = "auto"
) -> None:
    """Pair each architecture with the selected provider layer specification."""

    implementation = select_transformer_impl(spec, implementation)

    if implementation == "transformer_engine" and spec.family == "glm4_moe_lite":
        from functools import partial

        layer_spec = getattr(model, "transformer_layer_spec", None)
        if not isinstance(layer_spec, partial) or "use_transformer_engine" not in (layer_spec.keywords or {}):
            raise ValueError(
                "GLM provider did not expose the expected transformer layer spec partial"
            )
        keywords = dict(layer_spec.keywords or {})
        if not layer_spec.keywords.get("use_transformer_engine"):
            raise RuntimeError(
                "GLM-4.7-Flash setup2 requires a Transformer Engine layer spec; "
                "the local MLA attention spec is unsupported"
            )
        keywords["use_transformer_engine"] = True
        _set_model_option(
            model,
            "transformer_layer_spec",
            partial(layer_spec.func, *layer_spec.args, **keywords),
        )
        return
    if implementation == "transformer_engine":
        from megatron.bridge.models.gpt_provider import transformer_engine_layer_spec

        layer_spec = transformer_engine_layer_spec
    else:
        from megatron.bridge.models.gpt_provider import local_layer_spec

        layer_spec = local_layer_spec
    _set_model_option(model, "transformer_layer_spec", layer_spec)


def _build_cluster_dataset(args: Namespace):
    from megatron.bridge.data.builders import (
        DirectHFSFTDatasetConfig,
        HFDatasetSourceConfig,
        PromptCompletionSFTPreprocessingConfig,
    )

    from megatron_lab.cluster_data import prepare_cluster_data

    tokenizer = getattr(args, "cluster_tokenizer", None)
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(args.model_dir), local_files_only=True
        )
    prepared_dir = (
        Path(args.output_dir)
        / f"prepared-cluster-data-rank-{os.environ.get('RANK', '0')}"
    )
    prepare_cluster_data(
        Path(args.train_data),
        Path(args.eval_data),
        prepared_dir,
        tokenizer,
        args.max_length,
        dataset_id=getattr(args, "dataset_id", DATASET_ID),
        dataset_revision=args.dataset_revision,
        model_revision=args.model_revision,
    )
    # The completion already contains EOS.  Do not add another EOS or infer a
    # provider-specific assistant boundary (GLM-4.7-Flash has no such rule).
    preprocessing = PromptCompletionSFTPreprocessingConfig(
        strip_whitespace=False,
        add_eos=False,
        loss_mode="completion",
    )

    return DirectHFSFTDatasetConfig(
        seq_length=args.max_length,
        preprocessing=preprocessing,
        hf_processor_path=str(args.model_dir),
        source=HFDatasetSourceConfig(
            path_or_dataset="json",
            split="train",
            load_kwargs={"data_files": {"train": str(prepared_dir / "training.jsonl")}},
        ),
        validation_source=HFDatasetSourceConfig(
            path_or_dataset="json",
            split="validation",
            load_kwargs={"data_files": {"validation": str(prepared_dir / "validation.jsonl")}},
        ),
        do_validation=True,
        do_test=False,
        dataloader_type="cyclic",
        num_workers=0,
        pin_memory=True,
    )


def _cluster_skeleton(spec: ModelSpec, args: Namespace):
    """Build a local-only ConfigContainer around the detected HF provider."""

    from megatron.bridge import AutoBridge
    from megatron.bridge.training.config import (
        CheckpointConfig,
        ConfigContainer,
        LoggerConfig,
        OptimizerConfig,
        SchedulerConfig,
        TrainingConfig,
    )
    from megatron.bridge.training.tokenizers.config import TokenizerConfig

    provider = AutoBridge.from_hf_pretrained(
        str(args.model_dir)
    ).to_megatron_provider(load_weights=False)
    schedule_steps = args.schedule_steps or args.max_steps
    cfg = ConfigContainer(
        model=provider,
        train=TrainingConfig(
            train_iters=args.max_steps,
            global_batch_size=args.topology.global_batch_size,
            micro_batch_size=args.topology.micro_batch_size,
        ),
        optimizer=OptimizerConfig(
            optimizer=args.optimizer,
            lr=1.0e-4 if args.finetuning_mode == "lora" else 5.0e-6,
            min_lr=0.0,
            use_distributed_optimizer=args.distributed_optimizer,
        ),
        scheduler=SchedulerConfig(
            start_weight_decay=0.01,
            end_weight_decay=0.01,
            weight_decay_incr_style="constant",
            lr_warmup_iters=1 if schedule_steps > 1 else 0,
            lr_decay_iters=schedule_steps,
            max_steps=schedule_steps,
        ),
        dataset=_build_cluster_dataset(args),
        tokenizer=TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=str(args.model_dir),
        ),
        checkpoint=CheckpointConfig(
            pretrained_checkpoint=str(args.model_dir),
            save_interval=args.max_steps,
            ckpt_format="torch_dist",
            dist_ckpt_optim_fully_reshardable=getattr(
                args, "dist_ckpt_optim_fully_reshardable", False
            ),
        ),
        logger=LoggerConfig(
            log_interval=1,
            tensorboard_dir=str(args.output_dir / "tensorboard"),
        ),
        mixed_precision="bf16_mixed",
    )
    return cfg


def build_cluster_config(args: Namespace):
    """Build the setup2 config for Qwen3-MoE or GLM4-MoE-Lite."""

    spec = detect_model_config(Path(args.model_dir))
    validate_model_identity(args.model_id, spec)
    validate_reshardable_checkpoint_options(args)
    implementation = select_transformer_impl(
        spec,
        getattr(args, "transformer_impl", "auto"),
        sequence_parallel=args.sequence_parallel,
    )
    validate_dataset_manifest(
        Path(args.train_data), args.dataset_revision, getattr(args, "dataset_id", DATASET_ID)
    )
    _resolve_cluster_load_checkpoint(
        args, Path(args.output_dir) / "checkpoints"
    )
    cfg = _cluster_skeleton(spec, args)
    model = cfg.model
    topology = args.topology
    if spec.family == "qwen2" and topology.expert_parallel_size != 1:
        raise ValueError("dense feature models require EP=1")

    _set_model_option(model, "tensor_model_parallel_size", topology.tensor_parallel_size)
    _set_model_option(model, "pipeline_model_parallel_size", topology.pipeline_parallel_size)
    if hasattr(model, "expert_model_parallel_size"):
        _set_model_option(
            model,
            "expert_model_parallel_size",
            topology.expert_parallel_size if spec.family != "qwen2" else 1,
        )
    if hasattr(model, "expert_tensor_parallel_size"):
        _set_model_option(
            model, "expert_tensor_parallel_size", topology.expert_tensor_parallel_size
        )
    _set_model_option(model, "context_parallel_size", 1)
    _set_model_option(model, "sequence_parallel", args.sequence_parallel)
    _set_model_option(model, "seq_length", args.max_length)
    _set_model_option(model, "transformer_impl", implementation)
    _set_transformer_layer_spec(model, spec, implementation)
    # Native Spark runtime does not provide Apex's fused weight-gradient kernel.
    _set_optional_model_option(model, "gradient_accumulation_fusion", False)
    _set_model_option(model, "bf16", True)
    _set_model_option(model, "cross_entropy_loss_fusion", False)
    _set_optional_model_option(model, "cross_entropy_fusion_impl", "native")
    from megatron.core.transformer.enums import AttnBackend

    attention_backend = (
        AttnBackend.auto
        if implementation == "transformer_engine"
        else AttnBackend.local
    )
    _set_optional_model_option(model, "attention_backend", attention_backend)
    for option in (
        "masked_softmax_fusion",
        "apply_rope_fusion",
        "persist_layer_norm",
        "bias_activation_fusion",
        "bias_dropout_fusion",
        "moe_router_fusion",
    ):
        _set_optional_model_option(model, option, False)
    _set_optional_model_option(model, "mtp_num_layers", 0)
    _set_model_option(model, "cuda_graph_impl", "none")
    if spec.family != "qwen2":
        _set_model_option(model, "moe_token_dispatcher_type", "alltoall")
        # Core 0.19 names expert A2A overlap
        # ``overlap_moe_expert_parallel_comm``.  Keep the baseline explicit;
        # ``moe_a2a_overlap`` is a performance-recipe argument, not a provider
        # field (notably Qwen3 does not expose it).
        _set_optional_model_option(
            model, "overlap_moe_expert_parallel_comm", False
        )
        for option in (
            "moe_flex_dispatcher_backend",
            "moe_shared_expert_overlap",
            "moe_grouped_gemm",
            "moe_permute_fusion",
        ):
            _set_optional_model_option(
                model,
                option,
                None if option == "moe_flex_dispatcher_backend" else False,
            )
    recompute = None if args.recompute == "none" else args.recompute
    _set_model_option(model, "recompute_granularity", recompute)
    _set_model_option(
        model,
        "recompute_method",
        "uniform" if recompute == "full" else None,
    )
    _set_model_option(model, "recompute_num_layers", 1 if recompute else None)

    cfg.tokenizer.tokenizer_model = str(args.model_dir)
    cfg.dataset = _build_cluster_dataset(args)
    cfg.train.train_iters = args.max_steps
    cfg.train.global_batch_size = topology.global_batch_size
    cfg.train.micro_batch_size = topology.micro_batch_size
    cfg.validation.eval_interval = args.max_steps
    cfg.validation.eval_iters = args.eval_iters
    schedule_steps = args.schedule_steps or args.max_steps
    cfg.scheduler.lr_warmup_iters = 1 if schedule_steps > 1 else 0
    cfg.scheduler.lr_decay_iters = schedule_steps
    cfg.scheduler.max_steps = schedule_steps
    cfg.logger.log_interval = 1
    cfg.logger.tensorboard_dir = str(args.output_dir / "tensorboard")
    cfg.rng.seed = args.seed

    if args.finetuning_mode == "lora":
        from megatron.bridge.peft.lora import LoRA

        target_modules = (
            [
                "linear_q_down_proj",
                "linear_q_up_proj",
                "linear_kv_down_proj",
                "linear_kv_up_proj",
                "linear_proj",
            ]
            if spec.family == "glm4_moe_lite"
            else ["linear_qkv", "linear_proj"]
        )
        cfg.peft = LoRA(
            target_modules=target_modules,
            dim=args.lora_dim,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
    else:
        cfg.peft = None
        # Full 30B Adam is intentionally explicit.  The launcher does not
        # silently substitute a different optimizer or claim it fits.
        cfg.optimizer.optimizer = args.optimizer
        cfg.ddp.use_distributed_optimizer = args.distributed_optimizer
    cfg.optimizer.use_distributed_optimizer = args.distributed_optimizer

    cfg.ddp.overlap_grad_reduce = args.overlap_grad_reduce
    checkpoint_dir = args.output_dir / "checkpoints"
    cfg.checkpoint.pretrained_checkpoint = str(args.model_dir)
    cfg.checkpoint.save_interval = (
        getattr(args, "save_interval", None) or args.max_steps
    )
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.fully_parallel_save = args.fully_parallel_save
    cfg.checkpoint.fully_parallel_load = args.fully_parallel_load
    cfg.checkpoint.dist_ckpt_optim_fully_reshardable = getattr(
        args, "dist_ckpt_optim_fully_reshardable", False
    )
    cfg.checkpoint.async_save = (
        args.checkpoint_mode == "async" and args.stage in {"train", "resume"}
    )
    cfg.checkpoint.use_persistent_ckpt_worker = True
    cfg.checkpoint.save_optim = args.save_optimizer
    cfg.checkpoint.load_optim = args.load_optimizer and args.stage == "resume"
    load_checkpoint = _resolve_cluster_load_checkpoint(args, checkpoint_dir)
    if args.stage == "train":
        cfg.validation.skip_train = False
        cfg.checkpoint.finetune = True
        cfg.checkpoint.load = None
        cfg.checkpoint.save = str(checkpoint_dir)
    elif args.stage == "resume":
        cfg.validation.skip_train = False
        cfg.checkpoint.finetune = False
        cfg.checkpoint.load = str(load_checkpoint)
        cfg.checkpoint.save = str(checkpoint_dir)
    elif args.stage == "base":
        cfg.validation.skip_train = True
        cfg.checkpoint.finetune = True
        cfg.checkpoint.load = None
        cfg.checkpoint.save = None
    else:
        cfg.validation.skip_train = True
        cfg.checkpoint.finetune = False
        cfg.checkpoint.load = str(load_checkpoint)
        cfg.checkpoint.save = None
    return cfg, spec
