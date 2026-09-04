"""Build the Qwen2.5-14B Megatron Bridge LoRA configuration."""

from __future__ import annotations

from argparse import Namespace


MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
DATASET_ID = "HuggingFaceH4/ultrachat_200k"


def build_config(args: Namespace):
    from megatron.bridge import AutoBridge
    from megatron.bridge.data.builders import (
        ChatSFTPreprocessingConfig,
        DirectHFSFTDatasetConfig,
        HFDatasetSourceConfig,
    )
    from megatron.bridge.recipes.qwen import qwen25_14b_peft_config

    cfg = qwen25_14b_peft_config(peft_scheme="lora")
    cfg.model = AutoBridge.from_hf_pretrained(
        str(args.model_dir)
    ).to_megatron_provider(load_weights=False)
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = args.max_length
    cfg.model.transformer_impl = "local"
    cfg.model.cross_entropy_loss_fusion = False
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1

    cfg.tokenizer.tokenizer_model = str(args.model_dir)
    cfg.dataset = DirectHFSFTDatasetConfig(
        seq_length=args.max_length,
        preprocessing=ChatSFTPreprocessingConfig(loss_mode="assistant"),
        hf_processor_path=str(args.model_dir),
        source=HFDatasetSourceConfig(
            path_or_dataset="json",
            split="train",
            load_kwargs={"data_files": {"train": str(args.train_data)}},
        ),
        validation_source=HFDatasetSourceConfig(
            path_or_dataset="json",
            split="validation",
            load_kwargs={"data_files": {"validation": str(args.eval_data)}},
        ),
        do_validation=True,
        do_test=False,
        dataloader_type="cyclic",
        num_workers=0,
        pin_memory=True,
    )

    cfg.train.train_iters = args.max_steps
    cfg.train.global_batch_size = args.global_batch_size
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_interval = args.max_steps
    cfg.validation.eval_iters = args.eval_iters
    cfg.scheduler.lr_warmup_iters = 1 if args.max_steps > 1 else 0
    cfg.scheduler.lr_decay_iters = args.max_steps
    cfg.logger.log_interval = 1
    cfg.logger.tensorboard_dir = str(args.output_dir / "tensorboard")
    cfg.rng.seed = args.seed

    checkpoint_dir = args.output_dir / "checkpoints"
    cfg.checkpoint.pretrained_checkpoint = str(args.model_dir)
    cfg.checkpoint.save_interval = args.max_steps
    if args.stage == "train":
        cfg.validation.skip_train = False
        cfg.checkpoint.load = None
        cfg.checkpoint.save = str(checkpoint_dir)
    elif args.stage == "base":
        cfg.validation.skip_train = True
        cfg.checkpoint.load = None
        cfg.checkpoint.save = None
    else:
        cfg.validation.skip_train = True
        cfg.checkpoint.load = str(checkpoint_dir)
        cfg.checkpoint.save = None
    return cfg
