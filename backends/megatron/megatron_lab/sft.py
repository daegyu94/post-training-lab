"""Run one stage of the Spark-cluster Megatron Bridge workflow."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from megatron_lab.cluster import topology_from_environment
from megatron_lab.config import (
    DATASET_ID,
    build_cluster_config,
    select_transformer_impl,
)
from megatron_lab.model_cache import require_immutable_revision, validate_snapshot


def positive_int(value: str) -> int:
    """Parse a strictly positive CLI integer."""

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("base", "train", "resume", "tuned"), required=True)
    parser.add_argument("--setup", choices=("spark-cluster",), default="spark-cluster")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--load-checkpoint", type=Path)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument(
        "--save-interval",
        type=positive_int,
        help="setup2 checkpoint interval in optimizer steps (default: max-steps)",
    )
    parser.add_argument("--schedule-steps", type=int)
    parser.add_argument("--eval-iters", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--pad-to-max-length",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="pad every example to max-length instead of only capping it",
    )
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=2)
    parser.add_argument(
        "--transformer-impl",
        choices=("auto", "local", "transformer_engine"),
        default="auto",
        help="setup2 attention/layer backend; auto selects the model-safe default",
    )
    parser.add_argument("--finetuning-mode", choices=("lora", "full"), default="lora")
    parser.add_argument("--lora-dim", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--optimizer", default="adam")
    parser.add_argument("--distributed-optimizer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overlap-grad-reduce", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fully-parallel-save", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fully-parallel-load", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--dist-ckpt-optim-fully-reshardable",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use the optimizer checkpoint format that supports TP/PP resharding",
    )
    parser.add_argument("--checkpoint-mode", choices=("sync", "async"), default="sync")
    parser.add_argument("--save-optimizer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load-optimizer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sequence-parallel", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--recompute", choices=("none", "full", "selective"), default="full")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def rank() -> int:
    return int(os.environ.get("RANK", "0"))


class _Tee:
    def __init__(self, *streams: object) -> None:
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def configure_rank_log(args: argparse.Namespace) -> None:
    log_dir = os.environ.get("RANK_LOG_DIR")
    if not log_dir:
        return
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    destination = (path / f"rank-{rank()}-{args.stage}.log").open(
        "a", encoding="utf-8"
    )
    sys.stdout = _Tee(sys.__stdout__, destination)
    sys.stderr = _Tee(sys.__stderr__, destination)


def write_run_metadata(args: argparse.Namespace, spec: object, topology: object) -> None:
    """Write rank-local provenance and retain the rank-zero compatibility file."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_train_split = Path(args.train_data).name
    input_eval_split = Path(args.eval_data).name
    load_checkpoint = None
    if args.stage in {"resume", "tuned"}:
        load_checkpoint = str(
            getattr(args, "load_checkpoint", None)
            or (args.output_dir / "checkpoints")
        )
    model_spec = spec
    selected_transformer_impl = select_transformer_impl(
        model_spec,
        getattr(args, "transformer_impl", "auto"),
    )
    metadata = {
        "setup": args.setup,
        "stage": args.stage,
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "model_type": getattr(spec, "model_type", None),
        "model_snapshot": getattr(args, "model_snapshot", None),
        "dataset": args.dataset_id or DATASET_ID,
        "dataset_revision": args.dataset_revision,
        "train_split": input_train_split,
        "eval_split": input_eval_split,
        "input_files": {
            "train": str(args.train_data),
            "validation": str(args.eval_data),
        },
        "load_checkpoint": load_checkpoint,
        "transformer_impl": selected_transformer_impl,
        "seed": args.seed,
        "configuration": {
            "finetuning_mode": args.finetuning_mode,
            "max_length": args.max_length,
            "pad_to_max_length": getattr(args, "pad_to_max_length", False),
            "max_steps": args.max_steps,
            "save_interval": (
                getattr(args, "save_interval", None) or args.max_steps
            ),
            "schedule_steps": args.schedule_steps or args.max_steps,
            "optimizer": args.optimizer,
            "micro_batch_size": topology.micro_batch_size,
            "global_batch_size": topology.global_batch_size,
            "checkpoint_mode": args.checkpoint_mode,
            "fully_parallel_save": args.fully_parallel_save,
            "fully_parallel_load": args.fully_parallel_load,
            "dist_ckpt_optim_fully_reshardable": getattr(
                args, "dist_ckpt_optim_fully_reshardable", False
            ),
            "overlap_grad_reduce": args.overlap_grad_reduce,
        },
        "topology": topology.as_dict(),
    }
    encoded = json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    (args.output_dir / f"run-metadata-{args.stage}-rank-{rank()}.json").write_text(
        encoded, encoding="utf-8"
    )
    if rank() == 0:
        (args.output_dir / f"run-metadata-{args.stage}.json").write_text(
            encoded, encoding="utf-8"
        )


def main() -> None:
    args = parse_args()
    configure_rank_log(args)
    try:
        require_immutable_revision(args.model_revision, name="model revision")
        require_immutable_revision(args.dataset_revision, name="dataset revision")
        args.model_snapshot = validate_snapshot(args.model_dir, args.model_revision)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    topology = topology_from_environment(
        tp=args.tp,
        pp=args.pp,
        ep=args.ep,
        micro_batch_size=args.micro_batch_size,
        global_batch_size=args.global_batch_size,
    )
    args.topology = topology
    config, spec = build_cluster_config(args)
    write_run_metadata(args, spec, topology)
    print(
        f"[stage] rank={rank()} model={args.model_id} "
        f"model_type={spec.model_type} dataset={args.dataset_id} "
        f"model_revision={args.model_revision} dataset_revision={args.dataset_revision}",
        flush=True,
    )
    print(
        f"[stage] world_size={topology.world_size} tp={topology.tensor_parallel_size} "
        f"pp={topology.pipeline_parallel_size} ep={topology.expert_parallel_size} "
        f"dp={topology.data_parallel_size} gbs={topology.global_batch_size}",
        flush=True,
    )
    if spec.family == "glm4_moe_lite":
        print(
            "[boundary] setup2 uses native final-assistant prompt/completion "
            "preprocessing; token prefix, EOS, and non-empty supervision "
            "were checked during config preparation",
            flush=True,
        )

    from megatron.bridge.training.finetune import finetune
    from megatron.bridge.training.gpt_step import forward_step

    if os.environ.get("MEASURE_TIMING", "false") == "true":
        from megatron_lab.measurement import measure_execution
        with measure_execution(args.output_dir, args.stage):
            finetune(config=config, forward_step_func=forward_step)
    else:
        finetune(config=config, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
