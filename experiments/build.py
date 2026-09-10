"""Compose an experiment.json from simple knobs (backend/dataset/model/offload/epochs)
and run it through experiments.run's existing, tested validation/execution path.

This intentionally does not re-implement anything experiments/run.py already does:
it only maps high-level knobs onto the exact experiment.json schema load_experiment()
validates, writes that file to experiments/generated/<output-name>.json (build_plan()
hashes the literal file for provenance, so there is no in-memory shortcut), then calls
straight into run.load_setup / run.load_experiment / run.build_plan / run.execute.

Dataset presets are read directly from datasets_lab.public_data.PRESETS, the single
adapter both backends share, rather than hardcoding a second list.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _path in [str(ROOT), *(str(ROOT / "backends" / backend) for backend in ("trl", "megatron"))]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from experiments import run  # noqa: E402

DEEPSPEED_CONFIGS = {
    "cpu": "configs/deepspeed-zero3-cpu.json",
    "nvme": "configs/deepspeed-zero3-nvme.json",
}
DEFAULT_GENERATED_DIR = ROOT / "experiments" / "generated"


def _dataset_presets() -> dict[str, Any]:
    from datasets_lab.public_data import PRESETS

    return {key: spec for key, spec in PRESETS.items() if not spec.reference_only}


def _read_manifest(data_dir: Path) -> dict[str, Any] | None:
    """Best-effort: data_dir is the node's path, which controller may not be able to
    read directly (e.g. it's expressed as the Spark-side NFS mount point, not
    controller's). Returns None (skip validation) rather than failing the whole
    tool when controller simply can't see the path - the remote launcher's own
    validate_dataset_manifest() is the authoritative check at actual run time.
    """
    manifest_path = data_dir / "manifest.json"
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _check_manifest_matches(manifest: dict[str, Any], data_dir: Path, dataset_id: str, dataset_revision: str) -> None:
    if manifest.get("dataset") != dataset_id or manifest.get("dataset_revision") != dataset_revision:
        raise run.ConfigError(
            f"manifest at {data_dir / 'manifest.json'} is for "
            f"{manifest.get('dataset')}@{manifest.get('dataset_revision')}, not {dataset_id}@{dataset_revision}. "
            "Re-run prepare_public_data.py for the requested dataset first (see docs/datasets.md)."
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=("trl", "megatron"), required=True)
    parser.add_argument("--dataset", required=True, help="preset key; valid choices depend on --backend (see docs/datasets.md)")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True, help="40-hex immutable model snapshot revision")
    parser.add_argument("--dataset-revision", help="defaults to the preset's pinned revision")
    parser.add_argument("--setup", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nnodes", type=int, default=2, help="Spark nodes to use (this project's cluster has 2)")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--execute", action="store_true", help="run remote jobs; default is dry-run")
    parser.add_argument("--stage", choices=("all", "base", "train", "tuned"), default="all")
    parser.add_argument("--finetuning-mode", choices=("lora", "full"), default="lora")
    parser.add_argument("--learning-rate", default="2e-5")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, help="mutually exclusive with --max-steps")
    parser.add_argument("--max-steps", type=int, help="mutually exclusive with --epochs")

    trl_group = parser.add_argument_group("TRL-only")
    trl_group.add_argument("--distributed-backend", choices=("ddp", "fsdp2", "deepspeed"), default="ddp")
    trl_group.add_argument("--offload", choices=("none", "cpu", "nvme"), default="none", help="TRL only; forces --distributed-backend deepspeed when cpu/nvme")
    trl_group.add_argument("--optimizer", default="adamw", help="TRL: adamw|sgd. Megatron: passed through as-is (default adam)")
    trl_group.add_argument("--gradient-accumulation-steps", type=int, default=8)
    trl_group.add_argument("--train-samples", type=int)
    trl_group.add_argument("--eval-samples", type=int)

    megatron_group = parser.add_argument_group("Megatron-only", "parallelism defaults match this repo's already-verified 30B presets")
    megatron_group.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    megatron_group.add_argument("--pp", type=int, default=1, help="pipeline parallel size")
    megatron_group.add_argument("--ep", type=int, default=2, help="expert parallel size (MoE)")
    megatron_group.add_argument("--global-batch-size", type=int, default=8)
    megatron_group.add_argument("--micro-batch-size", type=int, default=1)

    args = parser.parse_args(argv)
    if args.epochs is not None and args.max_steps is not None:
        parser.error("--epochs and --max-steps are mutually exclusive")
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    return args


def build_experiment(args: argparse.Namespace, setup: dict[str, Any]) -> dict[str, Any]:
    presets = _dataset_presets()
    if args.dataset not in presets:
        raise run.ConfigError(f"--dataset must be one of {sorted(presets)} for backend {args.backend}")
    spec = presets[args.dataset]
    dataset_revision = args.dataset_revision or spec.default_revision
    if not dataset_revision:
        raise run.ConfigError(f"--dataset-revision is required: preset {args.dataset!r} has no pinned default revision")

    if args.offload != "none" and args.backend == "megatron":
        raise run.ConfigError("--offload is TRL-only; Megatron has no offload/parameter-offload path today")
    if args.offload != "none" and args.backend == "trl" and args.distributed_backend not in ("ddp", "deepspeed"):
        raise run.ConfigError("--offload requires --distributed-backend deepspeed (or the default, which it will set)")
    if args.offload == "nvme" and args.finetuning_mode == "lora":
        raise run.ConfigError("--offload nvme requires --finetuning-mode full; use --offload none for LoRA")

    env: dict[str, Any] = {
        "MODEL_ID": args.model_id, "MODEL_REVISION": args.model_revision,
        "DATASET_ID": spec.dataset_id, "DATASET_REVISION": dataset_revision,
        "FINETUNING_MODE": args.finetuning_mode, "MAX_LENGTH": args.max_length,
        "SEED": args.seed, "STAGE": args.stage,
    }

    node = setup["nodes"][0]
    data_dir = Path(node["data_dir"])
    manifest = _read_manifest(data_dir)
    if manifest is not None:
        _check_manifest_matches(manifest, data_dir, spec.dataset_id, dataset_revision)

    if args.backend == "trl":
        distributed_backend = args.distributed_backend
        if args.offload != "none":
            distributed_backend = "deepspeed"
            env["DEEPSPEED_CONFIG"] = DEEPSPEED_CONFIGS[args.offload]
        env.update({
            "DISTRIBUTED_BACKEND": distributed_backend,
            "OPTIMIZER": args.optimizer,
            "LEARNING_RATE": args.learning_rate,
            "GRADIENT_ACCUMULATION_STEPS": args.gradient_accumulation_steps,
        })
        if args.epochs is not None:
            env["NUM_TRAIN_EPOCHS"] = args.epochs
        elif args.max_steps is not None:
            env["MAX_STEPS"] = args.max_steps
        if args.train_samples is not None:
            env["TRAIN_SAMPLES"] = args.train_samples
        if args.eval_samples is not None:
            env["EVAL_SAMPLES"] = args.eval_samples
    else:
        env.update({
            "GLOBAL_BATCH_SIZE": args.global_batch_size, "MICRO_BATCH_SIZE": args.micro_batch_size,
            "TP": args.tp, "PP": args.pp, "EP": args.ep,
        })
        if args.epochs is not None:
            train_count = manifest.get("train_count") if manifest is not None else None
            if not isinstance(train_count, int) or train_count < 1:
                raise run.ConfigError(
                    f"cannot convert --epochs to Megatron steps: no readable train_count at "
                    f"{data_dir / 'manifest.json'} from this host. Pass --max-steps instead, "
                    "or run this from a host that can read the node's data_dir."
                )
            steps = math.ceil(args.epochs * train_count / args.global_batch_size)
            env["MAX_STEPS"] = steps
            env["SCHEDULE_STEPS"] = steps
            env["NUM_TRAIN_EPOCHS"] = args.epochs
        elif args.max_steps is not None:
            env["MAX_STEPS"] = args.max_steps

    return {
        "backend": args.backend, "setup": "spark",
        "nnodes": args.nnodes, "nproc_per_node": 1,
        "env": env,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        setup = run.load_setup(args.setup)
        experiment = build_experiment(args, setup)
        DEFAULT_GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        generated_path = DEFAULT_GENERATED_DIR / f"{args.output.name}.json"
        generated_path.write_text(json.dumps(experiment, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        experiment = run.load_experiment(generated_path)
        output = args.output.resolve()
        if output.exists():
            raise run.ConfigError(f"refusing to reuse existing output directory: {output}")
        plan = run.build_plan(setup, experiment, args.setup.resolve(), generated_path.resolve(), output, ROOT)
    except run.ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    if not args.execute:
        print(json.dumps({**plan, "mode": "dry-run", "timeout_seconds": args.timeout}, indent=2, sort_keys=True))
        return 0
    try:
        return run.execute(plan, args.timeout, output)
    except run.ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
