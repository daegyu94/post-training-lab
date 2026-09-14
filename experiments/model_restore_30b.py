"""Measure real single-node TRL checkpoint save and cold/warm restore."""

from __future__ import annotations

import argparse
import json
import shlex
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import checkpoint_memory_30b, run  # noqa: E402


VARIANTS = {
    "lora-r8": {"FINETUNING_MODE": "lora", "DISTRIBUTED_BACKEND": "ddp", "LORA_R": 8, "LORA_ALPHA": 16},
    "lora-r64": {"FINETUNING_MODE": "lora", "DISTRIBUTED_BACKEND": "ddp", "LORA_R": 64, "LORA_ALPHA": 128},
    "lora-r128": {"FINETUNING_MODE": "lora", "DISTRIBUTED_BACKEND": "ddp", "LORA_R": 128, "LORA_ALPHA": 256},
    "zero3-full": {
        "FINETUNING_MODE": "full",
        "DISTRIBUTED_BACKEND": "deepspeed",
        "DEEPSPEED_CONFIG": "configs/deepspeed-zero3-nvme.json",
    },
}


def experiment(model: str, variant: str, stage: str, load_dir: str | None = None) -> dict[str, Any]:
    spec = checkpoint_memory_30b.MODELS[model]
    env: dict[str, Any] = {
        "MODEL_ID": spec["id"],
        "MODEL_REVISION": spec["revision"],
        "DATASET_ID": checkpoint_memory_30b.DATASET_ID,
        "DATASET_REVISION": checkpoint_memory_30b.DATASET_REVISION,
        "MAX_STEPS": 1,
        "MAX_LENGTH": 512,
        "TRAIN_SAMPLES": 4,
        "EVAL_SAMPLES": 1,
        "GRADIENT_ACCUMULATION_STEPS": 1,
        "OPTIMIZER": "adamw",
        "LEARNING_RATE": "2e-5",
        "SEED": 42,
        "STAGE": stage,
        **VARIANTS[variant],
    }
    if load_dir is not None:
        env.update(LOAD_DIR=load_dir, RESTORE_ONLY=True)
    return {"backend": "trl", "setup": "spark", "nnodes": 1, "nproc_per_node": 1, "env": env}


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    result: dict[str, Any] = {"run_count": len(records)}
    train = [item["train"] for item in records]
    result["train"] = {
        "checkpoint_save_seconds_mean": statistics.fmean(item["checkpoint_save_seconds"] for item in train),
        "checkpoint_logical_bytes_mean": statistics.fmean(item["checkpoint_logical_bytes"] for item in train),
        "trainable_parameter_fraction_mean": statistics.fmean(item["trainable_parameter_fraction"] for item in train),
    }
    for cache_state in ("cold", "warm"):
        selected = [item[cache_state] for item in records]
        seconds = [item["model_restore_seconds"] for item in selected]
        result[cache_state] = {
            "model_restore_seconds_mean": statistics.fmean(seconds),
            "model_restore_seconds_stdev": statistics.stdev(seconds) if len(seconds) > 1 else 0.0,
            "effective_input_bytes_per_second_mean": statistics.fmean(
                item["restore_input_logical_bytes"] / item["model_restore_seconds"] for item in selected
            ),
            "process_storage_read_bytes_mean": statistics.fmean(
                item["restore_process_io"]["read_bytes"] for item in selected
            ),
        }
    return result


def _write_config(output: Path, name: str, value: dict[str, Any]) -> Path:
    path = output / "configs" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _remote(
    host: str,
    command: str,
    *,
    timeout: int,
    remote_run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> subprocess.CompletedProcess[str]:
    return remote_run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, command],
        check=False, capture_output=True, text=True, timeout=timeout,
    )


def prepare_cohort(
    setup: dict[str, Any], source: str, model: str, *, execute: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    updated = json.loads(json.dumps(setup))
    node = updated["nodes"][0]
    spec = checkpoint_memory_30b.MODELS[model]
    output_root = node["output_root"]["trl"] if isinstance(node["output_root"], dict) else node["output_root"]
    destination = str(Path(output_root) / "cohorts" / f"ultrachat-{model}-512-io-v1")
    script = str(Path(node["checkout"]) / "experiments" / "prepare_checkpoint_cohort.py")
    manifest = str(Path(destination) / "manifest.json")
    create = " ".join([
        shlex.quote(node["python"]["megatron"]), shlex.quote(script),
        "--source-dir", shlex.quote(source), "--output-dir", shlex.quote(destination),
        "--model-dir", shlex.quote(node["model_dirs"][spec["id"]]),
        "--model-revision", spec["revision"], "--max-length", "512",
        "--train-count", "4", "--eval-count", "1",
    ])
    command = f"if test -f {shlex.quote(manifest)}; then cat {shlex.quote(manifest)}; else {create}; fi"
    updated["nodes"] = [node]
    node["data_dir"] = destination
    planned = {"host": node["host"], "source": source, "path": destination, "command": command}
    if execute:
        result = _remote(node["host"], command, timeout=1800)
        if result.returncode:
            raise RuntimeError(f"cohort preparation failed: {result.stderr.strip()}")
        cohort = json.loads(result.stdout.strip())
        if cohort.get("dataset") != checkpoint_memory_30b.DATASET_ID or cohort.get("train_count") != 4:
            raise RuntimeError("cohort manifest does not match the single-node experiment")
        planned["manifest"] = cohort
    return updated, planned


def _fetch_summary(plan: dict[str, Any], stage: str, timeout: int) -> dict[str, Any]:
    rank = plan["ranks"][0]
    path = str(Path(rank["output"]) / f"summary-{stage}.json")
    result = _remote(rank["host"], f"cat {shlex.quote(path)}", timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"cannot fetch {stage} summary: {result.stderr.strip()}")
    return json.loads(result.stdout)


def _evict(plan: dict[str, Any], paths: list[str], timeout: int) -> dict[str, Any]:
    rank = plan["ranks"][0]
    script = str(Path(rank["checkout"]) / "experiments" / "checkpoint_io_probe.py")
    items = []
    for path in paths:
        command = " ".join([
            shlex.quote(rank["env"]["PYTHON"]), shlex.quote(script),
            "--checkpoint-dir", shlex.quote(path), "--evict-only",
        ])
        result = _remote(rank["host"], command, timeout=timeout)
        if result.returncode:
            raise RuntimeError(f"page-cache eviction failed for {path}: {result.stderr.strip()}")
        items.append(json.loads(result.stdout.strip()))
    return {"paths": paths, "items": items, "logical_bytes": sum(item["logical_bytes"] for item in items)}


def _cleanup(plans: list[dict[str, Any]], output_root: str, timeout: int) -> None:
    root = Path(output_root)
    for plan in plans:
        rank = plan["ranks"][0]
        path = Path(rank["output"])
        if path.parent != root or not path.name:
            raise RuntimeError(f"refusing cleanup outside TRL output root: {path}")
        result = _remote(rank["host"], f"rm -rf -- {shlex.quote(str(path))}", timeout=timeout)
        if result.returncode:
            raise RuntimeError(f"remote cleanup failed for {path}: {result.stderr.strip()}")


def _run(plan: dict[str, Any], timeout: int, output: Path) -> None:
    if run.execute(plan, timeout, output):
        raise RuntimeError(f"remote run failed: {output}")


def run_experiment(
    setup: dict[str, Any], setup_path: Path, output: Path, *, model: str, variant: str,
    repeats: int, timeout: int, execute: bool,
) -> dict[str, Any]:
    records = []
    output_root = setup["nodes"][0]["output_root"]
    output_root = output_root["trl"] if isinstance(output_root, dict) else output_root
    for repeat in range(repeats):
        train_name = f"{output.name}--train-{repeat + 1:02d}"
        train_config = _write_config(output, train_name, experiment(model, variant, "train"))
        train_output = output / "runs" / train_name
        train_plan = run.build_plan(setup, run.load_experiment(train_config), setup_path, train_config, train_output, ROOT)
        source = train_plan["ranks"][0]["output"]
        restore_plans = []
        for cache_state in ("cold", "warm"):
            name = f"{output.name}--restore-{cache_state}-{repeat + 1:02d}"
            config = _write_config(output, name, experiment(model, variant, "tuned", source))
            local_output = output / "runs" / name
            restore_plans.append(run.build_plan(setup, run.load_experiment(config), setup_path, config, local_output, ROOT))
        if not execute:
            records.append({"repeat": repeat + 1, "train_plan": train_plan, "restore_plans": restore_plans})
            continue
        _run(train_plan, timeout, train_output)
        train_summary = _fetch_summary(train_plan, "train", timeout)
        artifact = str(Path(source) / ("adapter" if variant.startswith("lora-") else "model"))
        eviction = _evict(train_plan, [train_plan["ranks"][0]["env"]["MODEL_DIR"], artifact], timeout)
        restore_summaries = {}
        for cache_state, plan in zip(("cold", "warm"), restore_plans, strict=True):
            _run(plan, timeout, Path(plan["output"]))
            summary = _fetch_summary(plan, "tuned", timeout)
            summary["restore_input_logical_bytes"] = eviction["logical_bytes"]
            restore_summaries[cache_state] = summary
        records.append({"repeat": repeat + 1, "train": train_summary, "eviction": eviction, **restore_summaries})
        (output / "manifest.json").write_text(json.dumps({
            "status": "running", "model": checkpoint_memory_30b.MODELS[model]["id"],
            "variant": variant, "repeats": repeats, "records": records,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _cleanup([train_plan, *restore_plans], output_root, timeout)
    if not execute:
        return {"status": "dry-run", "model": model, "variant": variant, "records": records}
    return {
        "status": "passed" if len(records) == repeats else "failed",
        "model": checkpoint_memory_30b.MODELS[model]["id"], "variant": variant,
        "repeats": repeats, "records": records, "summary": summarize_records(records),
        "remote_outputs_removed_after_collection": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", type=Path, required=True)
    parser.add_argument("--dataset-source", required=True, help="prepared UltraChat directory on spark1")
    parser.add_argument("--remote-checkout", help="override setup checkout on participating nodes")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(checkpoint_memory_30b.MODELS), required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise run.ConfigError(f"refusing to reuse output directory: {args.output}")
        if args.repeats < 1 or args.timeout < 1:
            raise run.ConfigError("--repeats and --timeout must be positive")
        setup = run.load_setup(args.setup)
        if args.remote_checkout:
            for node in setup["nodes"]:
                node["checkout"] = args.remote_checkout
        setup, cohort = prepare_cohort(setup, args.dataset_source, args.model, execute=args.execute)
        args.output.mkdir(parents=True)
        setup_path = checkpoint_memory_30b.generated_setup(setup, args.output)
        result = run_experiment(
            setup, setup_path.resolve(), args.output, model=args.model, variant=args.variant,
            repeats=args.repeats, timeout=args.timeout, execute=args.execute,
        )
        result["cohort"] = cohort
        (args.output / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, RuntimeError, run.ConfigError, subprocess.SubprocessError) as exc:
        print(f"experiment error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {"dry-run", "passed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
