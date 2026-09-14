"""Measure real Megatron checkpoint restore latency on node-local NVMe."""

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

from experiments import benchmarks, checkpoint_memory_30b, run  # noqa: E402


def experiment(model: str, stage: str, load_checkpoint: str | None = None) -> dict[str, Any]:
    spec = checkpoint_memory_30b.MODELS[model]
    env: dict[str, Any] = {
        "MODEL_ID": spec["id"],
        "MODEL_REVISION": spec["revision"],
        "DATASET_ID": checkpoint_memory_30b.DATASET_ID,
        "DATASET_REVISION": checkpoint_memory_30b.DATASET_REVISION,
        "FINETUNING_MODE": "lora",
        "MAX_STEPS": 1,
        "SCHEDULE_STEPS": 1,
        "MAX_LENGTH": 2048,
        "PAD_TO_MAX_LENGTH": True,
        "GLOBAL_BATCH_SIZE": 2,
        "MICRO_BATCH_SIZE": 1,
        "TP": 1,
        "PP": 1,
        "EP": 2,
        "EVAL_ITERS": 1,
        "SAVE_INTERVAL": 1,
        "SAVE_OPTIMIZER": False,
        "LOAD_OPTIMIZER": False,
        "CHECKPOINT_MODE": "sync",
        "CHECKPOINT_PLACEMENT": "shared",
        "MEASURE_TIMING": True,
        "SEED": 42,
        "STAGE": stage,
        **spec["megatron_env"],
    }
    if load_checkpoint is not None:
        env["LOAD_CHECKPOINT"] = load_checkpoint
    return {"backend": "megatron", "setup": "spark", "nnodes": 2, "nproc_per_node": 1, "env": env}


def restore_schedule(repeats: int) -> list[dict[str, Any]]:
    return [
        {"cache_state": cache_state, "warmup": pair == 0, "pair": pair}
        for pair in range(repeats + 1)
        for cache_state in ("cold", "warm")
    ]


def summarize_restore(metrics_by_rank: dict[str, Any]) -> dict[str, Any]:
    metrics = [item["metrics"] for item in metrics_by_rank.values()]
    seconds = [item.get("load_call_host_seconds_sum_per_rank") for item in metrics]
    if (
        not metrics
        or any(value is None for value in seconds)
        or any(item.get("load_call_count_per_rank") != 1 for item in metrics)
    ):
        raise ValueError("exactly one successful checkpoint load is required on every rank")
    critical_seconds = max(float(value) for value in seconds)
    logical = sum(float(item["load_process_logical_read_bytes_sum_per_rank"] or 0) for item in metrics)
    storage = sum(float(item["load_process_storage_read_bytes_sum_per_rank"] or 0) for item in metrics)
    if critical_seconds <= 0 or logical <= 0:
        raise ValueError("checkpoint load duration and logical read bytes must be positive")
    model_ready = [item.get("model_ready_seconds_per_rank") for item in metrics]
    return {
        "restore_seconds_max_across_ranks": critical_seconds,
        "process_logical_read_bytes_sum_across_ranks": logical,
        "process_storage_read_bytes_sum_across_ranks": storage,
        "effective_logical_read_bytes_per_second": logical / critical_seconds,
        "observed_storage_read_bytes_per_second": storage / critical_seconds,
        "model_ready_seconds_max_across_ranks": (
            max(float(value) for value in model_ready if value is not None)
            if any(value is not None for value in model_ready) else None
        ),
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for cache_state in ("cold", "warm"):
        selected = [
            item["restore"] for item in records
            if item["status"] == "passed" and not item["warmup"]
            and item["cache_state"] == cache_state
        ]
        seconds = [item["restore_seconds_max_across_ranks"] for item in selected]
        throughputs = [item["effective_logical_read_bytes_per_second"] for item in selected]
        ready = [item["model_ready_seconds_max_across_ranks"] for item in selected]
        result[cache_state] = {
            "run_count": len(selected),
            "restore_seconds_median": statistics.median(seconds) if seconds else None,
            "effective_logical_read_bytes_per_second_median": (
                statistics.median(throughputs) if throughputs else None
            ),
            "restore_seconds_relative_mad": checkpoint_memory_30b.relative_mad(seconds),
            "model_ready_seconds_median": (
                statistics.median(value for value in ready if value is not None)
                if any(value is not None for value in ready) else None
            ),
        }
    return result


def _write_config(output: Path, name: str, value: dict[str, Any]) -> Path:
    path = output / "configs" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _replicate_checkpoint(
    source_plan: dict[str, Any],
    *,
    remote_run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[str, list[dict[str, Any]]]:
    sources = {rank["env"]["CHECKPOINT_DIR"] for rank in source_plan["ranks"]}
    destinations = {str(Path(rank["output"]) / "restore-checkpoint") for rank in source_plan["ranks"]}
    if len(sources) != 1 or len(destinations) != 1:
        raise ValueError("restore experiment requires the same shared source and local path on every node")
    source, destination = sources.pop(), destinations.pop()
    copies = []
    for rank in source_plan["ranks"]:
        command = f"test ! -e {shlex.quote(destination)} && cp -a -- {shlex.quote(source)} {shlex.quote(destination)}"
        result = remote_run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", rank["host"], command],
            check=False, capture_output=True, text=True, timeout=1800,
        )
        if result.returncode:
            raise RuntimeError(f"checkpoint replication failed on rank {rank['rank']}: {result.stderr.strip()}")
        copies.append({"rank": rank["rank"], "host": rank["host"], "source": source, "destination": destination})
    return destination, copies


def _evict_checkpoint(
    plan: dict[str, Any],
    checkpoint: str,
    *,
    remote_run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[dict[str, Any]]:
    results = []
    for rank in plan["ranks"]:
        script = str(Path(rank["checkout"]) / "experiments" / "checkpoint_io_probe.py")
        command = " ".join([
            shlex.quote(rank["env"]["PYTHON"]), shlex.quote(script),
            "--checkpoint-dir", shlex.quote(checkpoint), "--evict-only",
        ])
        result = remote_run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", rank["host"], command],
            check=False, capture_output=True, text=True, timeout=1800,
        )
        if result.returncode:
            raise RuntimeError(f"checkpoint eviction failed on rank {rank['rank']}: {result.stderr.strip()}")
        results.append({"rank": rank["rank"], **json.loads(result.stdout.strip().splitlines()[-1])})
    return results


def run_experiment(
    setup: dict[str, Any],
    setup_path: Path,
    output: Path,
    *,
    model: str,
    repeats: int,
    timeout: int,
    execute: bool,
) -> dict[str, Any]:
    schedule = restore_schedule(repeats)
    source_config = _write_config(output, "source", experiment(model, "train"))
    source_output = output / "runs" / "source"
    source_plan = run.build_plan(
        setup, run.load_experiment(source_config), setup_path, source_config, source_output, ROOT
    )
    if not execute:
        return {"status": "dry-run", "source_plan": source_plan, "planned_restores": schedule}

    code = run.execute(source_plan, timeout, source_output)
    if code:
        return {"status": "failed", "source_exit_code": code, "records": []}
    checkpoint, copies = _replicate_checkpoint(source_plan)
    records = []
    restore_config = _write_config(output, "restore", experiment(model, "tuned", checkpoint))
    restore_experiment = run.load_experiment(restore_config)
    for index, item in enumerate(schedule):
        name = f"restore-{item['cache_state']}-{index:02d}"
        run_output = output / "runs" / name
        plan = run.build_plan(setup, restore_experiment, setup_path, restore_config, run_output, ROOT)
        record = {**item, "name": name, "status": "running", "output": str(run_output)}
        if item["cache_state"] == "cold":
            record["eviction"] = _evict_checkpoint(plan, checkpoint)
        code = run.execute(plan, timeout, run_output)
        record.update(status="passed" if code == 0 else "failed", exit_code=code)
        if code == 0:
            metrics = benchmarks.fetch_measurements(plan, run_output, stage="tuned")
            record["metrics_by_rank"] = metrics
            record["restore"] = summarize_restore(metrics)
        records.append(record)
        (output / "manifest.json").write_text(
            json.dumps({"status": "running", "model": checkpoint_memory_30b.MODELS[model]["id"],
                        "checkpoint_replicas": copies, "records": records}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if code:
            break
    status = "passed" if len(records) == len(schedule) and all(item["status"] == "passed" for item in records) else "failed"
    return {"status": status, "model": checkpoint_memory_30b.MODELS[model]["id"],
            "checkpoint_replicas": copies, "records": records, "summary": summarize_records(records)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(checkpoint_memory_30b.MODELS), default="qwen")
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
        setup, cohorts = checkpoint_memory_30b.prepare_cohorts(
            setup, execute=args.execute, model=args.model
        )
        setup_path = checkpoint_memory_30b.generated_setup(setup, args.output)
        args.output.mkdir(parents=True)
        result = run_experiment(
            setup, setup_path.resolve(), args.output, model=args.model,
            repeats=args.repeats, timeout=args.timeout, execute=args.execute,
        )
        result["cohorts"] = cohorts
        (args.output / "manifest.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, RuntimeError, run.ConfigError, subprocess.SubprocessError) as exc:
        print(f"experiment error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {"dry-run", "passed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
