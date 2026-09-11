"""Prepare UltraChat and run the local-NVMe 30B checkpoint or memory experiment."""

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

from experiments import benchmarks, run  # noqa: E402

MODEL_ID = "Qwen/Qwen3-30B-A3B"
MODEL_REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_REVISION = "8049631c405ae6576f93f445c6b8166f76f5505a"
DEFAULT_PLAN = ROOT / "experiments" / "megatron" / "checkpoint-memory-30b.json"
GENERATED = ROOT / "experiments" / "generated"


def _output_root(node: dict[str, Any], backend: str) -> str:
    value = node["output_root"]
    return value[backend] if isinstance(value, dict) else value


def cohort_path(node: dict[str, Any]) -> str:
    return str(Path(_output_root(node, "megatron")) / "cohorts" / "ultrachat-qwen3-30b-2048-v1")


def _cohort_command(node: dict[str, Any], output: str) -> str:
    q = shlex.quote
    manifest = str(Path(output) / "manifest.json")
    script = str(Path(node["checkout"]) / "experiments" / "prepare_checkpoint_cohort.py")
    model_dir = node["model_dirs"][MODEL_ID]
    create = " ".join([
        q(node["python"]["megatron"]), q(script), "--source-dir", q(node["data_dir"]),
        "--output-dir", q(output), "--model-dir", q(model_dir), "--model-revision", MODEL_REVISION,
        "--dataset-id", DATASET_ID, "--dataset-revision", DATASET_REVISION,
        "--max-length", "2048", "--train-count", "32", "--eval-count", "8",
    ])
    return f"if test -f {q(manifest)}; then cat {q(manifest)}; else {create}; fi"


def prepare_cohorts(
    setup: dict[str, Any],
    *,
    execute: bool,
    remote_run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    updated = json.loads(json.dumps(setup))
    planned = []
    manifests = []
    for node in updated["nodes"]:
        if MODEL_ID not in node["model_dirs"]:
            raise run.ConfigError(f"node {node['host']} has no {MODEL_ID} model path")
        destination = cohort_path(node)
        command = _cohort_command(node, destination)
        planned.append({"host": node["host"], "path": destination, "command": command})
        node["data_dir"] = destination
        if not execute:
            continue
        result = remote_run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", node["host"], command],
            check=False, capture_output=True, text=True, timeout=1800,
        )
        if result.returncode:
            raise RuntimeError(f"cohort preparation failed on {node['host']}: {result.stderr.strip()}")
        # The cache-hit branch `cat`s a pretty-printed manifest.json (multi-line);
        # the create branch prints a single compact line. json.loads handles both
        # as long as the whole trimmed stdout is parsed, not just its last line.
        manifest = json.loads(result.stdout.strip())
        if (
            manifest.get("dataset") != DATASET_ID
            or manifest.get("dataset_revision") != DATASET_REVISION
            or manifest.get("model_revision") != MODEL_REVISION
            or manifest.get("max_length") != 2048
            or manifest.get("train_count") != 32
            or manifest.get("eval_count") != 8
        ):
            raise RuntimeError(f"cohort manifest mismatch on {node['host']}")
        manifests.append(manifest)
    if execute:
        selections = {item["source_selection_sha256"] for item in manifests}
        # source_selection_sha256 only hashes the chosen prompt IDs, so nodes with the
        # same selection but divergently-provisioned UltraChat copies could still pass;
        # also compare the actual per-file content hashes across nodes.
        contents = {
            tuple(sorted((name, entry["sha256"]) for name, entry in item["files"].items()))
            for item in manifests
        }
        if len(selections) != 1 or len(contents) != 1:
            raise RuntimeError("cohort selection or content differs across Spark nodes")
    return updated, planned


def summarize_resources(lines: list[str]) -> dict[str, Any]:
    samples = [json.loads(line) for line in lines if line.strip()]
    if not samples:
        raise ValueError("resource sampler output is empty")
    first, last = samples[0], samples[-1]
    elapsed = float(last["monotonic_seconds"]) - float(first["monotonic_seconds"])
    read_bytes = last["read_bytes"] - first["read_bytes"]
    write_bytes = last["write_bytes"] - first["write_bytes"]
    return {
        "sample_count": len(samples),
        "elapsed_seconds": elapsed,
        "device_major_minor": first["device_major_minor"],
        "mem_available_min_bytes": min(item["memavailable_bytes"] for item in samples),
        "mem_available_first_bytes": first["memavailable_bytes"],
        "host_memory_pressure_bytes": first["memavailable_bytes"] - min(item["memavailable_bytes"] for item in samples),
        "swap_used_max_bytes": max(item["swaptotal_bytes"] - item["swapfree_bytes"] for item in samples),
        "device_read_bytes_delta": read_bytes,
        "device_write_bytes_delta": write_bytes,
        "device_read_operations_delta": last["read_operations"] - first["read_operations"],
        "device_write_operations_delta": last["write_operations"] - first["write_operations"],
        "device_read_time_ms_delta": last["read_time_ms"] - first["read_time_ms"],
        "device_write_time_ms_delta": last["write_time_ms"] - first["write_time_ms"],
        "device_busy_time_ms_delta": last["busy_time_ms"] - first["busy_time_ms"],
        "device_in_flight_max": max(item["in_flight"] for item in samples),
        "device_read_bytes_per_second": read_bytes / elapsed if elapsed > 0 else None,
        "device_write_bytes_per_second": write_bytes / elapsed if elapsed > 0 else None,
    }


def collect_post_run(
    plan: dict[str, Any],
    run_output: Path,
    _: dict[str, Any],
    *,
    remote_run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    by_rank = {}
    for rank in plan["ranks"]:
        remote_probe = str(Path(rank["output"]) / "measurements" / f"checkpoint-io-rank-{rank['rank']}.json")
        script = str(Path(rank["checkout"]) / "experiments" / "checkpoint_io_probe.py")
        command = " ".join([
            shlex.quote(rank["env"]["PYTHON"]), shlex.quote(script),
            "--checkpoint-dir", shlex.quote(rank["env"]["CHECKPOINT_DIR"]),
            "--output", shlex.quote(remote_probe),
        ])
        result = remote_run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", rank["host"], command],
            check=False, capture_output=True, text=True, timeout=1800,
        )
        if result.returncode:
            raise RuntimeError(f"checkpoint I/O probe failed on rank {rank['rank']}: {result.stderr.strip()}")
        probe = json.loads(result.stdout.strip().splitlines()[-1])
        resource_path = str(Path(rank["output"]) / "measurements" / f"resources-node-{rank['rank']}.jsonl")
        resource = remote_run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", rank["host"], f"cat -- {shlex.quote(resource_path)}"],
            check=False, capture_output=True, text=True, timeout=35,
        )
        if resource.returncode:
            raise RuntimeError(f"cannot fetch resource samples for rank {rank['rank']}")
        destination = run_output / "measurements"
        destination.mkdir(parents=True, exist_ok=True)
        (destination / f"checkpoint-io-rank-{rank['rank']}.json").write_text(
            json.dumps(probe, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (destination / f"resources-node-{rank['rank']}.jsonl").write_text(resource.stdout, encoding="utf-8")
        by_rank[str(rank["rank"])] = {"checkpoint_io": probe, "resources": summarize_resources(resource.stdout.splitlines())}
    return {"ranks": by_rank, "aggregate": aggregate_probe(by_rank)}


def cleanup_checkpoints(
    plan: dict[str, Any], _run_output: Path, _item: dict[str, Any],
    *, remote_run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    """Delete a run's node-local checkpoint directory once its metrics and read
    probes are already captured, so local NVMe isn't exhausted across repeats."""
    for rank in plan["ranks"]:
        checkpoint_dir = rank["env"].get("CHECKPOINT_DIR")
        if not checkpoint_dir:
            continue
        result = remote_run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", rank["host"],
             f"rm -rf -- {shlex.quote(checkpoint_dir)}"],
            check=False, capture_output=True, text=True, timeout=60,
        )
        if result.returncode:
            raise RuntimeError(f"checkpoint cleanup failed on rank {rank['rank']}: {result.stderr.strip()}")


def aggregate_probe(by_rank: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "logical_checkpoint_bytes": sum(item["checkpoint_io"]["inventory"]["logical_bytes"] for item in by_rank.values()),
        "allocated_checkpoint_bytes": sum(item["checkpoint_io"]["inventory"]["allocated_bytes"] for item in by_rank.values()),
        "host_mem_available_min_bytes": min(item["resources"]["mem_available_min_bytes"] for item in by_rank.values()),
        "host_memory_pressure_max_bytes": max(item["resources"]["host_memory_pressure_bytes"] for item in by_rank.values()),
        "stage_device_write_bytes": sum(item["resources"]["device_write_bytes_delta"] for item in by_rank.values()),
    }
    reads = {}
    for condition in ("warm_after_write", "cold_buffered", "warm_buffered", "direct"):
        values = [item["checkpoint_io"]["reads"].get(condition) for item in by_rank.values()]
        if any(value is None for value in values):
            reads[condition] = None
            continue
        elapsed = max(float(value["seconds"]) for value in values)
        logical = sum(int(value["logical_bytes"]) for value in values)
        physical = sum(int(value["physical_read_bytes"]) for value in values)
        reads[condition] = {
            "seconds_max_across_ranks": elapsed,
            "logical_bytes": logical,
            "physical_read_bytes": physical,
            "logical_bytes_per_second": logical / elapsed if elapsed > 0 else None,
            "physical_to_logical_ratio": physical / logical if logical else None,
            "rank_classifications": [value.get("classification") for value in values],
        }
    result["reads"] = reads
    return result


def relative_mad(values: list[float]) -> float | None:
    if not values:
        return None
    median = statistics.median(values)
    return statistics.median(abs(value - median) for value in values) / median if median else 0.0


def checkpoint_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    variants = sorted({item["variant"] for item in records if item.get("variant")})
    for variant in variants:
        selected = [
            item for item in records
            if item.get("status") == "passed" and not item.get("warmup")
            and item.get("variant") == variant and item.get("post_run")
        ]
        throughputs = [item["post_run"]["aggregate"]["reads"]["cold_buffered"]["logical_bytes_per_second"] for item in selected]
        throughputs = [float(value) for value in throughputs if value is not None]
        sizes = [float(item["post_run"]["aggregate"]["logical_checkpoint_bytes"]) for item in selected]
        save = [item["metrics"]["save_call_host_seconds_max_across_ranks"] for item in selected]
        save = [float(value) for value in save if value is not None]
        dispersion = relative_mad(save)
        summary[variant] = {
            "run_count": len(selected),
            "logical_checkpoint_bytes_median": statistics.median(sizes) if sizes else None,
            "cold_buffered_read_bytes_per_second_median": statistics.median(throughputs) if throughputs else None,
            "save_call_seconds_median": statistics.median(save) if save else None,
            "save_call_relative_mad": dispersion,
            "extend_to_eight_runs": dispersion is not None and dispersion > 0.1,
        }
    return summary


def generated_setup(setup: dict[str, Any], output: Path) -> Path:
    GENERATED.mkdir(parents=True, exist_ok=True)
    path = GENERATED / f"{output.name}-checkpoint-memory-setup.json"
    path.write_text(json.dumps(setup, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def memory_experiments() -> list[dict[str, Any]]:
    common = {
        "MODEL_ID": MODEL_ID, "MODEL_REVISION": MODEL_REVISION,
        "DATASET_ID": DATASET_ID, "DATASET_REVISION": DATASET_REVISION,
        "MAX_LENGTH": 2048, "PAD_TO_MAX_LENGTH": True,
        "RESOURCE_SAMPLING": True, "SEED": 42, "STAGE": "train",
    }
    megatron = {
        **common, "FINETUNING_MODE": "lora", "MAX_STEPS": 4,
        "SCHEDULE_STEPS": 4, "GLOBAL_BATCH_SIZE": 2, "MICRO_BATCH_SIZE": 1,
        "TP": 1, "PP": 1, "EP": 2, "SAVE_INTERVAL": 4,
        "CHECKPOINT_MODE": "sync", "SAVE_OPTIMIZER": True,
        "CHECKPOINT_PLACEMENT": "local", "MEASURE_TIMING": True,
    }
    trl = {
        **common, "FINETUNING_MODE": "lora", "OPTIMIZER": "adamw",
        "LEARNING_RATE": "2e-5", "MAX_STEPS": 4,
        "TRAIN_SAMPLES": 32, "EVAL_SAMPLES": 8,
        "GRADIENT_ACCUMULATION_STEPS": 1,
    }
    return [
        {"name": "len-4096", "repeats": 3, "pilot": True, "experiment": {
            "backend": "megatron", "setup": "spark", "nnodes": 2, "nproc_per_node": 1,
            "env": {**megatron, "MAX_LENGTH": 4096},
        }},
        {"name": "len-8192", "repeats": 3, "pilot": True, "experiment": {
            "backend": "megatron", "setup": "spark", "nnodes": 2, "nproc_per_node": 1,
            "env": {**megatron, "MAX_LENGTH": 8192},
        }},
        {"name": "trl-ddp", "repeats": 3, "pilot": False, "experiment": {
            "backend": "trl", "setup": "spark", "nnodes": 2, "nproc_per_node": 1,
            "env": {**trl, "DISTRIBUTED_BACKEND": "ddp"},
        }},
        {"name": "trl-fsdp2", "repeats": 3, "pilot": False, "experiment": {
            "backend": "trl", "setup": "spark", "nnodes": 2, "nproc_per_node": 1,
            "env": {**trl, "DISTRIBUTED_BACKEND": "fsdp2"},
        }},
        {"name": "trl-zero3-nvme", "repeats": 3, "pilot": False, "experiment": {
            "backend": "trl", "setup": "spark", "nnodes": 2, "nproc_per_node": 1,
            "env": {
                **trl, "DISTRIBUTED_BACKEND": "deepspeed",
                "DEEPSPEED_CONFIG": "configs/deepspeed-zero3-nvme.json",
                "FINETUNING_MODE": "full", "OPTIMIZER": "sgd", "MAX_STEPS": 1,
            },
        }},
    ]


def _write_experiment(experiment: dict[str, Any], name: str) -> Path:
    GENERATED.mkdir(parents=True, exist_ok=True)
    path = GENERATED / f"{name}.json"
    path.write_text(json.dumps(experiment, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def collect_memory_estimates(
    setup: dict[str, Any], *, execute: bool,
    remote_run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[dict[str, Any]]:
    node = setup["nodes"][0]
    script = str(Path(node["checkout"]) / "experiments" / "estimate_memory.py")
    model_dir = node["model_dirs"][MODEL_ID]
    specs = [
        ("meg-lora-2048", ["--backend", "megatron", "--finetuning-mode", "lora", "--optimizer", "adam", "--ep", "2", "--seq-len", "2048"]),
        ("meg-lora-4096", ["--backend", "megatron", "--finetuning-mode", "lora", "--optimizer", "adam", "--ep", "2", "--seq-len", "4096"]),
        ("meg-lora-8192", ["--backend", "megatron", "--finetuning-mode", "lora", "--optimizer", "adam", "--ep", "2", "--seq-len", "8192"]),
        ("trl-ddp-lora", ["--backend", "trl", "--finetuning-mode", "lora", "--optimizer", "adamw", "--distributed-backend", "ddp"]),
        ("trl-fsdp2-lora", ["--backend", "trl", "--finetuning-mode", "lora", "--optimizer", "adamw", "--distributed-backend", "fsdp2"]),
        ("trl-zero3-nvme-full-sgd", ["--backend", "trl", "--finetuning-mode", "full", "--optimizer", "sgd", "--distributed-backend", "deepspeed", "--zero-stage", "3", "--offload", "nvme"]),
        ("meg-full-sgd", ["--backend", "megatron", "--finetuning-mode", "full", "--optimizer", "sgd", "--ep", "2"]),
        ("meg-full-adam", ["--backend", "megatron", "--finetuning-mode", "full", "--optimizer", "adam", "--ep", "2"]),
    ]
    records = []
    for name, arguments in specs:
        command = " ".join(shlex.quote(value) for value in [
            node["python"]["megatron"], script, "--model-dir", model_dir,
            "--world-size", "2", "--micro-batch-size", "1", "--budget-gib", "119", "--json",
            *arguments,
        ])
        record: dict[str, Any] = {"name": name, "host": node["host"], "command": command}
        if execute:
            result = remote_run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", node["host"], command],
                check=False, capture_output=True, text=True, timeout=300,
            )
            if result.returncode:
                raise RuntimeError(f"memory estimate failed for {name}: {result.stderr.strip()}")
            record["result"] = json.loads(result.stdout)
        records.append(record)
    return records


def collect_run_resources(
    plan: dict[str, Any], output: Path,
    *, remote_run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    result = {}
    destination = output / "measurements"
    destination.mkdir(parents=True, exist_ok=True)
    for rank in plan["ranks"]:
        remote = str(Path(rank["output"]) / "measurements" / f"resources-node-{rank['rank']}.jsonl")
        fetched = remote_run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", rank["host"],
             f"cat -- {shlex.quote(remote)}"],
            check=False, capture_output=True, text=True, timeout=35,
        )
        if fetched.returncode:
            raise RuntimeError(f"cannot fetch resource samples for rank {rank['rank']}")
        (destination / f"resources-node-{rank['rank']}.jsonl").write_text(fetched.stdout, encoding="utf-8")
        result[str(rank["rank"])] = summarize_resources(fetched.stdout.splitlines())
    return result


def collect_trl_summary(
    plan: dict[str, Any], output: Path,
    *, remote_run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    rank = plan["ranks"][0]
    remote = str(Path(rank["output"]) / "summary-train.json")
    fetched = remote_run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", rank["host"],
         f"cat -- {shlex.quote(remote)}"],
        check=False, capture_output=True, text=True, timeout=35,
    )
    if fetched.returncode:
        raise RuntimeError("cannot fetch TRL training summary")
    summary = json.loads(fetched.stdout)
    (output / "summary-train.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


class _LocalFileResult:
    def __init__(self, stdout: str) -> None:
        self.returncode = 0
        self.stdout = stdout
        self.stderr = ""


def _local_file_run(paths: list[Path], fallback: Callable[..., Any] = subprocess.run) -> Callable[..., Any]:
    """A remote_run/run stand-in that replays already-fetched local files instead
    of SSHing, in call order. Used to resume a run whose metrics were already
    collected before an earlier invocation was interrupted. A run can itself be
    interrupted mid-collection (e.g. one rank's file fetched, the other not), so
    a missing local file falls back to the real fetch rather than failing resume."""
    iterator = iter(paths)

    def run_fn(*args: Any, **kwargs: Any) -> Any:
        path = next(iterator)
        if path.exists():
            return _LocalFileResult(path.read_text(encoding="utf-8"))
        return fallback(*args, **kwargs)

    return run_fn


def run_memory_matrix(
    setup: dict[str, Any], setup_path: Path, output: Path, *, execute: bool, timeout: int, resume: bool = False,
) -> dict[str, Any]:
    estimates = collect_memory_estimates(setup, execute=execute)
    if execute:
        output.mkdir(parents=True, exist_ok=resume)
    records = []
    for condition in memory_experiments():
        config_path = _write_experiment(condition["experiment"], f"{output.name}-{condition['name']}")
        experiment = run.load_experiment(config_path)
        count = condition["repeats"] + int(condition["pilot"])
        for index in range(count):
            pilot = bool(condition["pilot"] and index == 0)
            run_name = f"{output.name}--{condition['name']}--run-{index:02d}"
            run_output = output / "runs" / run_name
            plan = run.build_plan(setup, experiment, setup_path.resolve(), config_path.resolve(), run_output, ROOT)
            record: dict[str, Any] = {
                "condition": condition["name"], "run": index, "pilot": pilot,
                "status": "dry-run", "output": str(run_output), "plan": plan,
            }
            if execute:
                # A prior, interrupted invocation may have already completed and recorded
                # this exact run (run.execute writes its own per-run manifest.json on exit).
                # Resuming reads that run's already-fetched local metrics back instead of
                # re-running real 30B GPU work.
                local_manifest = run_output / "manifest.json"
                resumed = resume and local_manifest.exists()
                if resumed:
                    code = 0 if json.loads(local_manifest.read_text(encoding="utf-8")).get("status") == "passed" else 1
                else:
                    code = run.execute(plan, timeout, run_output)
                record.update(status="passed" if code == 0 else "failed", exit_code=code)
                if code == 0:
                    resource_paths = [run_output / "measurements" / f"resources-node-{r['rank']}.jsonl" for r in plan["ranks"]]
                    record["resources"] = collect_run_resources(
                        plan, run_output, remote_run=_local_file_run(resource_paths) if resumed else subprocess.run
                    )
                    if experiment["backend"] == "megatron":
                        if resumed:
                            metric_paths = [run_output / "measurements" / f"rank-{r['rank']}-train.jsonl" for r in plan["ranks"]]
                            record["metrics_by_rank"] = benchmarks.fetch_measurements(
                                plan, run_output, run=_local_file_run(metric_paths)
                            )
                        else:
                            record["metrics_by_rank"] = benchmarks.fetch_measurements(plan, run_output)
                            try:
                                cleanup_checkpoints(plan, run_output, condition)
                            except Exception as exc:
                                record["cleanup_error"] = str(exc)
                    else:
                        record["trl_summary"] = collect_trl_summary(
                            plan, run_output,
                            remote_run=_local_file_run([run_output / "summary-train.json"]) if resumed else subprocess.run,
                        )
                if pilot and code != 0:
                    records.append(record)
                    break
            records.append(record)
            if execute:
                (output / "manifest.json").write_text(
                    json.dumps({"status": "running", "estimates": estimates, "records": records},
                               indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
    status = "dry-run" if not execute else (
        "passed" if all(item["status"] == "passed" for item in records) else "completed-with-failures"
    )
    result = {"status": status, "estimates": estimates, "records": records}
    if execute:
        (output / "manifest.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--phase", choices=("checkpoint", "memory"), default="checkpoint")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--resume", action="store_true",
                         help="memory phase only: continue into an existing --output, reusing "
                              "already-completed runs instead of refusing or re-running them")
    args = parser.parse_args(argv)
    if args.resume and args.phase != "memory":
        raise SystemExit("--resume is only supported for --phase memory")
    try:
        if args.output.exists() and not args.resume:
            raise run.ConfigError(f"refusing to reuse output directory: {args.output}")
        setup = run.load_setup(args.setup)
        setup, cohort_plan = prepare_cohorts(setup, execute=args.execute)
        setup_path = generated_setup(setup, args.output)
        if args.phase == "memory":
            result = run_memory_matrix(setup, setup_path, args.output, execute=args.execute,
                                        timeout=args.timeout, resume=args.resume)
        else:
            result = benchmarks.run_benchmark(
                setup_path=setup_path,
                benchmark_path=args.plan,
                output=args.output,
                execute=args.execute,
                steps=8,
                repeats=args.repeats,
                within_run_warmup=2,
                checkpoint_intervals=[2, 2],
                timeout=args.timeout,
                post_run_fn=collect_post_run if args.execute else None,
                cleanup_fn=cleanup_checkpoints if args.execute else None,
            )
        result["cohorts"] = cohort_plan
        if args.execute and args.phase == "checkpoint":
            result["checkpoint_summary"] = checkpoint_summary(result["records"])
            (args.output / "manifest.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
    except (OSError, ValueError, RuntimeError, run.ConfigError, subprocess.SubprocessError) as exc:
        print(f"experiment error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") in {"dry-run", "passed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
