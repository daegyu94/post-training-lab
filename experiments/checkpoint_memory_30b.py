"""Prepare UltraChat and run the local-NVMe 30B checkpoint or memory experiment."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import benchmarks, run  # noqa: E402

MODELS = {
    "qwen": {
        "id": "Qwen/Qwen3-30B-A3B",
        "revision": "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
        "cohort": "ultrachat-qwen3-30b-2048-v1",
        "base_experiment": "experiments/megatron/qwen3-30b-lora.json",
        "megatron_env": {"TRANSFORMER_IMPL": "transformer_engine"},
    },
    "glm": {
        "id": "zai-org/GLM-4.7-Flash",
        "revision": "7dd20894a642a0aa287e9827cb1a1f7f91386b67",
        "cohort": "ultrachat-glm-4.7-flash-2048-v1",
        "base_experiment": "experiments/megatron/glm-4.7-flash-30b-lora.json",
        "megatron_env": {"TRANSFORMER_IMPL": "transformer_engine"},
    },
}
DEFAULT_MODEL = "qwen"
MODEL_ID = MODELS[DEFAULT_MODEL]["id"]
MODEL_REVISION = MODELS[DEFAULT_MODEL]["revision"]
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_REVISION = "8049631c405ae6576f93f445c6b8166f76f5505a"
DEFAULT_PLAN = ROOT / "experiments" / "megatron" / "checkpoint-memory-30b.json"
GENERATED = ROOT / "experiments" / "generated"


def resolve_checkpoint_plan(model: str, path: Path | None) -> Path:
    """Keep the plan's effective model/data identity aligned with its cohort."""
    if path is None:
        path = DEFAULT_PLAN
    benchmark = benchmarks.load_benchmark_plan(path)
    base = run.load_experiment(ROOT / MODELS[model]["base_experiment"])
    common = {**base["env"], **benchmark.get("common_env", {})}
    expected = {
        "MODEL_ID": MODELS[model]["id"], "MODEL_REVISION": MODELS[model]["revision"],
        "DATASET_ID": DATASET_ID, "DATASET_REVISION": DATASET_REVISION,
    }
    for cell in benchmark["cells"]:
        for variant in cell["variants"]:
            effective = {**common, **variant.get("env", {})}
            for key, value in expected.items():
                if effective.get(key) != value:
                    raise run.ConfigError(
                        f"checkpoint plan {path}: {cell['name']}/{variant['name']} {key} "
                        f"does not match --model {model} and its cohort"
                    )
    return path


def validate_resume_model(output: Path, model: str) -> None:
    """Do not relabel another model's saved runs when --resume is used."""
    manifest = output / "manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if previous.get("model") not in (None, MODELS[model]["id"]):
            raise run.ConfigError("--resume model differs from the existing output; use a new --output")
    # A controller interruption can leave only per-run manifests. Older root
    # manifests also lack 'model', so verify their recorded effective env too.
    for manifest in (output / "runs").glob("*/manifest.json"):
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        ranks = previous.get("ranks", [])
        expected = {f"MODEL_ID={MODELS[model]['id']}", f"MODEL_REVISION={MODELS[model]['revision']}"}
        # The runner stores the exported command, not the original rank env.
        if not ranks or any(
            not expected.issubset({token.rstrip(';') for token in shlex.split(rank.get("command", ""))})
            for rank in ranks
        ):
            raise run.ConfigError(f"--resume model identity differs or is missing in {manifest}; use a new --output")


def _output_root(node: dict[str, Any], backend: str) -> str:
    value = node["output_root"]
    return value[backend] if isinstance(value, dict) else value


def cohort_path(node: dict[str, Any], model: str = DEFAULT_MODEL, max_length: int = 2048) -> str:
    """Cohort directory for one model. Row selection depends on the tokenizer, so
    each model gets its own cohort rather than sharing Qwen's selection."""
    name = MODELS[model]["cohort"].replace("-2048-v1", f"-{max_length}-v1")
    return str(Path(_output_root(node, "megatron")) / "cohorts" / name)


def _cohort_command(
    node: dict[str, Any], output: str, model: str = DEFAULT_MODEL, max_length: int = 2048
) -> str:
    q = shlex.quote
    manifest = str(Path(output) / "manifest.json")
    script = str(Path(node["checkout"]) / "experiments" / "prepare_checkpoint_cohort.py")
    model_dir = node["model_dirs"][MODELS[model]["id"]]
    create = " ".join([
        q(node["python"]["megatron"]), q(script), "--source-dir", q(node["data_dir"]),
        "--output-dir", q(output), "--model-dir", q(model_dir), "--model-revision", MODELS[model]["revision"],
        "--dataset-id", DATASET_ID, "--dataset-revision", DATASET_REVISION,
        "--max-length", str(max_length), "--train-count", "32", "--eval-count", "8",
    ])
    return f"if test -f {q(manifest)}; then cat {q(manifest)}; else {create}; fi"


def prepare_cohorts(
    setup: dict[str, Any],
    *,
    execute: bool,
    model: str = DEFAULT_MODEL,
    max_length: int = 2048,
    remote_run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    updated = json.loads(json.dumps(setup))
    planned = []
    manifests = []
    for node in updated["nodes"]:
        if MODELS[model]["id"] not in node["model_dirs"]:
            raise run.ConfigError(f"node {node['host']} has no {MODELS[model]['id']} model path")
        destination = cohort_path(node, model, max_length)
        command = _cohort_command(node, destination, model, max_length)
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
            or manifest.get("model_revision") != MODELS[model]["revision"]
            or manifest.get("max_length") != max_length
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
    swap_used = [item["swaptotal_bytes"] - item["swapfree_bytes"] for item in samples]
    return {
        "sample_count": len(samples),
        "elapsed_seconds": elapsed,
        "device_major_minor": first["device_major_minor"],
        "mem_total_bytes": first["memtotal_bytes"],
        "mem_available_min_bytes": min(item["memavailable_bytes"] for item in samples),
        "mem_available_min_fraction": min(item["memavailable_bytes"] for item in samples) / first["memtotal_bytes"],
        "mem_available_first_bytes": first["memavailable_bytes"],
        "host_memory_pressure_bytes": first["memavailable_bytes"] - min(item["memavailable_bytes"] for item in samples),
        "swap_used_first_bytes": swap_used[0],
        "swap_used_max_bytes": max(swap_used),
        "swap_used_peak_increase_bytes": max(0, max(swap_used) - swap_used[0]),
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


def cleanup_checkpoints(
    plan: dict[str, Any], _run_output: Path, _item: dict[str, Any],
    *, remote_run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    """Delete node-local checkpoints after their metrics and inventory are captured."""
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


def collect_post_run(
    plan: dict[str, Any], output: Path, _item: dict[str, Any],
    *, remote_run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    resources = collect_run_resources(plan, output, remote_run=remote_run)
    by_rank: dict[str, Any] = {}
    for rank in plan["ranks"]:
        script = str(Path(rank["checkout"]) / "experiments" / "checkpoint_io_probe.py")
        command = " ".join(shlex.quote(value) for value in [
            rank["env"]["PYTHON"], script, "--checkpoint-dir",
            rank["env"]["CHECKPOINT_DIR"], "--inventory-only",
        ])
        result = remote_run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", rank["host"], command],
            check=False, capture_output=True, text=True, timeout=60,
        )
        if result.returncode:
            raise RuntimeError(f"checkpoint inventory failed on rank {rank['rank']}: {result.stderr.strip()}")
        by_rank[str(rank["rank"])] = {
            "inventory": json.loads(result.stdout), "resources": resources[str(rank["rank"])],
        }
    return {
        "scope": "node-local checkpoint write only; no read or restore",
        "by_rank": by_rank,
        "aggregate": {
            "logical_checkpoint_bytes": sum(item["inventory"]["logical_bytes"] for item in by_rank.values()),
            "allocated_checkpoint_bytes": sum(item["inventory"]["allocated_bytes"] for item in by_rank.values()),
            "checkpoint_file_count": sum(item["inventory"]["file_count"] for item in by_rank.values()),
            "host_mem_available_min_bytes": min(item["resources"]["mem_available_min_bytes"] for item in by_rank.values()),
            "stage_device_write_bytes": sum(item["resources"]["device_write_bytes_delta"] for item in by_rank.values()),
        },
    }


def reclaim_remote(
    plan: dict[str, Any], *, remote_run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    """Kill any orphaned training process and clear the remote output directory left
    behind by an interrupted run, so a retry can claim that run's output fresh (the
    session/rank claim in run.execute otherwise refuses a directory that already
    exists). Best-effort: a controller crash means there is nothing local to compare
    against, so failures here are not fatal -- a stale mkdir claim just means the
    retry itself fails loudly instead of silently reusing bad state."""
    for rank in plan["ranks"]:
        remote_run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", rank["host"],
             f"pkill -9 -f -- {shlex.quote(rank['output'])} 2>/dev/null; "
             f"rm -rf -- {shlex.quote(rank['output'])}"],
            check=False, capture_output=True, text=True, timeout=60,
        )


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
        sizes = [float(item["post_run"]["aggregate"]["logical_checkpoint_bytes"]) for item in selected]
        save = [item["metrics"]["save_call_host_seconds_max_across_ranks"] for item in selected]
        save = [float(value) for value in save if value is not None]
        host_blocking = []
        for item in selected:
            enqueue = item["metrics"].get("save_call_host_seconds_max_across_ranks")
            finalize = item["metrics"].get("blocking_finalization_host_seconds_max_across_ranks") or 0
            if enqueue is not None:
                host_blocking.append(float(enqueue) + float(finalize))
        dispersion = relative_mad(save)
        summary[variant] = {
            "run_count": len(selected),
            "logical_checkpoint_bytes_median": statistics.median(sizes) if sizes else None,
            "save_call_seconds_median": statistics.median(save) if save else None,
            "save_call_relative_mad": dispersion,
            "save_plus_finalization_host_seconds_median": (
                statistics.median(host_blocking) if host_blocking else None
            ),
            "extend_to_eight_runs": dispersion is not None and dispersion > 0.1,
        }
    return summary


def generated_setup(setup: dict[str, Any], output: Path) -> Path:
    GENERATED.mkdir(parents=True, exist_ok=True)
    path = GENERATED / f"{output.name}-checkpoint-memory-setup.json"
    path.write_text(json.dumps(setup, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def memory_experiments(model: str = DEFAULT_MODEL) -> list[dict[str, Any]]:
    common = {
        "MODEL_ID": MODELS[model]["id"], "MODEL_REVISION": MODELS[model]["revision"],
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
        **MODELS[model]["megatron_env"],
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
        {"name": "trl-fsdp2-dcp", "repeats": 3, "pilot": False, "experiment": {
            "backend": "trl", "setup": "spark", "nnodes": 2, "nproc_per_node": 1,
            "env": {
                **trl, "DISTRIBUTED_BACKEND": "fsdp2", "MAX_STEPS": 1,
                "MAX_LENGTH": 512, "TRAIN_SAMPLES": 4, "EVAL_SAMPLES": 1,
            },
        }},
        {"name": "trl-zero3-nvme", "repeats": 3, "pilot": False, "experiment": {
            "backend": "trl", "setup": "spark", "nnodes": 2, "nproc_per_node": 1,
            "env": {
                **trl, "DISTRIBUTED_BACKEND": "deepspeed",
                "DEEPSPEED_CONFIG": "configs/deepspeed-zero3-nvme.json",
                "FINETUNING_MODE": "full", "OPTIMIZER": "adamw", "MAX_STEPS": 1,
            },
        }},
    ]


def _write_experiment(experiment: dict[str, Any], name: str) -> Path:
    GENERATED.mkdir(parents=True, exist_ok=True)
    path = GENERATED / f"{name}.json"
    path.write_text(json.dumps(experiment, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def collect_memory_estimates(
    setup: dict[str, Any], *, execute: bool, model: str = DEFAULT_MODEL,
    remote_run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[dict[str, Any]]:
    node = setup["nodes"][0]
    script = str(Path(node["checkout"]) / "experiments" / "estimate_memory.py")
    model_dir = node["model_dirs"][MODELS[model]["id"]]
    specs = [
        ("meg-lora-2048", ["--backend", "megatron", "--finetuning-mode", "lora", "--optimizer", "adam", "--ep", "2", "--seq-len", "2048"]),
        ("meg-lora-4096", ["--backend", "megatron", "--finetuning-mode", "lora", "--optimizer", "adam", "--ep", "2", "--seq-len", "4096"]),
        ("meg-lora-8192", ["--backend", "megatron", "--finetuning-mode", "lora", "--optimizer", "adam", "--ep", "2", "--seq-len", "8192"]),
        ("trl-ddp-lora", ["--backend", "trl", "--finetuning-mode", "lora", "--optimizer", "adamw", "--distributed-backend", "ddp"]),
        ("trl-fsdp2-lora", ["--backend", "trl", "--finetuning-mode", "lora", "--optimizer", "adamw", "--distributed-backend", "fsdp2"]),
        ("trl-zero3-nvme-full-adamw", ["--backend", "trl", "--finetuning-mode", "full", "--optimizer", "adamw", "--distributed-backend", "deepspeed", "--zero-stage", "3", "--offload", "nvme"]),
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


def collect_trl_dcp(
    plan: dict[str, Any], output: Path,
    *, remote_run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    by_rank = {}
    destination = output / "measurements"
    destination.mkdir(parents=True, exist_ok=True)
    for rank in plan["ranks"]:
        checkpoint = str(Path(rank["output"]) / "adapter")
        script = str(Path(rank["checkout"]) / "experiments" / "checkpoint_io_probe.py")
        command = " ".join(shlex.quote(value) for value in [
            rank["env"]["PYTHON"], script, "--checkpoint-dir", checkpoint, "--inventory-only",
        ])
        fetched = remote_run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", rank["host"], command],
            check=False, capture_output=True, text=True, timeout=60,
        )
        if fetched.returncode:
            raise RuntimeError(f"cannot inventory TRL DCP on rank {rank['rank']}: {fetched.stderr.strip()}")
        inventory = json.loads(fetched.stdout)
        names = {item["name"] for item in inventory["files"]}
        if not any(name.endswith(".distcp") for name in names):
            raise RuntimeError(f"TRL FSDP2 output on rank {rank['rank']} has no PyTorch DCP shard")
        path = destination / f"trl-dcp-inventory-rank-{rank['rank']}.json"
        path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        by_rank[str(rank["rank"])] = inventory
    if not any(
        item["name"].endswith(".metadata")
        for inventory in by_rank.values() for item in inventory["files"]
    ):
        raise RuntimeError("TRL FSDP2 output has no PyTorch DCP metadata")
    return {
        "scope": "TRL FSDP2 model-only PyTorch DCP; FileSystemWriter sync_files=True default",
        "by_rank": by_rank,
        "logical_checkpoint_bytes": sum(item["logical_bytes"] for item in by_rank.values()),
        "allocated_checkpoint_bytes": sum(item["allocated_bytes"] for item in by_rank.values()),
    }


def cleanup_trl_checkpoint(
    plan: dict[str, Any],
    *, remote_run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    for rank in plan["ranks"]:
        checkpoint = str(Path(rank["output"]) / "adapter")
        result = remote_run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", rank["host"],
             f"rm -rf -- {shlex.quote(checkpoint)}"],
            check=False, capture_output=True, text=True, timeout=60,
        )
        if result.returncode:
            raise RuntimeError(f"TRL checkpoint cleanup failed on rank {rank['rank']}: {result.stderr.strip()}")


def _prior_terminal_status(resume: bool, local_manifest: Path) -> str | None:
    """The prior invocation's status for this run, if it reached a terminal state
    (passed/failed). None means either no prior attempt or one interrupted mid-run
    (stuck at "running"), both of which require executing this run fresh."""
    if not resume or not local_manifest.exists():
        return None
    status = json.loads(local_manifest.read_text(encoding="utf-8")).get("status")
    return status if status in ("passed", "failed") else None


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
    conditions: list[str] | None = None, model: str = DEFAULT_MODEL, repeats: int | None = None,
) -> dict[str, Any]:
    estimates = collect_memory_estimates(setup, execute=execute, model=model)
    if execute:
        output.mkdir(parents=True, exist_ok=resume)
    records = []
    selected_conditions = memory_experiments(model)
    if conditions is not None:
        available = {item["name"] for item in selected_conditions}
        unknown = set(conditions) - available
        if unknown:
            raise run.ConfigError(f"unknown --condition value(s): {', '.join(sorted(unknown))}")
        selected_conditions = [item for item in selected_conditions if item["name"] in conditions]
    for condition in selected_conditions:
        config_path = _write_experiment(condition["experiment"], f"{output.name}-{condition['name']}")
        experiment = run.load_experiment(config_path)
        count = (repeats if repeats is not None else condition["repeats"]) + int(condition["pilot"])
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
                # re-running real 30B GPU work. A manifest stuck at "running" means the
                # *previous* invocation was itself interrupted mid-run (not a real pass or
                # fail) -- that local state and any orphaned remote process/directory can't
                # be trusted, so reclaim them and execute fresh rather than recording a
                # false failure.
                local_manifest = run_output / "manifest.json"
                prior_status = _prior_terminal_status(resume, local_manifest)
                resumed = prior_status is not None
                if resumed:
                    code = 0 if prior_status == "passed" else 1
                else:
                    if resume and local_manifest.exists():
                        reclaim_remote(plan)
                        shutil.rmtree(run_output)
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
                        if condition["name"] == "trl-fsdp2-dcp":
                            record["trl_dcp"] = collect_trl_dcp(plan, run_output)
                            cleanup_trl_checkpoint(plan)
                if pilot and code != 0:
                    records.append(record)
                    break
            records.append(record)
            if execute:
                (output / "manifest.json").write_text(
                    json.dumps({"status": "running", "model": MODELS[model]["id"],
                                "estimates": estimates, "records": records},
                               indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
    status = "dry-run" if not execute else (
        "passed" if all(item["status"] == "passed" for item in records) else "completed-with-failures"
    )
    result = {"status": status, "model": MODELS[model]["id"], "estimates": estimates, "records": records}
    if execute:
        (output / "manifest.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", type=Path, required=True)
    parser.add_argument("--plan", type=Path,
                        help="checkpoint plan override; default follows --model")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--phase", choices=("checkpoint", "memory"), default="checkpoint")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--within-run-warmup", type=int)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--resume", action="store_true",
                         help="memory phase only: continue into an existing --output, reusing "
                              "already-completed runs instead of refusing or re-running them")
    parser.add_argument("--model", choices=sorted(MODELS), default=DEFAULT_MODEL,
                         help="which verified 30B model to measure (default: qwen)")
    parser.add_argument("--condition", action="append",
                         help="memory phase only: run only this condition name (repeatable). "
                              "Default runs the full matrix from memory_experiments().")
    args = parser.parse_args(argv)
    if args.resume and args.phase != "memory":
        raise SystemExit("--resume is only supported for --phase memory")
    if args.condition and args.phase != "memory":
        raise SystemExit("--condition is only supported for --phase memory")
    try:
        if args.output.exists() and not args.resume:
            raise run.ConfigError(f"refusing to reuse output directory: {args.output}")
        positive = (args.repeats, args.steps, args.checkpoint_interval, args.timeout)
        if any(value is not None and value < 1 for value in positive):
            raise run.ConfigError("counts must be positive and --within-run-warmup non-negative")
        if args.within_run_warmup is not None and args.within_run_warmup < 0:
            raise run.ConfigError("counts must be positive and --within-run-warmup non-negative")
        if args.resume:
            validate_resume_model(args.output, args.model)
        if args.phase == "checkpoint":
            args.plan = resolve_checkpoint_plan(args.model, args.plan)
        setup = run.load_setup(args.setup)
        if args.phase == "checkpoint":
            cohort_max_length = int(
                benchmarks.load_benchmark_plan(args.plan).get("common_env", {}).get("MAX_LENGTH", 2048)
            )
        else:
            cohort_max_length = 512 if args.condition == ["trl-fsdp2-dcp"] else 2048
        setup, cohort_plan = prepare_cohorts(
            setup, execute=args.execute, model=args.model, max_length=cohort_max_length
        )
        setup_path = generated_setup(setup, args.output)
        if args.phase == "memory":
            result = run_memory_matrix(setup, setup_path, args.output, execute=args.execute,
                                        timeout=args.timeout, resume=args.resume, conditions=args.condition,
                                        model=args.model, repeats=args.repeats)
        else:
            result = benchmarks.run_benchmark(
                setup_path=setup_path,
                benchmark_path=args.plan,
                base_experiment_path=ROOT / MODELS[args.model]["base_experiment"],
                output=args.output,
                execute=args.execute,
                steps=args.steps,
                repeats=args.repeats,
                within_run_warmup=args.within_run_warmup,
                checkpoint_intervals=(
                    [args.checkpoint_interval, args.checkpoint_interval]
                    if args.checkpoint_interval is not None else None
                ),
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
