"""Validate and optionally run a bounded Spark experiment on configured nodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import uuid
from typing import Any, Callable


REVISION_RE = re.compile(r"^[0-9a-fA-F]{40}$")
ENV_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
HARDWARE_ENV = ("NCCL", "OMP", "HF_", "TOKENIZERS_")
HARDWARE_EXACT = {"PYTHON_HEADERS", "CPATH"}
COMMON_RESERVED = {
    "NODE_RANK", "PYTHON", "OUTPUT_DIR", "NNODES", "NPROC_PER_NODE",
    "MASTER_ADDR", "MASTER_PORT", "MODEL_DIR", "DATA_DIR",
    "OBSERVATORY_RUN_ID", "FRAMEWORK_METRICS_DIR", "CHECKPOINT_DIR",
}
TRL_ENV = {
    "MODEL_ID", "MODEL_REVISION", "DATASET_ID", "DATASET_REVISION",
    "FINETUNING_MODE", "OPTIMIZER", "LEARNING_RATE", "MAX_STEPS", "NUM_TRAIN_EPOCHS",
    "MAX_LENGTH", "GRADIENT_ACCUMULATION_STEPS", "DISTRIBUTED_BACKEND",
    "FSDP2", "DEEPSPEED_CONFIG", "STAGE", "TRAIN_SAMPLES", "EVAL_SAMPLES", "SEED",
}
MEGATRON_ENV = {
    "MODEL_ID", "MODEL_REVISION", "DATASET_ID", "DATASET_REVISION",
    "FINETUNING_MODE", "CHECKPOINT_MODE", "RECOMPUTE", "MAX_STEPS", "NUM_TRAIN_EPOCHS",
    "SCHEDULE_STEPS", "EVAL_ITERS", "MAX_LENGTH", "PAD_TO_MAX_LENGTH", "GLOBAL_BATCH_SIZE",
    "MICRO_BATCH_SIZE", "TP", "PP", "EP", "SEED", "SAVE_INTERVAL",
    "DISTRIBUTED_OPTIMIZER", "SAVE_OPTIMIZER", "LOAD_OPTIMIZER",
    "FULLY_PARALLEL_SAVE", "FULLY_PARALLEL_LOAD", "SEQUENCE_PARALLEL",
    "OVERLAP_GRAD_REDUCE", "DIST_CKPT_OPTIM_FULLY_RESHARDABLE",
    "RESUME_AFTER_TRAIN", "RESUME_MAX_STEPS", "RESUME_TP", "RESUME_PP", "RESUME_EP",
    "TRANSFORMER_IMPL", "MEASURE_TIMING", "STAGE",
}


class ConfigError(ValueError):
    """Raised when setup or experiment input violates the runner contract."""


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read JSON config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"config must contain a JSON object: {path}")
    return value


def _keys(value: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigError(f"{label} has unsupported fields: {', '.join(unknown)}")


def _scalar_env(value: Any, label: str) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        return str(value)
    raise ConfigError(f"{label} environment values must be scalar")


def _hardware_env(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ConfigError(f"{label} must be an object")
    result: dict[str, str] = {}
    for key, raw in value.items():
        if not isinstance(key, str) or not ENV_NAME_RE.fullmatch(key) or not (key.startswith(HARDWARE_ENV) or key in HARDWARE_EXACT):
            raise ConfigError(f"{label} contains non-hardware variable: {key}")
        result[key] = _scalar_env(raw, f"{label}.{key}")
    return result


def load_setup(path: Path) -> dict[str, Any]:
    value = _json(path)
    _keys(value, {"setup", "nodes", "master_addr", "master_port", "env"}, "setup")
    if value.get("setup") != "spark":
        raise ConfigError("setup.setup must be 'spark'")
    nodes = value.get("nodes")
    if not isinstance(nodes, list) or not nodes or len(nodes) > 2:
        raise ConfigError("setup.nodes must contain one or two nodes")
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise ConfigError(f"setup.nodes[{index}] must be an object")
        _keys(node, {"host", "checkout", "python", "env", "model_dirs", "data_dir", "output_root"}, f"node[{index}]")
        for field in ("host", "checkout", "data_dir"):
            if not isinstance(node.get(field), str) or not node[field]:
                raise ConfigError(f"node[{index}].{field} must be a non-empty string")
        output_root = node.get("output_root")
        if isinstance(output_root, dict):
            if set(output_root) != {"trl", "megatron"} or any(
                not isinstance(item, str) or not item for item in output_root.values()
            ):
                raise ConfigError(f"node[{index}].output_root must map trl and megatron to paths")
        elif not isinstance(output_root, str) or not output_root:
            raise ConfigError(f"node[{index}].output_root must be a path or backend path map")
        if node["host"].startswith("-"):
            raise ConfigError(f"node[{index}].host cannot start with '-'")
        python = node.get("python")
        if not isinstance(python, dict) or set(python) != {"trl", "megatron"}:
            raise ConfigError(f"node[{index}].python must map trl and megatron")
        if any(not isinstance(item, str) or not item for item in python.values()):
            raise ConfigError(f"node[{index}].python values must be non-empty strings")
        models = node.get("model_dirs")
        if not isinstance(models, dict) or any(not isinstance(k, str) or not isinstance(v, str) or not v for k, v in models.items()):
            raise ConfigError(f"node[{index}].model_dirs must map model IDs to paths")
        node["env"] = _hardware_env(node.get("env", {}), f"node[{index}].env")
    value["env"] = _hardware_env(value.get("env", {}), "setup.env")
    if not isinstance(value.get("master_addr"), str) or not value["master_addr"]:
        raise ConfigError("setup.master_addr must be a non-empty string")
    try:
        port = int(value.get("master_port", 29500))
    except (TypeError, ValueError) as exc:
        raise ConfigError("setup.master_port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise ConfigError("setup.master_port must be in 1..65535")
    value["master_port"] = port
    return value


def _positive(env: dict[str, str], name: str, default: int) -> int:
    try:
        value = int(env.get(name, str(default)))
    except ValueError as exc:
        raise ConfigError(f"experiment env {name} must be an integer") from exc
    if value < 1:
        raise ConfigError(f"experiment env {name} must be at least 1")
    return value


def load_experiment(path: Path) -> dict[str, Any]:
    value = _json(path)
    _keys(value, {"backend", "setup", "nnodes", "nproc_per_node", "env"}, "experiment")
    backend = value.get("backend")
    if backend not in {"trl", "megatron"}:
        raise ConfigError("experiment.backend must be 'trl' or 'megatron'")
    if value.get("setup") != "spark":
        raise ConfigError("experiment.setup must be 'spark'")
    try:
        nnodes = int(value.get("nnodes"))
        nproc = int(value.get("nproc_per_node"))
    except (TypeError, ValueError) as exc:
        raise ConfigError("experiment nnodes and nproc_per_node must be integers") from exc
    if nnodes not in {1, 2}:
        raise ConfigError("experiment.nnodes must be 1 or 2")
    if nproc != 1:
        raise ConfigError("experiment.nproc_per_node must be exactly 1")
    env = value.get("env")
    if not isinstance(env, dict):
        raise ConfigError("experiment.env must be an object")
    allowed = TRL_ENV if backend == "trl" else MEGATRON_ENV
    for key in env:
        if not isinstance(key, str) or not ENV_NAME_RE.fullmatch(key) or key in COMMON_RESERVED:
            raise ConfigError(f"experiment.env cannot override runner-owned variable: {key}")
        if key not in allowed:
            raise ConfigError(f"experiment.env.{key} is unsupported for backend {backend}")
    value["env"] = {key: _scalar_env(raw, f"experiment.env.{key}") for key, raw in env.items()}
    for name in ("MODEL_ID", "MODEL_REVISION", "DATASET_ID", "DATASET_REVISION"):
        if not value["env"].get(name):
            raise ConfigError(f"experiment.env.{name} is required")
    for name in ("MODEL_REVISION", "DATASET_REVISION"):
        if not REVISION_RE.fullmatch(value["env"][name]):
            raise ConfigError(f"experiment.env.{name} must be a 40-hex immutable revision")
    if backend == "trl":
        distributed = value["env"].get("DISTRIBUTED_BACKEND", "ddp")
        if value["env"].get("FSDP2", "false").lower() == "true":
            if "DISTRIBUTED_BACKEND" in value["env"] and distributed != "fsdp2":
                raise ConfigError("TRL FSDP2=true conflicts with DISTRIBUTED_BACKEND")
            distributed = "fsdp2"
        stage = value["env"].get("STAGE", "all")
        if distributed not in {"ddp", "fsdp2", "deepspeed"}:
            raise ConfigError("TRL DISTRIBUTED_BACKEND must be ddp, fsdp2, or deepspeed")
        if stage not in {"all", "base", "train", "tuned"}:
            raise ConfigError("TRL STAGE must be all, base, train, or tuned")
        if distributed == "fsdp2" and stage in {"all", "tuned"}:
            raise ConfigError("TRL fsdp2 only supports STAGE=base or STAGE=train")
        if distributed == "deepspeed" and not value["env"].get("DEEPSPEED_CONFIG"):
            raise ConfigError("TRL deepspeed requires DEEPSPEED_CONFIG")
        if (
            distributed == "deepspeed"
            and Path(value["env"].get("DEEPSPEED_CONFIG", "")).name == "deepspeed-zero3-nvme.json"
            and value["env"].get("FINETUNING_MODE", "lora") == "lora"
        ):
            raise ConfigError("TRL LoRA with NVMe offload is unsupported; use DDP or full fine-tuning")
    else:
        stage = value["env"].get("STAGE", "all")
        if stage not in {"all", "base", "train", "tuned"}:
            raise ConfigError("Megatron STAGE must be all, base, train, or tuned")
        tp = _positive(value["env"], "TP", 1)
        pp = _positive(value["env"], "PP", 1)
        ep = _positive(value["env"], "EP", 2)
        mbs = _positive(value["env"], "MICRO_BATCH_SIZE", 1)
        gbs = _positive(value["env"], "GLOBAL_BATCH_SIZE", 8)
        world = nnodes * nproc
        model_parallel = tp * pp
        expert_parallel = pp * ep
        if world % model_parallel or world % expert_parallel:
            raise ConfigError("Megatron world size must be divisible by TP*PP and PP*EP")
        dp = world // model_parallel
        if gbs % (mbs * dp):
            raise ConfigError("Megatron GLOBAL_BATCH_SIZE must be divisible by MICRO_BATCH_SIZE*DP")
        if value["env"].get("SEQUENCE_PARALLEL", "false").lower() == "true":
            if tp < 2 or value["env"].get("TRANSFORMER_IMPL") != "transformer_engine":
                raise ConfigError("Megatron sequence parallel requires TP>=2 and TRANSFORMER_IMPL=transformer_engine")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _commit(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _dirty(root: Path) -> str:
    try:
        return ";".join(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines())
    except (OSError, subprocess.CalledProcessError):
        return ""


def _safe_run_id(output: Path) -> str:
    name = output.name
    if not name or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ConfigError("output directory name must contain only letters, digits, '.', '_' or '-'")
    return name


def build_plan(setup: dict[str, Any], experiment: dict[str, Any], setup_path: Path, experiment_path: Path, output: Path, root: Path) -> dict[str, Any]:
    if len(setup["nodes"]) < experiment["nnodes"]:
        raise ConfigError("setup has fewer nodes than experiment.nnodes")
    model_id = experiment["env"]["MODEL_ID"]
    ranks = []
    run_id = _safe_run_id(output)
    hosts: set[str] = set()
    master_addr = setup["master_addr"] if experiment["nnodes"] == 2 else "127.0.0.1"
    for rank, node in enumerate(setup["nodes"][: experiment["nnodes"]]):
        if node["host"] in hosts:
            raise ConfigError("each participating node must have a distinct host")
        hosts.add(node["host"])
        if model_id not in node["model_dirs"]:
            raise ConfigError(f"node[{rank}] has no model_dirs entry for {model_id}")
        output_root = node["output_root"]
        if isinstance(output_root, dict):
            output_root = output_root[experiment["backend"]]
        remote_output = os.path.join(output_root, run_id)
        env = dict(setup["env"])
        env.update(node["env"])
        env.update(experiment["env"])
        env.update({
            "NNODES": str(experiment["nnodes"]), "NPROC_PER_NODE": "1", "NODE_RANK": str(rank),
            "MASTER_ADDR": master_addr, "MASTER_PORT": str(setup["master_port"]),
            "PYTHON": node["python"][experiment["backend"]], "MODEL_DIR": node["model_dirs"][model_id],
            "DATA_DIR": node["data_dir"], "OUTPUT_DIR": remote_output,
            "OBSERVATORY_RUN_ID": run_id,
        })
        if experiment["backend"] == "megatron":
            env["CHECKPOINT_DIR"] = os.path.join(
                node["checkout"], "artifacts", "checkpoints", run_id
            )
        ranks.append({"rank": rank, "host": node["host"], "checkout": node["checkout"], "output": remote_output, "env": env})
    return {
        "backend": experiment["backend"], "setup": "spark", "run_id": run_id,
        "session_id": uuid.uuid4().hex,
        "setup_path": str(setup_path), "experiment_path": str(experiment_path), "output": str(output),
        "config_sha256": {"setup": _sha(setup_path), "experiment": _sha(experiment_path)},
        "controller_commit": _commit(root), "controller_dirty": _dirty(root), "ranks": ranks,
    }


def remote_command(rank: dict[str, Any], backend: str, timeout: int, run_id: str, expected_commit: str = "unknown", session_id: str | None = None) -> tuple[str, str]:
    q = shlex.quote
    exports = "; ".join(f"export {key}={q(value)}" for key, value in sorted(rank["env"].items()))
    export_clause = f"{exports}; " if exports else ""
    backend_root = os.path.join(rank["checkout"], "backends", backend)
    script = "./scripts/run_spark_cluster.sh"
    pidfile = os.path.join(rank["output"], f".runner-{run_id}-rank{rank['rank']}.pid")
    session_id = session_id or run_id
    owner_dir = os.path.join(rank["output"], f".runner-session-{run_id}")
    session_claim = os.path.join(owner_dir, session_id)
    rank_claim = os.path.join(rank["output"], f".runner-rank-{run_id}-{rank['rank']}")
    cancel_file = os.path.join(rank["output"], f".runner-cancel-{run_id}-rank{rank['rank']}")
    inner = f"if [ -e {q(cancel_file)} ]; then exit 143; fi; echo $$ > {q(pidfile)}; exec timeout --signal=TERM --kill-after=30s {timeout} {q(script)}"
    source = "source scripts/spark_runtime_env.sh && " if backend == "megatron" else ""
    provenance = (
        f"printf '[runner] remote_commit=%s\\n[runner] remote_dirty=\\n' {q(expected_commit)}; "
    )
    # NFS directory/negative lookup caches can hide a peer claim for up to 60s.
    claim_attempts = min(timeout, 70) * 10
    body = (
        f"set -eu; "
        f"if mkdir {q(rank['output'])} 2>/dev/null; then created_output=1; "
        f"elif [ -d {q(rank['output'])} ]; then created_output=0; "
        f"else echo 'cannot allocate remote output' >&2; exit 2; fi; "
        f"if [ \"$created_output\" = 1 ]; then "
        f"mkdir {q(owner_dir)} || {{ echo 'cannot claim new remote output' >&2; exit 2; }}; "
        f"mkdir {q(session_claim)} || {{ echo 'cannot record remote session' >&2; exit 2; }}; "
        f"else attempts=0; while [ ! -d {q(session_claim)} ] && [ \"$attempts\" -lt {claim_attempts} ]; do attempts=$((attempts + 1)); sleep 0.1; done; "
        f"if [ ! -d {q(session_claim)} ]; then echo 'refusing prior, incomplete, or concurrent remote output' >&2; exit 2; fi; fi; "
        f"if ! mkdir {q(rank_claim)} 2>/dev/null; then echo 'refusing duplicate rank claim' >&2; exit 2; fi; "
        f"if [ -e {q(cancel_file)} ]; then echo 'remote launch cancelled before start' >&2; exit 143; fi; "
        f"cd {q(backend_root)}; {provenance}"
        f"{export_clause}set +e; {source}setsid bash -c {q(inner)}; rc=$?; set -e; rm -f {q(pidfile)}; exit $rc"
    )
    cleanup = (
        f"set -u; mkdir -p {q(rank['output'])} || exit 4; : > {q(cancel_file)} || exit 4; "
        f"attempts=0; while [ ! -s {q(pidfile)} ] && [ \"$attempts\" -lt 50 ]; do attempts=$((attempts + 1)); sleep 0.1; done; "
        f"if [ ! -s {q(pidfile)} ]; then exit 3; fi; pid=$(cat {q(pidfile)}) || exit 4; "
        f"if ! kill -0 -- \"-$pid\" 2>/dev/null && ! kill -0 \"$pid\" 2>/dev/null; then exit 0; fi; "
        f"kill -TERM -- \"-$pid\" 2>/dev/null || kill -TERM \"$pid\" 2>/dev/null || exit 4; "
        f"attempts=0; while kill -0 -- \"-$pid\" 2>/dev/null || kill -0 \"$pid\" 2>/dev/null; do attempts=$((attempts + 1)); [ \"$attempts\" -lt 50 ] || break; sleep 0.1; done; "
        f"if kill -0 -- \"-$pid\" 2>/dev/null || kill -0 \"$pid\" 2>/dev/null; then "
        f"kill -KILL -- \"-$pid\" 2>/dev/null || kill -KILL \"$pid\" 2>/dev/null || exit 4; "
        f"attempts=0; while kill -0 -- \"-$pid\" 2>/dev/null || kill -0 \"$pid\" 2>/dev/null; do attempts=$((attempts + 1)); [ \"$attempts\" -lt 50 ] || exit 4; sleep 0.1; done; fi; "
        f"exit 0"
    )
    return body, cleanup


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _finish_log(log: Any) -> None:
    if not log.closed:
        log.close()


def _stop_processes(
    processes: list[tuple[Any, Any, dict[str, Any], dict[str, Any]]],
    *,
    ssh_binary: str,
    run: Callable[..., Any],
    status: str,
    reason: str,
) -> None:
    """Cancel only transports launched by this controller and their recorded groups."""
    for proc, log, item, rank in processes:
        cleanup_error: str | None = None
        try:
            result = run(
                [ssh_binary, "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2", rank["host"], rank["cleanup"]],
                timeout=35,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            cleanup_code = getattr(result, "returncode", None)
            if cleanup_code != 0:
                cleanup_error = "remote cleanup was not confirmed" if cleanup_code is None else f"remote cleanup exited {cleanup_code}"
        except BaseException as exc:
            cleanup_error = f"remote cleanup failed: {exc}"
        try:
            proc.terminate()
        except (AttributeError, OSError):
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except (AttributeError, OSError, subprocess.TimeoutExpired) as exc:
                cleanup_error = cleanup_error or f"local transport cleanup failed: {exc}"
        except AttributeError:
            # Test doubles may expose only terminate/poll; real Popen transports
            # always support wait and are handled by the branch above.
            pass
        except BaseException as exc:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except (AttributeError, OSError, subprocess.TimeoutExpired) as kill_exc:
                cleanup_error = cleanup_error or f"local transport cleanup failed: {kill_exc}"
            else:
                cleanup_error = cleanup_error or f"local transport wait interrupted: {exc}"
        try:
            transport_code = proc.poll()
        except BaseException:
            transport_code = None
        item.update({
            "status": status,
            "exit_code": None,
            "cancellation_reason": reason,
            "transport_exit_code": transport_code,
        })
        if cleanup_error:
            item["cleanup_error"] = cleanup_error
        _finish_log(log)


def _record_remote_provenance(item: dict[str, Any]) -> None:
    for line in Path(item["log"]).read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("[runner] remote_commit="):
            item["remote_commit"] = line.split("=", 1)[1]
        elif line.startswith("[runner] remote_dirty="):
            item["remote_dirty"] = line.split("=", 1)[1]


def execute(plan: dict[str, Any], timeout: int, output: Path, ssh_binary: str = "ssh", popen: Callable[..., Any] = subprocess.Popen, run: Callable[..., Any] = subprocess.run) -> int:
    """Launch ranks and leave a terminal manifest even when controller work fails."""
    if plan.get("controller_dirty"):
        raise ConfigError("refusing dirty controller checkout")
    started_at = time.monotonic()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ConfigError(f"refusing to reuse existing output directory: {output}") from exc
    plan.setdefault("session_id", uuid.uuid4().hex)
    manifest: dict[str, Any] = {**plan, "status": "running", "timeout_seconds": timeout, "ranks": []}
    processes: list[tuple[Any, Any, dict[str, Any], dict[str, Any]]] = []
    outcome = "passed"
    result_code = 0
    controller_error: str | None = None
    try:
        for rank in plan["ranks"]:
            command, cleanup = remote_command(rank, plan["backend"], timeout, plan["run_id"], plan["controller_commit"], plan["session_id"])
            manifest["ranks"].append({
                "rank": rank["rank"], "host": rank["host"], "output": rank["output"],
                "log": str(output / f"rank-{rank['rank']}.log"), "command": command,
                "status": "not-started", "exit_code": None, "remote_commit": None,
                "remote_dirty": None,
            })
            rank["command"], rank["cleanup"] = command, cleanup
        _write_manifest(output / "manifest.json", manifest)
        for item in manifest["ranks"]:
            log = open(item["log"], "w", encoding="utf-8")
            try:
                proc = popen(
                    [ssh_binary, "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2", item["host"], plan["ranks"][item["rank"]]["command"]],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            except OSError as exc:
                log.write(f"[runner] failed to start ssh: {exc}\n")
                _finish_log(log)
                item.update({"status": "failed", "exit_code": 127, "error": str(exc)})
                raise
            item["status"] = "running"
            processes.append((proc, log, item, plan["ranks"][item["rank"]]))
        deadline = time.monotonic() + timeout + 35
        while processes:
            if time.monotonic() >= deadline:
                outcome, result_code = "timeout", 1
                break
            remaining = []
            completed = []
            for proc, log, item, rank in processes:
                code = proc.poll()
                if code is None:
                    remaining.append((proc, log, item, rank))
                else:
                    completed.append((log, item, code))
            # Remove completed transports before reading their logs. A controller
            # error while recording provenance must only cancel live ranks.
            processes = remaining
            peer_failed = False
            for log, item, code in completed:
                _finish_log(log)
                item.update({"status": "passed" if code == 0 else "failed", "exit_code": code})
                _record_remote_provenance(item)
                if code != 0:
                    peer_failed = True
            if peer_failed:
                outcome, result_code = "failed", 1
                break
            _write_manifest(output / "manifest.json", manifest)
            if processes:
                time.sleep(0.05)
    except KeyboardInterrupt:
        outcome, result_code = "interrupted", 130
    except BaseException as exc:
        outcome, result_code = "failed", 1
        controller_error = f"controller error: {exc}"
    finally:
        if processes:
            if outcome == "interrupted":
                cancel_status, reason = "interrupted", "controller-interrupt"
            elif outcome == "timeout":
                cancel_status, reason = "timeout", "controller-timeout"
            else:
                cancel_status, reason = "cancelled", "controller-failure" if outcome == "failed" else "controller-cleanup"
            _stop_processes(processes, ssh_binary=ssh_binary, run=run, status=cancel_status, reason=reason)
        for item in manifest["ranks"]:
            if item["status"] == "running":
                item.update({"status": "not-started", "exit_code": None, "cancellation_reason": "controller-failure"})
        if controller_error:
            manifest["controller_error"] = controller_error
        if outcome == "passed" and any(item["status"] != "passed" for item in manifest["ranks"]):
            outcome, result_code = "failed", 1
        manifest["status"] = outcome
        manifest["elapsed_seconds"] = round(time.monotonic() - started_at, 3)
        try:
            _write_manifest(output / "manifest.json", manifest)
        except BaseException as exc:
            # Cleanup has already run. Do not claim a durable manifest when its write failed.
            result_code = 1
            print(f"runner error: cannot write manifest: {exc}", file=sys.stderr)
    return result_code

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--backend", choices=("trl", "megatron"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--execute", action="store_true", help="run remote jobs; default is dry-run")
    args = parser.parse_args(argv)
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    try:
        setup = load_setup(args.setup)
        experiment = load_experiment(args.experiment)
        if experiment["backend"] != args.backend:
            raise ConfigError("--backend must match experiment.backend")
        output = args.output.resolve()
        if output.exists():
            raise ConfigError(f"refusing to reuse existing output directory: {output}")
        plan = build_plan(setup, experiment, args.setup.resolve(), args.experiment.resolve(), output, Path(__file__).resolve().parents[1])
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    if not args.execute:
        print(json.dumps({**plan, "mode": "dry-run", "timeout_seconds": args.timeout}, indent=2, sort_keys=True))
        return 0
    try:
        return execute(plan, args.timeout, output)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
