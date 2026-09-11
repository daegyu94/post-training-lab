"""Run the repeated Megatron feature measurements through the repository runner.

The driver is deliberately serial: a cell owns one output directory and a
repeated run never reuses another run's remote claim.  It records raw evidence
and keeps failed or warmup runs out of the timing summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = Path(__file__).resolve().parent / "megatron" / "benchmark-plan.json"
sys.path.insert(0, str(ROOT))
from experiments import run as runner  # noqa: E402

sys.path.insert(0, str(ROOT / "backends" / "megatron"))
from megatron_lab.feature_lab import parse_megatron_log  # noqa: E402


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read benchmark plan {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("benchmark plan must be an object")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe(value: Any) -> str:
    text = str(value).lower() if isinstance(value, bool) else str(value)
    result = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)
    return result.strip("._-") or "value"


def _env_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def load_benchmark_plan(path: Path) -> dict[str, Any]:
    plan = _read(path)
    if plan.get("backend") != "megatron" or plan.get("setup") != "spark":
        raise ValueError("benchmark plan must target backend megatron and setup spark")
    defaults = plan.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("benchmark defaults must be an object")
    for key in ("steps", "repeats", "within_run_warmup"):
        if not isinstance(defaults.get(key), int) or defaults[key] < 1:
            raise ValueError(f"defaults.{key} must be a positive integer")
    intervals = defaults.get("checkpoint_intervals")
    if not isinstance(intervals, list) or not intervals or any(not isinstance(x, int) or x < 1 for x in intervals):
        raise ValueError("defaults.checkpoint_intervals must contain positive integers")
    cells = plan.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("benchmark plan must contain cells")
    for cell in cells:
        if not isinstance(cell, dict) or not isinstance(cell.get("name"), str):
            raise ValueError("each benchmark cell needs a name")
        if not isinstance(cell.get("variants"), list) or len(cell["variants"]) < 2:
            raise ValueError(f"cell {cell.get('name')} must contain at least two variants")
        for variant in cell["variants"]:
            if not isinstance(variant, dict) or not isinstance(variant.get("name"), str):
                raise ValueError(f"cell {cell['name']} has an invalid variant")
            if not isinstance(variant.get("env", {}), dict):
                raise ValueError(f"cell {cell['name']} variant env must be an object")
    return plan


def schedule_runs(plan: dict[str, Any], repeats: int, within_run_warmup: int) -> list[dict[str, Any]]:
    """Return warmups first, followed by AB/BA measured ordering per cell."""
    if repeats < 1 or within_run_warmup < 1:
        raise ValueError("repeats and within_run_warmup must be positive")
    result: list[dict[str, Any]] = []
    run_index = 0
    for cell in plan["cells"]:
        variants = cell["variants"]
        for variant in variants:
            result.append({"cell": cell["name"], "variant": variant["name"], "warmup": True,
                           "run_index": run_index, "warmup_steps": 0, "steps": within_run_warmup})
            run_index += 1
        for repeat in range(repeats):
            order = variants if repeat % 2 == 0 else list(reversed(variants))
            for variant in order:
                result.append({"cell": cell["name"], "variant": variant["name"], "warmup": False,
                               "repeat": repeat, "run_index": run_index, "warmup_steps": within_run_warmup})
                run_index += 1
    return result


def _invariant_env(env: dict[str, Any], exclude: set[str]) -> dict[str, str]:
    invariant = {key: _env_value(value) for key, value in env.items() if key not in exclude}
    # A runner-owned topology value is part of the comparison contract.
    return {key: invariant[key] for key in sorted(invariant)}


def validate_cell_invariants(cell: dict[str, Any], base_env: dict[str, Any]) -> None:
    variants = cell["variants"]
    exclude = set(cell.get("varying", []))
    first = dict(base_env)
    first.update(variants[0].get("env", {}))
    expected = _invariant_env(first, exclude)
    for variant in variants[1:]:
        actual_env = dict(base_env)
        actual_env.update(variant.get("env", {}))
        actual = _invariant_env(actual_env, exclude)
        if actual != expected:
            differing = sorted(key for key in set(actual) | set(expected) if actual.get(key) != expected.get(key))
            raise ValueError(f"cell {cell['name']} changes an invariant setting: {', '.join(differing)}")


def _metric_number(item: dict[str, Any], names: tuple[str, ...]) -> float | None:
    for name in names:
        value = item.get(name)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def summarize_measurements(records: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [r for r in records if r.get("status") == "passed" and not r.get("warmup") and r.get("parsed")]
    values = [float(r["parsed"]["steady_step_median_ms"]) for r in measured]
    if not values:
        return {"count": 0, "interpretation": "no successful measured repeats"}
    wall = [float(r.get("elapsed_seconds", 0.0)) for r in measured]
    peaks_alloc = [float(r["metrics"]["peak_cuda_allocated_bytes"]) for r in measured if r.get("metrics", {}).get("peak_cuda_allocated_bytes") is not None]
    peaks_reserved = [float(r["metrics"]["peak_cuda_reserved_bytes"]) for r in measured if r.get("metrics", {}).get("peak_cuda_reserved_bytes") is not None]
    save = [float(r["metrics"]["save_call_host_seconds_max_across_ranks"]) for r in measured if r.get("metrics", {}).get("save_call_host_seconds_max_across_ranks") is not None]
    final = [float(r["metrics"]["blocking_finalization_host_seconds_max_across_ranks"]) for r in measured if r.get("metrics", {}).get("blocking_finalization_host_seconds_max_across_ranks") is not None]
    checkpoint_timers = [timer for r in measured for timer in r.get("parsed", {}).get("checkpoint_save_rank_range_ms", [])]
    def stats(items: list[float]) -> dict[str, Any] | None:
        if not items:
            return None
        return {"median": statistics.median(items), "min": min(items), "max": max(items),
                "stdev": statistics.stdev(items) if len(items) > 1 else 0.0}
    return {"count": len(values), "steady_step_median_ms": stats(values), "wall_seconds": stats(wall),
            "peak_cuda_allocated_bytes_max": max(peaks_alloc) if peaks_alloc else None,
            "peak_cuda_reserved_bytes_max": max(peaks_reserved) if peaks_reserved else None,
            "save_call_host_seconds_max_across_ranks": stats(save),
            "blocking_finalization_host_seconds_max_across_ranks": stats(final),
            "native_checkpoint_timer_ms": checkpoint_timers or None,
            "interpretation": "descriptive timing and finite loss/grad evidence; no quality or speedup claim"}


def parse_measurement_lines(lines: list[str]) -> dict[str, Any]:
    alloc: list[float] = []
    reserved: list[float] = []
    saves: list[float] = []
    finalizations: list[float] = []
    recognized_events = 0
    for line in lines:
        if not line.strip():
            continue
        item = json.loads(line)
        event = str(item.get("event", item.get("name", item.get("type", ""))))
        seconds = _metric_number(item, ("host_seconds", "seconds", "elapsed_seconds"))
        if event == "save" and seconds is not None and item.get("success", True):
            recognized_events += 1
            saves.append(seconds)
        elif (event == "finalize_async_saves" and seconds is not None
              and item.get("success", True) and bool(item.get("blocking", False))):
            recognized_events += 1
            finalizations.append(seconds)
        elif event == "stage":
            recognized_events += 1
            value = _metric_number(item, ("peak_cuda_allocated_bytes",))
            if value is not None: alloc.append(value)
            value = _metric_number(item, ("peak_cuda_reserved_bytes",))
            if value is not None: reserved.append(value)
    return {"peak_cuda_allocated_bytes": max(alloc) if alloc else None,
            "peak_cuda_reserved_bytes": max(reserved) if reserved else None,
            "save_call_host_seconds_sum_per_rank": sum(saves) if saves else None,
            "blocking_finalization_host_seconds_sum_per_rank": sum(finalizations) if finalizations else None,
            "recognized_events": recognized_events}


def validate_fetched_metrics(metrics_by_rank: dict[str, Any]) -> None:
    if (not metrics_by_rank
            or any(not isinstance(value.get("metrics"), dict) or not value["metrics"]
                   or value["metrics"].get("recognized_events", 1) == 0
                   for value in metrics_by_rank.values())):
        raise ValueError("measurement JSONL is missing or empty for one or more ranks")


def fetch_measurements(plan: dict[str, Any], run_output: Path, *, run: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    by_rank: dict[str, Any] = {}
    for rank in plan["ranks"]:
        remote = str(Path(rank["output"]) / "measurements" / f"rank-{rank['rank']}-train.jsonl")
        command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", rank["host"], f"cat -- {shlex.quote(remote)}"]
        result = run(command, check=False, capture_output=True, text=True, timeout=35)
        if getattr(result, "returncode", 1) != 0:
            raise RuntimeError(f"cannot fetch measurements for rank {rank['rank']}")
        raw = result.stdout or ""
        destination = run_output / "measurements" / f"rank-{rank['rank']}-train.jsonl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(raw, encoding="utf-8")
        by_rank[str(rank["rank"])] = {"path": str(destination), "sha256": _sha(destination), "metrics": parse_measurement_lines(raw.splitlines())}
    return by_rank


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _record(path: Path, item: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, sort_keys=True) + "\n")
        handle.flush()


def run_benchmark(*, setup_path: Path, benchmark_path: Path, output: Path, execute: bool = False,
                  steps: int | None = None, repeats: int | None = None,
                  within_run_warmup: int | None = None, checkpoint_intervals: list[int] | None = None,
                  timeout: int = 900,
                  execute_fn: Callable[..., int] = runner.execute,
                  fetch_fn: Callable[..., dict[str, Any]] = fetch_measurements,
                  post_run_fn: Callable[[dict[str, Any], Path, dict[str, Any]], dict[str, Any]] | None = None,
                  cleanup_fn: Callable[[dict[str, Any], Path, dict[str, Any]], None] | None = None,
                  ) -> dict[str, Any]:
    if output.exists():
        raise ValueError(f"refusing to reuse existing benchmark output: {output}")
    benchmark = load_benchmark_plan(benchmark_path)
    setup = runner.load_setup(setup_path)
    base_path = ROOT / benchmark.get("base_experiment", "experiments/megatron/smoke.json")
    base = runner.load_experiment(base_path)
    if base["backend"] != "megatron" or base["setup"] != "spark":
        raise ValueError("benchmark base experiment must target Megatron Spark")
    defaults = benchmark["defaults"]
    steps = steps if steps is not None else defaults["steps"]
    repeats = repeats if repeats is not None else defaults["repeats"]
    within_run_warmup = within_run_warmup if within_run_warmup is not None else defaults["within_run_warmup"]
    intervals = checkpoint_intervals if checkpoint_intervals is not None else defaults["checkpoint_intervals"]
    if steps < 1 or repeats < 1 or within_run_warmup < 1 or timeout < 1:
        raise ValueError("steps, repeats, within-run warmup, and timeout must be positive")
    if len(intervals) != 2 or any(value < 1 for value in intervals):
        raise ValueError("checkpoint intervals must contain two positive integers")
    if steps <= within_run_warmup:
        raise ValueError("steps must exceed within-run warmup")
    base_env = dict(base["env"])
    base_env.update(benchmark.get("common_env", {}))
    for key, value in {
        "STAGE": "train", "MEASURE_TIMING": "true", "SEED": 42,
        "GLOBAL_BATCH_SIZE": 4, "EP": 1, "PP": 1,
    }.items():
        base_env.setdefault(key, value)
    base_env.update({"MAX_STEPS": steps, "SCHEDULE_STEPS": steps})
    for cell in benchmark["cells"]:
        for variant in cell["variants"]:
            if "SAVE_INTERVAL" in variant.get("env", {}) and variant["env"]["SAVE_INTERVAL"] in defaults["checkpoint_intervals"]:
                # CLI cost overrides replace only the checkpoint axis values.
                index = defaults["checkpoint_intervals"].index(variant["env"]["SAVE_INTERVAL"])
                variant["env"]["SAVE_INTERVAL"] = intervals[index] if index < len(intervals) else variant["env"]["SAVE_INTERVAL"]
        validate_cell_invariants(cell, base_env)
    schedule = schedule_runs(benchmark, repeats, within_run_warmup)
    if not execute:
        return {"status": "dry-run", "steps": steps, "repeats": repeats, "within_run_warmup": within_run_warmup,
                "effective_common_env": base_env, "planned_runs": schedule,
                "source_commit": _git_head(), "plan_sha256": _sha(benchmark_path)}
    output.mkdir(parents=True)
    (output / "plan.json").write_text(json.dumps(benchmark, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    records_path = output / "records.jsonl"
    source_commit = _git_head()
    metadata: dict[str, Any] = {"status": "running", "source_commit": source_commit, "plan_sha256": _sha(benchmark_path),
                                "steps": steps, "repeats": repeats, "within_run_warmup": within_run_warmup,
                                "timeout_seconds": timeout, "records": []}
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    failed_variants: set[tuple[str, str]] = set()
    summaries: dict[str, dict[str, Any]] = {}
    for item in schedule:
        key = (item["cell"], item["variant"])
        if key in failed_variants:
            record = {**item, "status": "skipped", "reason": "variant failed earlier; remaining repeats skipped"}
            _record(records_path, record); metadata["records"].append(record)
            (output / "manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            continue
        cell = next(c for c in benchmark["cells"] if c["name"] == item["cell"])
        variant = next(v for v in cell["variants"] if v["name"] == item["variant"])
        run_name = f"{_safe(output.name)}--{_safe(item['cell'])}--{_safe(item['variant'])}--run-{item['run_index']:03d}"
        run_output = output / "runs" / run_name
        effective = dict(base)
        effective["env"] = dict(base_env)
        effective_steps = item.get("steps", steps)
        effective["env"].update({"MAX_STEPS": effective_steps, "SCHEDULE_STEPS": effective_steps})
        if not item["cell"].startswith("checkpoint-"):
            effective["env"]["SAVE_INTERVAL"] = effective_steps
        effective["env"].update({key: _env_value(value) for key, value in variant.get("env", {}).items()})
        config_path = output / "configs" / f"{run_name}.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(effective, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        started = time.monotonic()
        record: dict[str, Any] = {**item, "status": "running", "output": str(run_output), "config": str(config_path),
                                  "config_sha256": _sha(config_path), "controller_commit": source_commit}
        print(f"[benchmark] start {run_name}", flush=True)
        try:
            plan = runner.build_plan(setup, runner.load_experiment(config_path), setup_path.resolve(), config_path.resolve(), run_output, ROOT)
            code = execute_fn(plan, timeout, run_output)
            record["elapsed_seconds"] = round(time.monotonic() - started, 3)
            if code == 130:
                raise KeyboardInterrupt
            record["status"] = "passed" if code == 0 else "failed"
            record["exit_code"] = code
            if code == 0:
                log = run_output / f"rank-{len(plan['ranks']) - 1}.log"
                cell_feature_prefixes = (
                    ("overlap", "overlap-grad-reduce"), ("recompute", "recompute"),
                    ("sequence-parallel", "sequence-parallel"), ("checkpoint", "checkpoint"),
                    ("lora-ratio", "lora-ratio"),
                )
                try:
                    feature = next(name for prefix, name in cell_feature_prefixes if item["cell"].startswith(prefix))
                except StopIteration:
                    raise ValueError(f"cell {item['cell']!r} does not match a known feature prefix") from None
                record["parsed"] = parse_megatron_log(log, feature=feature, variant=item["variant"], run_index=item["run_index"], warmup_steps=item.get("warmup_steps", 0), exit_code=0, completed=True)
                if len(record["parsed"]["steps"]) != effective_steps:
                    raise ValueError(f"native log contains {len(record['parsed']['steps'])} steps; expected {effective_steps}")
                record["metrics_by_rank"] = fetch_fn(plan, run_output)
                validate_fetched_metrics(record["metrics_by_rank"])
                metric_values = list(record["metrics_by_rank"].values())
                record["metrics"] = {key: max((v["metrics"].get(key) for v in metric_values if v["metrics"].get(key) is not None), default=None) for key in ("peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes", "save_call_host_seconds_sum_per_rank", "blocking_finalization_host_seconds_sum_per_rank")}
                record["metrics"]["save_call_host_seconds_max_across_ranks"] = record["metrics"].pop("save_call_host_seconds_sum_per_rank")
                record["metrics"]["blocking_finalization_host_seconds_max_across_ranks"] = record["metrics"].pop("blocking_finalization_host_seconds_sum_per_rank")
                if post_run_fn is not None and not item["warmup"]:
                    try:
                        record["post_run"] = post_run_fn(plan, run_output, item)
                    except Exception as exc:
                        # Telemetry collection failing must not retract a training run that already passed.
                        record["post_run_error"] = str(exc)
                if cleanup_fn is not None:
                    try:
                        cleanup_fn(plan, run_output, item)
                    except Exception as exc:
                        record["cleanup_error"] = str(exc)
            else:
                failed_variants.add(key)
        except KeyboardInterrupt:
            record.update({"status": "interrupted", "error": "controller interrupt",
                           "elapsed_seconds": round(time.monotonic() - started, 3)})
            _record(records_path, record)
            metadata["records"].append(record)
            metadata["status"] = "interrupted"
            metadata["elapsed_seconds"] = round(sum(float(r.get("elapsed_seconds", 0)) for r in metadata["records"]), 3)
            (output / "manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return metadata
        except Exception as exc:
            record.update({"status": "failed", "error": str(exc), "elapsed_seconds": round(time.monotonic() - started, 3)})
            failed_variants.add(key)
        _record(records_path, record); metadata["records"].append(record)
        (output / "manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[benchmark] {record['status']} {run_name}", flush=True)
        if record["status"] == "passed" and not item["warmup"]:
            summaries.setdefault(item["cell"] + ":" + item["variant"], {}).setdefault("records", []).append(record)
    metadata["status"] = "passed" if not failed_variants else "completed-with-failures"
    metadata["summaries"] = {key: summarize_measurements(value["records"]) for key, value in summaries.items()}
    metadata["elapsed_seconds"] = round(sum(float(r.get("elapsed_seconds", 0)) for r in metadata["records"]), 3)
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--within-run-warmup", type=int)
    parser.add_argument("--checkpoint-intervals", type=int, nargs=2)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args(argv)
    try:
        result = run_benchmark(setup_path=args.setup, benchmark_path=args.plan, output=args.output, execute=args.execute,
                               steps=args.steps, repeats=args.repeats, within_run_warmup=args.within_run_warmup,
                               checkpoint_intervals=args.checkpoint_intervals, timeout=args.timeout)
    except (OSError, ValueError, runner.ConfigError, subprocess.SubprocessError) as exc:
        print(f"configuration or benchmark error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") in {"dry-run", "passed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
