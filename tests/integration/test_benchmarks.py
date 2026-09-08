from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments import benchmarks


def _plan() -> dict:
    return {
        "backend": "megatron", "setup": "spark",
        "defaults": {"steps": 8, "repeats": 2, "within_run_warmup": 2, "checkpoint_intervals": [4, 8]},
        "cells": [{"name": "cell", "varying": ["MAX_LENGTH"], "variants": [
            {"name": "a", "env": {"MAX_LENGTH": 2048}},
            {"name": "b", "env": {"MAX_LENGTH": 4096}},
        ]}],
    }


def test_schedule_warmups_then_alternating_measured_order() -> None:
    runs = benchmarks.schedule_runs(_plan(), 4, 2)
    assert [(r["variant"], r["warmup"]) for r in runs[:2]] == [("a", True), ("b", True)]
    assert [r["variant"] for r in runs[2:]] == ["a", "b", "b", "a", "a", "b", "b", "a"]
    assert all(r["warmup_steps"] == 0 for r in runs[:2])
    assert all(r["warmup_steps"] == 2 for r in runs[2:])


def test_invariants_reject_unrequested_difference() -> None:
    cell = _plan()["cells"][0]
    cell["variants"][1]["env"]["SEED"] = 7
    with pytest.raises(ValueError, match="invariant"):
        benchmarks.validate_cell_invariants(cell, {"SEED": 42})


def test_measurement_events_keep_enqueue_and_finalize_separate() -> None:
    result = benchmarks.parse_measurement_lines([
        json.dumps({"event": "save", "host_seconds": 1.25, "success": True}),
        json.dumps({"event": "finalize_async_saves", "host_seconds": 4.5, "blocking": True, "success": True}),
        json.dumps({"event": "stage", "peak_cuda_allocated_bytes": 10, "peak_cuda_reserved_bytes": 20}),
        json.dumps({"event": "stage", "peak_cuda_allocated_bytes": 30, "peak_cuda_reserved_bytes": 25}),
    ])
    assert result == {"peak_cuda_allocated_bytes": 30.0, "peak_cuda_reserved_bytes": 25.0,
                      "save_call_host_seconds_sum_per_rank": 1.25, "blocking_finalization_host_seconds_sum_per_rank": 4.5, "recognized_events": 4}


def test_summary_excludes_warmup_and_failed_records() -> None:
    parsed = {"steady_step_median_ms": 5.0}
    records = [
        {"status": "passed", "warmup": True, "parsed": parsed, "elapsed_seconds": 3},
        {"status": "failed", "warmup": False, "parsed": parsed, "elapsed_seconds": 4},
        {"status": "passed", "warmup": False, "parsed": parsed, "elapsed_seconds": 7,
         "metrics": {"peak_cuda_allocated_bytes": 8}},
    ]
    summary = benchmarks.summarize_measurements(records)
    assert summary["count"] == 1
    assert summary["steady_step_median_ms"]["median"] == 5.0
    assert summary["wall_seconds"]["median"] == 7


def test_fetch_quotes_remote_path_and_writes_checksum(tmp_path: Path) -> None:
    class Result:
        returncode = 0
        stdout = json.dumps({"event": "stage", "peak_cuda_allocated_bytes": 4}) + "\n"

    seen = []

    def fake_run(command, **kwargs):
        seen.append(command)
        return Result()

    plan = {"ranks": [{"rank": 0, "host": "spark1", "output": "/shared/out/a b"}]}
    result = benchmarks.fetch_measurements(plan, tmp_path, run=fake_run)
    assert "'/shared/out/a b/measurements/rank-0-train.jsonl'" in seen[0][-1]
    assert result["0"]["sha256"]



def test_real_parser_verifies_complete_finite_native_steps(tmp_path: Path) -> None:
    log = tmp_path / "rank-last.log"
    log.write_text("\n".join(
        f"iteration {step} / 3 elapsed time per iteration (ms): {step + 1} | lm loss: 1.5 | grad norm: 2.0 | number of skipped iterations: 0 | number of nan iterations: 0"
        for step in range(1, 4)
    ) + "\n", encoding="utf-8")
    parsed = benchmarks.parse_megatron_log(log, feature="checkpoint", variant="sync", run_index=0,
                                           warmup_steps=1, exit_code=0, completed=True)
    assert parsed["completion_verified"] is True
    assert parsed["steady_step_median_ms"] == 3.5
    assert parsed["steady_step_ids"] == [2, 3]


def test_run_benchmark_uses_fake_runner_and_fetch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = _plan()
    plan["cells"][0]["name"] = "recompute-test"
    plan["cells"][0]["variants"][0]["name"] = "full"
    plan["cells"][0]["variants"][1]["name"] = "selective"
    plan["base_experiment"] = "base.json"
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    base_path = tmp_path / "base.json"
    base_path.write_text("{}", encoding="utf-8")
    base = {"backend": "megatron", "setup": "spark", "nnodes": 1, "nproc_per_node": 1,
            "env": {"MODEL_ID": "m", "MODEL_REVISION": "a" * 40,
                    "DATASET_ID": "d", "DATASET_REVISION": "b" * 40}}
    monkeypatch.setattr(benchmarks, "ROOT", tmp_path)
    monkeypatch.setattr(benchmarks.runner, "load_setup", lambda path: {"nodes": []})
    def fake_load(path):
        return base.copy() if path == base_path else json.loads(path.read_text(encoding="utf-8"))
    monkeypatch.setattr(benchmarks.runner, "load_experiment", fake_load)
    monkeypatch.setattr(benchmarks, "_git_head", lambda: "c" * 40)
    captured = []

    def fake_build(setup, experiment, setup_path, experiment_path, output, root):
        captured.append((experiment["env"]["MAX_STEPS"], output.name))
        return {"ranks": [{"rank": 0, "host": "local", "output": str(output / "remote")}],
                "backend": "megatron", "run_id": output.name, "controller_commit": "c" * 40}

    def fake_execute(plan_value, timeout, output):
        output.mkdir(parents=True, exist_ok=True)
        config = json.loads((output.parents[1] / "configs" / f"{output.name}.json").read_text(encoding="utf-8"))
        count = int(config["env"]["MAX_STEPS"])
        lines = [f"iteration {step} / {count} elapsed time per iteration (ms): {step + 2} | lm loss: 1 | grad norm: 2 | number of skipped iterations: 0 | number of nan iterations: 0" for step in range(1, count + 1)]
        (output / "rank-0.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(benchmarks.runner, "build_plan", fake_build)
    result = benchmarks.run_benchmark(setup_path=tmp_path / "setup.json", benchmark_path=plan_path,
                                      output=tmp_path / "result", execute=True, steps=4, repeats=1,
                                      within_run_warmup=1, checkpoint_intervals=[2, 3],
                                      execute_fn=fake_execute,
                                      fetch_fn=lambda plan_value, run_output: {"0": {"metrics": {"stage": True}}})
    assert result["status"] == "passed"
    assert len(result["records"]) == 4
    assert all(value["count"] == 1 for value in result["summaries"].values())
    assert [value[0] for value in captured[:2]] == [1, 1]
    assert all(value[1].startswith("result--") for value in captured)


def test_missing_measurement_metrics_are_rejected() -> None:
    with pytest.raises(ValueError, match="missing or empty"):
        benchmarks.validate_fetched_metrics({"0": {"metrics": {}}})
