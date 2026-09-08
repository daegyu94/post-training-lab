from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from experiments import run


MODEL_REVISION = "a" * 40
DATASET_REVISION = "b" * 40


def config_files(tmp_path: Path, *, backend: str = "trl", nnodes: int = 2) -> tuple[Path, Path]:
    setup = {
        "setup": "spark",
        "master_addr": "spark1",
        "master_port": 29670,
        "env": {"NCCL_DEBUG": "WARN", "OMP_NUM_THREADS": "2"},
        "nodes": [
            {
                "host": f"spark{index + 1}",
                "checkout": "/srv/post-training-unified",
                "python": {"trl": "/env/trl/bin/python", "megatron": "/env/megatron/bin/python"},
                "env": {},
                "model_dirs": {"Qwen/Qwen2.5-0.5B-Instruct": "/models/qwen"},
                "data_dir": "/data/smoke",
                "output_root": f"/results/controller/spark{index + 1}",
            }
            for index in range(2)
        ],
    }
    env: dict[str, Any] = {
        "MODEL_ID": "Qwen/Qwen2.5-0.5B-Instruct",
        "MODEL_REVISION": MODEL_REVISION,
        "DATASET_REVISION": DATASET_REVISION,
        "DATASET_ID": "HuggingFaceH4/ultrachat_200k",
    }
    if backend == "trl":
        env.update({"DISTRIBUTED_BACKEND": "ddp", "STAGE": "all"})
    else:
        env.update({"TP": 1, "PP": 1, "EP": 1, "MICRO_BATCH_SIZE": 1, "GLOBAL_BATCH_SIZE": 2})
    experiment = {"backend": backend, "setup": "spark", "nnodes": nnodes, "nproc_per_node": 1, "env": env}
    setup_path = tmp_path / "setup.json"
    experiment_path = tmp_path / "experiment.json"
    setup_path.write_text(json.dumps(setup), encoding="utf-8")
    experiment_path.write_text(json.dumps(experiment), encoding="utf-8")
    return setup_path, experiment_path


def test_dry_run_prints_hashes_without_creating_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    setup_path, experiment_path = config_files(tmp_path, nnodes=1)
    output = tmp_path / "dry-run"
    assert run.main(["--backend", "trl", "--setup", str(setup_path), "--experiment", str(experiment_path), "--output", str(output)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["mode"] == "dry-run"
    assert printed["config_sha256"]["setup"]
    assert printed["ranks"][0]["env"]["NODE_RANK"] == "0"
    assert not output.exists()


def test_rejects_reserved_and_invalid_topology(tmp_path: Path) -> None:
    setup_path, experiment_path = config_files(tmp_path, backend="megatron")
    experiment = json.loads(experiment_path.read_text())
    experiment["env"]["NODE_RANK"] = "0"
    experiment_path.write_text(json.dumps(experiment))
    with pytest.raises(run.ConfigError, match="runner-owned"):
        run.load_experiment(experiment_path)

    experiment["env"].pop("NODE_RANK")
    experiment["env"].update({"TP": 2, "EP": 3})
    experiment_path.write_text(json.dumps(experiment))
    with pytest.raises(run.ConfigError, match="divisible"):
        run.load_experiment(experiment_path)


def test_rejects_sharded_trl_all_stage(tmp_path: Path) -> None:
    _, experiment_path = config_files(tmp_path)
    experiment = json.loads(experiment_path.read_text())
    experiment["env"].update({"DISTRIBUTED_BACKEND": "fsdp2", "STAGE": "all"})
    experiment_path.write_text(json.dumps(experiment))
    with pytest.raises(run.ConfigError, match="only support"):
        run.load_experiment(experiment_path)


def test_remote_command_quotes_values_and_has_bounded_timeout(tmp_path: Path) -> None:
    setup_path, experiment_path = config_files(tmp_path, nnodes=1)
    setup = run.load_setup(setup_path)
    experiment = run.load_experiment(experiment_path)
    experiment["env"]["NCCL_DEBUG"] = "value with 'quotes'"
    plan = run.build_plan(setup, experiment, setup_path, experiment_path, tmp_path / "quoted", tmp_path)
    command, cleanup = run.remote_command(plan["ranks"][0], "megatron", 900, "quoted")
    assert "timeout --signal=TERM --kill-after=30s 900" in command
    assert "source scripts/spark_runtime_env.sh" in command
    assert "value with" in command and "quotes" in command
    assert ".runner-quoted-rank0.pid" in cleanup
    assert "kill -TERM -- \"-$pid\"" in cleanup


def test_shared_output_uses_same_session_and_rank_specific_claims(tmp_path: Path) -> None:
    setup_path, experiment_path = config_files(tmp_path)
    setup = run.load_setup(setup_path)
    experiment = run.load_experiment(experiment_path)
    setup["nodes"][1]["output_root"] = setup["nodes"][0]["output_root"]
    plan = run.build_plan(setup, experiment, setup_path, experiment_path, tmp_path / "shared", tmp_path)
    assert plan["ranks"][0]["output"] == plan["ranks"][1]["output"]
    first, _ = run.remote_command(plan["ranks"][0], "trl", 900, "shared", "unknown", "session-a")
    second, _ = run.remote_command(plan["ranks"][1], "trl", 900, "shared", "unknown", "session-a")
    assert ".runner-session-shared" in first
    assert ".runner-session-shared" in second
    assert ".runner-rank-shared-0" in first
    assert ".runner-rank-shared-1" in second
    assert "refusing prior, incomplete, or concurrent remote output" in first
    assert ".runner-cancel-shared-rank0" in first


def test_remote_provenance_is_recorded_and_mismatch_is_rejected(tmp_path: Path) -> None:
    setup_path, experiment_path = config_files(tmp_path, nnodes=1)
    setup = run.load_setup(setup_path)
    experiment = run.load_experiment(experiment_path)
    plan = run.build_plan(setup, experiment, setup_path, experiment_path, tmp_path / "provenance", tmp_path)
    command, _ = run.remote_command(plan["ranks"][0], "trl", 900, "provenance", "deadbeef")
    assert "remote_commit" in command and "remote_dirty" in command
    assert "remote commit mismatch" in command
    assert "remote checkout is dirty" in command


class FakeProcess:
    def __init__(self, code: int | None) -> None:
        self.code = code
        self.terminated = False
        self.polls = 0

    def poll(self) -> int | None:
        self.polls += 1
        return self.code

    def terminate(self) -> None:
        self.terminated = True
        self.code = -15


class InterruptProcess(FakeProcess):
    def poll(self) -> int | None:
        raise KeyboardInterrupt


def test_execute_records_rank_completion_failure_and_targeted_cleanup(tmp_path: Path) -> None:
    setup_path, experiment_path = config_files(tmp_path)
    setup = run.load_setup(setup_path)
    experiment = run.load_experiment(experiment_path)
    output = tmp_path / "execution"
    plan = run.build_plan(setup, experiment, setup_path, experiment_path, output, tmp_path)
    processes = [FakeProcess(1), FakeProcess(None)]
    cleanup_calls: list[list[str]] = []

    def fake_popen(*_: Any, **__: Any) -> FakeProcess:
        return processes.pop(0)

    def fake_run(args: list[str], **_: Any) -> None:
        cleanup_calls.append(args)

    assert run.execute(plan, 7, output, popen=fake_popen, run=fake_run) == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["ranks"][0]["exit_code"] == 1
    assert manifest["ranks"][1]["exit_code"] is None
    assert manifest["ranks"][1]["cancellation_reason"] == "controller-failure"
    assert manifest["ranks"][1]["cleanup_error"] == "remote cleanup was not confirmed"
    assert cleanup_calls and ".runner-execution-rank1.pid" in cleanup_calls[0][-1]
    assert manifest["elapsed_seconds"] >= 0


def test_execute_interrupt_from_sleep_updates_every_rank_and_cleans_remote(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    setup_path, experiment_path = config_files(tmp_path, nnodes=1)
    setup = run.load_setup(setup_path)
    experiment = run.load_experiment(experiment_path)
    output = tmp_path / "interrupt"
    plan = run.build_plan(setup, experiment, setup_path, experiment_path, output, tmp_path)
    cleanup_calls: list[list[str]] = []

    def fake_popen(*_: Any, **__: Any) -> FakeProcess:
        return FakeProcess(None)

    def fake_run(args: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        cleanup_calls.append(args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(run.time, "sleep", lambda _: (_ for _ in ()).throw(KeyboardInterrupt()))
    assert run.execute(plan, 900, output, popen=fake_popen, run=fake_run) == 130
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "interrupted"
    assert manifest["ranks"][0]["status"] == "interrupted"
    assert manifest["ranks"][0]["exit_code"] is None
    assert manifest["ranks"][0]["cancellation_reason"] == "controller-interrupt"
    assert cleanup_calls


def test_execute_timeout_records_timeout_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    setup_path, experiment_path = config_files(tmp_path, nnodes=1)
    setup = run.load_setup(setup_path)
    experiment = run.load_experiment(experiment_path)
    output = tmp_path / "timeout"
    plan = run.build_plan(setup, experiment, setup_path, experiment_path, output, tmp_path)
    clock = iter((0.0, 0.0, 100.0, *([101.0] * 100)))
    monkeypatch.setattr(run.time, "monotonic", lambda: next(clock))

    def fake_popen(*_: Any, **__: Any) -> FakeProcess:
        return FakeProcess(None)

    assert run.execute(plan, 1, output, popen=fake_popen, run=lambda *_args, **_kwargs: None) == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "timeout"
    assert manifest["ranks"][0]["status"] == "timeout"


def _local_rank(tmp_path: Path, rank: int, output: Path) -> dict[str, Any]:
    checkout = tmp_path / "checkout"
    script = checkout / "backends" / "trl" / "scripts" / "run_spark_cluster.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    script.chmod(0o755)
    return {"rank": rank, "checkout": str(checkout), "output": str(output), "env": {}}


def test_remote_claims_allow_concurrent_shared_session_and_refuse_stale_output(tmp_path: Path) -> None:
    remote_output = tmp_path / "remote-output"
    first_rank = _local_rank(tmp_path, 0, remote_output)
    second_rank = _local_rank(tmp_path, 1, remote_output)
    first, _ = run.remote_command(first_rank, "trl", 5, "local", "unknown", "session-a")
    second, _ = run.remote_command(second_rank, "trl", 5, "local", "unknown", "session-a")
    first_process = subprocess.Popen(["bash", "-c", first], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    second_process = subprocess.Popen(["bash", "-c", second], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    first_stdout, first_stderr = first_process.communicate(timeout=10)
    second_stdout, second_stderr = second_process.communicate(timeout=10)
    assert first_process.returncode == 0, first_stdout + first_stderr
    assert second_process.returncode == 0, second_stdout + second_stderr
    assert (remote_output / ".runner-session-local" / "session-a").is_dir()
    assert (remote_output / ".runner-rank-local-0").is_dir()
    assert (remote_output / ".runner-rank-local-1").is_dir()

    stale_output = tmp_path / "stale-output"
    stale_output.mkdir()
    stale_rank = _local_rank(tmp_path, 0, stale_output)
    stale, _ = run.remote_command(stale_rank, "trl", 5, "stale", "unknown", "session-a")
    refused = subprocess.run(["bash", "-c", stale], capture_output=True, text=True, timeout=10)
    assert refused.returncode == 2
    assert "incomplete" in refused.stderr


def test_execute_interrupt_from_manifest_write_cleans_started_rank(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    setup_path, experiment_path = config_files(tmp_path, nnodes=1)
    plan = run.build_plan(run.load_setup(setup_path), run.load_experiment(experiment_path), setup_path, experiment_path, tmp_path / "manifest-interrupt", tmp_path)
    original_write = run._write_manifest
    writes = 0
    cleanup_calls: list[list[str]] = []

    def interrupted_write(path: Path, manifest: dict[str, Any]) -> None:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise KeyboardInterrupt
        original_write(path, manifest)

    monkeypatch.setattr(run, "_write_manifest", interrupted_write)
    assert run.execute(plan, 900, tmp_path / "manifest-interrupt", popen=lambda *_a, **_k: FakeProcess(None), run=lambda args, **_k: (cleanup_calls.append(args), subprocess.CompletedProcess(args, 0))[1]) == 130
    manifest = json.loads((tmp_path / "manifest-interrupt" / "manifest.json").read_text())
    assert manifest["status"] == "interrupted"
    assert cleanup_calls


def test_execute_interrupt_from_provenance_preserves_completed_exit_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    setup_path, experiment_path = config_files(tmp_path)
    plan = run.build_plan(run.load_setup(setup_path), run.load_experiment(experiment_path), setup_path, experiment_path, tmp_path / "provenance-interrupt", tmp_path)
    processes = [FakeProcess(0), FakeProcess(None)]
    cleanup_calls: list[list[str]] = []
    monkeypatch.setattr(run, "_record_remote_provenance", lambda _: (_ for _ in ()).throw(KeyboardInterrupt()))

    assert run.execute(plan, 900, tmp_path / "provenance-interrupt", popen=lambda *_a, **_k: processes.pop(0), run=lambda args, **_k: (cleanup_calls.append(args), subprocess.CompletedProcess(args, 0))[1]) == 130
    manifest = json.loads((tmp_path / "provenance-interrupt" / "manifest.json").read_text())
    assert manifest["ranks"][0]["exit_code"] == 0
    assert manifest["ranks"][1]["status"] == "interrupted"
    assert cleanup_calls
