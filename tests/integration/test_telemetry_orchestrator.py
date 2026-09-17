from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from scripts import run_telemetry_cluster as telemetry


def test_setup_drives_targets_and_remote_cleanup(tmp_path: Path) -> None:
    setup = tmp_path / "setup.json"
    setup.write_text(
        json.dumps(
            {
                "setup": "spark",
                "nodes": [
                    {"host": "spark@spark1", "checkout": "/shared/lab"},
                    {"host": "spark@spark2", "checkout": "/shared/lab"},
                ],
            }
        ),
        encoding="utf-8",
    )
    plan = telemetry.load_plan(setup)

    assert plan["cluster_name"] == "spark-cluster"
    assert plan["targets"] == "spark1=spark1,spark2=spark2"
    launch, cleanup = telemetry.remote_commands(plan["nodes"][0], 60, "test-session")
    assert "cd /shared/lab/third_party/post-training-telemetry" in launch
    assert "NODE_ADDR=spark1" in launch
    assert 'tools_dir="$HOME/telemetry-tools"' in launch
    assert ".telemetry-orchestrator-test-session.pid" in launch
    assert "kill -TERM" in cleanup


class InterruptProcess:
    def __init__(self, interrupt: bool) -> None:
        self.terminated = False
        self.polls = 0
        self.interrupt = interrupt

    def poll(self) -> int | None:
        self.polls += 1
        if self.interrupt and self.polls == 1:
            raise KeyboardInterrupt
        return None

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: int) -> int:
        return 0


def test_interrupt_cleans_every_remote_node() -> None:
    processes: list[InterruptProcess] = []
    cleanup_calls: list[list[str]] = []
    plan = {
        "cluster_name": "spark-cluster",
        "targets": "spark1=spark1,spark2=spark2",
        "nodes": [
            {"host": "spark@spark1", "address": "spark1", "checkout": "/shared/lab"},
            {"host": "spark@spark2", "address": "spark2", "checkout": "/shared/lab"},
        ],
    }

    def popen(*_args: Any, **_kwargs: Any) -> InterruptProcess:
        process = InterruptProcess(interrupt=not processes)
        processes.append(process)
        return process

    def run(args: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        cleanup_calls.append(args)
        return subprocess.CompletedProcess(args, 0)

    assert telemetry.run_stack(
        plan,
        60,
        {"TOOLS_DIR": "/tools", "OUTPUT_DIR": "/data"},
        popen=popen,
        run=run,
    ) == 130
    assert [call[-2] for call in cleanup_calls] == ["spark@spark1", "spark@spark2"]
    assert all(process.terminated for process in processes)
