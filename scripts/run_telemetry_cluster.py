"""Run telemetry collectors and the controller server as one foreground process."""

from __future__ import annotations

import argparse
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


ROOT = Path(__file__).parents[1]
TELEMETRY_ROOT = ROOT / "third_party" / "post-training-telemetry"
SSH_OPTIONS = (
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=10",
    "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=2",
)
TARGET_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class ConfigError(ValueError):
    pass


def load_plan(path: Path, cluster_name: str | None = None) -> dict[str, Any]:
    try:
        setup = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read setup {path}: {exc}") from exc
    if not isinstance(setup, dict) or not isinstance(setup.get("setup"), str):
        raise ConfigError("setup must contain a setup name")
    nodes = setup.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ConfigError("setup.nodes must contain at least one node")

    planned = []
    seen = set()
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise ConfigError(f"setup.nodes[{index}] must be an object")
        host = node.get("host")
        checkout = node.get("checkout")
        if not isinstance(host, str) or not host or host.startswith("-"):
            raise ConfigError(f"setup.nodes[{index}].host must be a safe non-empty string")
        if not isinstance(checkout, str) or not checkout.startswith("/"):
            raise ConfigError(f"setup.nodes[{index}].checkout must be an absolute path")
        address = host.rsplit("@", 1)[-1]
        if not TARGET_RE.fullmatch(address):
            raise ConfigError(f"cannot derive a telemetry address from host: {host}")
        if address in seen:
            raise ConfigError(f"duplicate telemetry address: {address}")
        seen.add(address)
        planned.append({"host": host, "address": address, "checkout": checkout})

    name = cluster_name or f"{setup['setup']}-cluster"
    if not TARGET_RE.fullmatch(name):
        raise ConfigError("cluster name may contain only letters, digits, dots, underscores, and hyphens")
    return {
        "cluster_name": name,
        "nodes": planned,
        "targets": ",".join(f"{node['address']}={node['address']}" for node in planned),
    }


def remote_commands(node: dict[str, str], duration: int, session: str) -> tuple[str, str]:
    q = shlex.quote
    telemetry_root = os.path.join(node["checkout"], "third_party", "post-training-telemetry")
    pidfile = f'.telemetry-orchestrator-{session}.pid'
    launch = (
        "set -eu; tools_dir=\"$HOME/telemetry-tools\"; output_dir=\"$HOME/telemetry-data\"; "
        f"pidfile=\"$output_dir/{pidfile}\"; mkdir -p \"$output_dir\"; cd {q(telemetry_root)}; "
        "setsid env "
        f"NODE_ADDR={q(node['address'])} NODE_NAME={q(node['address'])} "
        f"DURATION={duration} TOOLS_DIR=\"$tools_dir\" OUTPUT_DIR=\"$output_dir\" "
        "bash scripts/run_telemetry.sh node & child=$!; printf '%s\\n' \"$child\" > \"$pidfile\"; "
        "cleanup() { kill -TERM -- \"-$child\" 2>/dev/null || kill -TERM \"$child\" 2>/dev/null || true; "
        "wait \"$child\" 2>/dev/null || true; rm -f \"$pidfile\"; }; "
        "trap cleanup EXIT; trap 'exit 130' INT; trap 'exit 143' TERM; wait \"$child\""
    )
    cleanup = (
        "set -u; output_dir=\"$HOME/telemetry-data\"; "
        f"pidfile=\"$output_dir/{pidfile}\"; [ -s \"$pidfile\" ] || exit 0; "
        "pid=$(cat \"$pidfile\") || exit 1; "
        "kill -TERM -- \"-$pid\" 2>/dev/null || kill -TERM \"$pid\" 2>/dev/null || true; "
        "rm -f \"$pidfile\""
    )
    return launch, cleanup


def _stop(processes: list[Any]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def run_stack(
    plan: dict[str, Any],
    duration: int,
    environment: dict[str, str],
    *,
    popen: Callable[..., Any] = subprocess.Popen,
    run: Callable[..., Any] = subprocess.run,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    session = uuid.uuid4().hex
    processes: list[Any] = []
    cleanups: list[tuple[str, str]] = []
    try:
        for node in plan["nodes"]:
            launch, cleanup = remote_commands(node, duration, session)
            processes.append(popen(["ssh", *SSH_OPTIONS, node["host"], launch]))
            cleanups.append((node["host"], cleanup))
            print(f"Telemetry collector starting: {node['host']} ({node['address']})", flush=True)

        server_env = environment | {
            "CLUSTER_NAME": plan["cluster_name"],
            "DEMO_LIVE": "0",
            "SERVER_CONFIG_ONLY": "0",
            "TELEMETRY_TARGETS": plan["targets"],
        }
        processes.append(
            popen(
                ["bash", "scripts/run_telemetry.sh", "server"],
                cwd=TELEMETRY_ROOT,
                env=server_env,
            )
        )
        while True:
            for process in processes:
                code = process.poll()
                if code is not None:
                    return code
            sleep(0.2)
    except KeyboardInterrupt:
        return 130
    finally:
        for host, cleanup in cleanups:
            try:
                run(["ssh", *SSH_OPTIONS, host, cleanup], check=False, timeout=15)
            except (OSError, subprocess.SubprocessError) as exc:
                print(f"Telemetry cleanup was not confirmed on {host}: {exc}", file=sys.stderr)
        _stop(processes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", type=Path, required=True)
    parser.add_argument("--cluster-name")
    parser.add_argument("--duration", type=int, default=3600)
    args = parser.parse_args(argv)
    if args.duration < 1:
        parser.error("--duration must be at least 1")
    for name in ("TOOLS_DIR", "OUTPUT_DIR"):
        if not os.environ.get(name):
            parser.error(f"set {name} to a controller-local path")
    try:
        plan = load_plan(args.setup, args.cluster_name)
        return run_stack(plan, args.duration, dict(os.environ))
    except (ConfigError, OSError, subprocess.SubprocessError) as exc:
        print(f"telemetry orchestration error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
