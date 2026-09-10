"""Report availability and ownership class of the profiling tool stack."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class Tool:
    command: tuple[str, ...]
    layer: str
    ownership: str


TOOLS = {
    "prometheus": Tool(("prometheus", "--version"), "time_series", "open_source"),
    "node_exporter": Tool(("node_exporter", "--version"), "host", "open_source"),
    "dcgm_exporter": Tool(("dcgm-exporter", "--version"), "gpu", "open_source_vendor_dependency"),
    "grafana": Tool(("grafana-server", "--version"), "visualization", "open_source"),
    "otel_collector": Tool(("otelcol-contrib", "--version"), "distributed_trace", "open_source"),
    "nccl_tests": Tool(("all_reduce_perf", "-h"), "network_baseline", "open_source_vendor_dependency"),
    "docker": Tool(("docker", "compose", "version"), "lab_runtime", "open_source_components"),
    "podman": Tool(("podman", "compose", "version"), "lab_runtime", "open_source"),
    "nsight_systems": Tool(("nsys", "--version"), "diagnostic_fallback", "vendor"),
    "nsight_compute": Tool(("ncu", "--version"), "diagnostic_fallback", "vendor"),
}


def _run_version(command: tuple[str, ...]) -> tuple[bool, str | None, str | None]:
    executable = shutil.which(command[0])
    if executable is None:
        return False, None, None
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=10)
        output = (completed.stdout or completed.stderr).strip().splitlines()
        version = output[0] if output else f"exit code {completed.returncode}"
    except subprocess.SubprocessError as error:
        version = str(error)
    return True, executable, version


def inspect_tools() -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for name, tool in TOOLS.items():
        available, path, version = _run_version(tool.command)
        result[name] = {
            "available": available,
            "path": path,
            "version": version,
            "layer": tool.layer,
            "ownership": tool.ownership,
        }

    torch_available = importlib.util.find_spec("torch") is not None
    torch_version = importlib.metadata.version("torch") if torch_available else None
    result["pytorch_profiler"] = {
        "available": torch_available,
        "path": None,
        "version": torch_version,
        "layer": "selected_trace",
        "ownership": "open_source_vendor_dependency_on_cuda",
    }
    return result


def main() -> None:
    print(json.dumps(inspect_tools(), indent=2))


if __name__ == "__main__":
    main()
