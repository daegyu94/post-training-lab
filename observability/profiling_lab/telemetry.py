"""Collect supported nvidia-smi fields without substituting zero for N/A."""

import argparse
import csv
import json
import math
import socket
import subprocess
import time
from pathlib import Path

from profiling_lab.prometheus_textfile import GaugeSample, write_gauges


def optional_number(value: str) -> float | None:
    try:
        result = float(value.strip())
        return result if math.isfinite(result) else None
    except ValueError:
        return None


def query(kind: str, fields: list[str]) -> list[list[str]]:
    result = subprocess.run(
        ["nvidia-smi", f"--query-{kind}={','.join(fields)}", "--format=csv,noheader,nounits"],
        check=True, text=True, capture_output=True, timeout=10,
    )
    return list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))


def snapshot() -> dict:
    fields = ["index", "utilization.gpu", "power.draw", "temperature.gpu", "clocks.sm", "memory.used", "memory.total"]
    gpus = []
    for row in query("gpu", fields):
        gpu = {key: optional_number(value) for key, value in zip(fields, row)}
        gpu["unavailable_fields"] = [key for key in fields if gpu[key] is None]
        gpus.append(gpu)
    processes = []
    for row in query("compute-apps", ["gpu_uuid", "pid", "process_name", "used_gpu_memory"]):
        processes.append({"gpu_uuid": row[0], "pid": int(row[1]), "process_name": row[2],
                          "used_gpu_memory_mib": optional_number(row[3])})
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable", "MemFree", "Buffers", "Cached", "SwapTotal", "SwapFree"}:
            memory[key + "_bytes"] = int(value.split()[0]) * 1024
    return {"timestamp": time.time(), "hostname": socket.gethostname(), "host_memory": memory,
            "gpus": gpus, "compute_processes": processes,
            "null_reason": "nvidia-smi field unavailable; not zero"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--textfile-dir", type=Path)
    parser.add_argument("--duration", type=float, default=120)
    parser.add_argument("--interval", type=float, default=1)
    args = parser.parse_args()
    if args.duration <= 0 or args.interval <= 0:
        parser.error("duration and interval must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.duration
    with args.output.open("x") as output:
        while time.monotonic() < deadline:
            value = snapshot()
            output.write(json.dumps(value) + "\n")
            output.flush()
            if args.textfile_dir:
                samples = [GaugeSample("profiling_gpu_sample_timestamp_seconds", "Last successful GPU sample.", value["timestamp"])]
                for gpu in value["gpus"]:
                    for field, name in [("utilization.gpu", "utilization_percent"), ("power.draw", "power_watts"), ("temperature.gpu", "temperature_celsius"), ("clocks.sm", "sm_clock_mhz")]:
                        if gpu[field] is not None:
                            samples.append(GaugeSample(f"profiling_gpu_{name}", f"nvidia-smi {field}.", gpu[field], {"gpu": str(int(gpu["index"]))}))
                for process in value["compute_processes"]:
                    if process["used_gpu_memory_mib"] is not None:
                        samples.append(GaugeSample(
                            "profiling_gpu_process_memory_bytes",
                            "nvidia-smi compute-process GPU memory; not total unified memory.",
                            process["used_gpu_memory_mib"] * 1024**2,
                            {"pid": str(process["pid"]), "gpu_uuid": process["gpu_uuid"]},
                        ))
                write_gauges(args.textfile_dir, "gpu.prom", samples)
            time.sleep(min(args.interval, max(0, deadline - time.monotonic())))


if __name__ == "__main__":
    main()
