"""Serve a bounded synthetic B300 cluster for the existing Grafana dashboards."""

from __future__ import annotations

import argparse
import json
import math
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from profiling_lab.prometheus_textfile import GaugeSample, format_gauges


_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_GIB = 1024 ** 3


def load_topology(directory: Path) -> tuple[dict, dict]:
    """Load the JSON-compatible YAML fixtures without a YAML dependency."""
    try:
        gpu = json.loads((directory / "gpu_topology.yaml").read_text(encoding="utf-8"))
        storage = json.loads((directory / "storage_topology.yaml").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid live-demo topology: {error}") from error
    for key, payload, nodes, count in (
        ("GPU", gpu, "gpu_nodes", "gpus_per_node"),
        ("storage", storage, "storage_nodes", "ssds_per_node"),
    ):
        if (not isinstance(payload.get(nodes), list) or not payload[nodes]
                or not all(isinstance(node, str) and _NAME.fullmatch(node) for node in payload[nodes])
                or not isinstance(payload.get(count), int) or payload[count] <= 0):
            raise ValueError(f"invalid {key} topology")
    if not isinstance(gpu.get("gpu_model"), str) or not isinstance(gpu.get("intra_node_interconnect"), str):
        raise ValueError("invalid GPU topology")
    for payload in (gpu, storage):
        network = payload.get("network")
        if (not isinstance(network, dict) or not isinstance(network.get("transport"), str)
                or not isinstance(network.get("bandwidth_gbps"), (int, float))
                or network["bandwidth_gbps"] <= 0):
            raise ValueError("invalid network topology")
    return gpu, storage


def _phase(elapsed: float) -> tuple[str, dict[str, float]]:
    cycle = elapsed % 100
    if cycle < 35:
        return "training", {"gpu": 94, "tokens": 7600, "step": 1.1, "read": .5, "write": .2, "rx": 180, "tx": 180, "busy": .10}
    if cycle < 55:
        return "data_wait", {"gpu": 61, "tokens": 4700, "step": 1.7, "read": 6, "write": .2, "rx": 420, "tx": 150, "busy": .82}
    if cycle < 75:
        return "collective", {"gpu": 75, "tokens": 5800, "step": 1.4, "read": .5, "write": .2, "rx": 330, "tx": 330, "busy": .15}
    if cycle < 90:
        return "checkpoint", {"gpu": 68, "tokens": 5100, "step": 1.6, "read": .3, "write": 5, "rx": 160, "tx": 320, "busy": .68}
    return "recovery", {"gpu": 87, "tokens": 6800, "step": 1.2, "read": 1, "write": .5, "rx": 220, "tx": 220, "busy": .24}


class Demo:
    def __init__(self, topology_dir: Path) -> None:
        self.gpu, self.storage = load_topology(topology_dir)
        self.network = f"{self.gpu['network']['transport']}-{self.gpu['network']['bandwidth_gbps']:g}Gbps"
        if self.network != f"{self.storage['network']['transport']}-{self.storage['network']['bandwidth_gbps']:g}Gbps":
            raise ValueError("GPU and storage network topology must match")
        self.started = time.monotonic()
        self.lock = threading.Lock()
        self.counters: dict[tuple[str, str], tuple[float, float]] = {}

    def _counter(self, node: str, name: str, rate: float, now: float) -> float:
        key = (node, name)
        value, previous = self.counters.get(key, (1_000_000_000.0, now))
        value += max(0, now - previous) * rate
        self.counters[key] = value, now
        return value

    def metrics(self, endpoint: str) -> list[GaugeSample]:
        now = time.monotonic()
        phase, values = _phase(now - self.started)
        with self.lock:
            if endpoint in self.gpu["gpu_nodes"]:
                return self._gpu_node(endpoint, now, phase, values)
            if endpoint in self.storage["storage_nodes"]:
                return self._storage_node(endpoint, now, phase, values)
            if endpoint == "topology":
                return self._topology()
        raise ValueError(f"unknown demo endpoint: {endpoint}")

    def _gpu_node(self, node: str, now: float, phase: str, value: dict[str, float]) -> list[GaugeSample]:
        samples = [
            GaugeSample("profiling_gpu_sample_timestamp_seconds", "Synthetic GPU sample timestamp.", time.time()),
            GaugeSample("node_memory_MemAvailable_bytes", "Synthetic host memory available.", 760 * _GIB),
            GaugeSample("node_memory_SwapTotal_bytes", "Synthetic host swap total.", 32 * _GIB),
            GaugeSample("node_memory_SwapFree_bytes", "Synthetic host swap free.", 32 * _GIB),
            GaugeSample("live_demo_phase_info", "Current synthetic demo phase.", 1, {"phase": phase}),
        ]
        read_bytes = value["read"] * _GIB
        write_bytes = value["write"] * _GIB
        for name, rate in (
            ("node_disk_read_bytes_total", read_bytes),
            ("node_disk_written_bytes_total", write_bytes),
            ("node_disk_reads_completed_total", read_bytes / (256 * 1024)),
            ("node_disk_writes_completed_total", write_bytes / (256 * 1024)),
            ("node_disk_io_time_seconds_total", value["busy"]),
        ):
            samples.append(GaugeSample(name, "Synthetic disk counter.", self._counter(node, name, rate, now), {"device": "nvme0n1"}))
        for name, rate in (
            ("node_network_receive_bytes_total", value["rx"] * 1e9 / 8),
            ("node_network_transmit_bytes_total", value["tx"] * 1e9 / 8),
        ):
            samples.append(GaugeSample(name, "Synthetic RoCE counter.", self._counter(node, name, rate, now), {"device": "roce0"}))
        samples += [
            GaugeSample("node_filesystem_size_bytes", "Synthetic data filesystem size.", 64 * 1024 * _GIB, {"mountpoint": "/mnt/data", "fstype": "xfs"}),
            GaugeSample("node_filesystem_avail_bytes", "Synthetic data filesystem free space.", 39 * 1024 * _GIB, {"mountpoint": "/mnt/data", "fstype": "xfs"}),
        ]
        for rank in range(self.gpu["gpus_per_node"]):
            wobble = math.sin(now / 4 + rank) * 2
            labels = {"run_id": "live-demo", "framework": "synthetic", "node": node, "rank": str(rank), "local_rank": str(rank)}
            samples += [
                GaugeSample("profiling_gpu_utilization_percent", "Synthetic GPU utilization.", max(0, min(100, value["gpu"] + wobble)), {"gpu": str(rank)}),
                GaugeSample("profiling_gpu_power_watts", "Synthetic GPU power.", 820 + value["gpu"] * 3, {"gpu": str(rank)}),
                GaugeSample("profiling_gpu_temperature_celsius", "Synthetic GPU temperature.", 54 + value["gpu"] * .2 + wobble, {"gpu": str(rank)}),
                GaugeSample("profiling_gpu_sm_clock_mhz", "Synthetic GPU SM clock.", 1800 + value["gpu"] * 5, {"gpu": str(rank)}),
                GaugeSample("profiling_gpu_process_memory_bytes", "Synthetic training process GPU memory.", 144 * _GIB, {"pid": str(9000 + rank), "gpu_uuid": f"DEMO-{node}-{rank}"}),
                GaugeSample("training_sample_timestamp_seconds", "Synthetic training sample timestamp.", time.time(), labels),
                GaugeSample("training_step", "Synthetic training step.", int(now - self.started), labels),
                GaugeSample("training_tokens_per_second", "Synthetic rank throughput.", value["tokens"], labels),
                GaugeSample("training_step_time_seconds", "Synthetic training step duration.", value["step"], labels),
                GaugeSample("training_loss", "Synthetic training loss.", 1.4 + rank * .01, labels),
                GaugeSample("training_gpu_allocation", "Synthetic rank GPU allocation.", 1, {**labels, "gpu": str(rank)}),
            ]
            timers = {"forward": .28, "backward": .47, "communication": .12 if phase != "collective" else .48,
                      "data_loader": .08 if phase != "data_wait" else .62, "checkpoint": .01 if phase != "checkpoint" else .40}
            samples.extend(GaugeSample("training_timer_seconds", "Synthetic rank timer.", timer, {**labels, "timer": name}) for name, timer in timers.items())
        return samples

    def _storage_node(self, node: str, now: float, phase: str, value: dict[str, float]) -> list[GaugeSample]:
        samples = []
        for index in range(self.storage["ssds_per_node"]):
            labels = {"device": f"nvme{index}n1"}
            samples += [
                GaugeSample("smartctl_device", "Synthetic SSD inventory.", 1, {**labels, "protocol": "NVMe", "model_name": "Demo SSD", "firmware_version": "1.0"}),
                GaugeSample("smartctl_device_smart_status", "Synthetic SSD SMART status.", 1, labels),
                GaugeSample("smartctl_device_smartctl_exit_status", "Synthetic smartctl exit status.", 0, labels),
                GaugeSample("smartctl_device_critical_warning", "Synthetic SSD critical warning.", 0, labels),
                GaugeSample("smartctl_device_temperature", "Synthetic SSD temperature.", 38 + index, {**labels, "temperature_type": "current"}),
                GaugeSample("smartctl_device_percentage_used", "Synthetic SSD endurance used.", 3 + index, labels),
                GaugeSample("smartctl_device_available_spare", "Synthetic SSD available spare.", 100 - index, labels),
                GaugeSample("smartctl_device_media_errors", "Synthetic SSD media errors.", 0, labels),
                GaugeSample("smartctl_device_bytes_written", "Synthetic SSD host bytes written.", self._counter(node, f"ssd-{index}", value["write"] * _GIB / self.storage["ssds_per_node"], now), labels),
            ]
        return samples

    def _topology(self) -> list[GaugeSample]:
        samples = []
        for node in self.gpu["gpu_nodes"]:
            samples.append(GaugeSample("profiling_topology_component_info", "Synthetic topology component.", 1, {"kind": "compute", "component": node, "role": "gpu-node"}))
            for gpu in range(self.gpu["gpus_per_node"]):
                component = f"{node}/gpu-{gpu}"
                samples.append(GaugeSample("profiling_topology_component_info", "Synthetic topology component.", 1, {"kind": "compute", "component": component, "role": self.gpu["gpu_model"]}))
                for peer in range(self.gpu["gpus_per_node"]):
                    if gpu != peer:
                        samples.append(GaugeSample("profiling_topology_edge_info", "Synthetic topology edge.", 1, {"kind": "compute", "source": component, "destination": f"{node}/gpu-{peer}", "relation": self.gpu["intra_node_interconnect"]}))
            samples.append(GaugeSample("profiling_topology_edge_info", "Synthetic topology edge.", 1, {"kind": "compute", "source": node, "destination": "roce-fabric", "relation": self.network}))
        samples.append(GaugeSample("profiling_topology_component_info", "Synthetic topology component.", 1, {"kind": "compute", "component": "roce-fabric", "role": "network"}))
        for node in self.storage["storage_nodes"]:
            samples.append(GaugeSample("profiling_topology_component_info", "Synthetic topology component.", 1, {"kind": "storage", "component": node, "role": "storage-node"}))
            for ssd in range(self.storage["ssds_per_node"]):
                samples += [
                    GaugeSample("profiling_topology_component_info", "Synthetic topology component.", 1, {"kind": "storage", "component": f"{node}/nvme{ssd}n1", "role": "ssd"}),
                    GaugeSample("profiling_topology_edge_info", "Synthetic topology edge.", 1, {"kind": "storage", "source": node, "destination": f"{node}/nvme{ssd}n1", "relation": "attached"}),
                ]
            samples.append(GaugeSample("profiling_topology_edge_info", "Synthetic topology edge.", 1, {"kind": "storage", "source": node, "destination": "roce-fabric", "relation": self.network}))
        samples.append(GaugeSample("profiling_topology_component_info", "Synthetic topology component.", 1, {"kind": "storage", "component": "roce-fabric", "role": "network"}))
        return samples


def prometheus_config(demo: Demo, address: str, cluster: str = "demo-b300") -> str:
    def targets(job: str, endpoints: list[str], extra: str = "") -> list[str]:
        lines = [f"  - job_name: {job}", "    static_configs:"]
        for endpoint in endpoints:
            lines += [f"      - targets: ['{address}']", "        labels:", f"          cluster: {cluster}", f"          instance: {endpoint}", f"          __metrics_path__: /metrics/{endpoint}"]
            if extra:
                lines.append(extra)
        return lines
    lines = ["global:", "  scrape_interval: 2s", "scrape_configs:"]
    lines += targets("spark", [*demo.gpu["gpu_nodes"], "topology"])
    lines += targets("storage-smart", demo.storage["storage_nodes"], "          storage_system: demo")
    return "\n".join(lines) + "\n"


def serve(demo: Demo, listen: str) -> None:
    host, separator, raw_port = listen.rpartition(":")
    if not separator or not host or not raw_port.isdigit():
        raise ValueError("--listen must be host:port")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            endpoint = urlparse(self.path).path.removeprefix("/metrics/")
            try:
                body = format_gauges(demo.metrics(endpoint)).encode()
            except ValueError:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_: object) -> None:
            pass

    ThreadingHTTPServer((host, int(raw_port)), Handler).serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology-dir", type=Path, default=Path(__file__).parents[1] / "examples" / "live-demo")
    parser.add_argument("--listen", default="127.0.0.1:19110")
    parser.add_argument("--write-prometheus-config", type=Path)
    args = parser.parse_args()
    try:
        demo = Demo(args.topology_dir)
        if args.write_prometheus_config:
            args.write_prometheus_config.write_text(prometheus_config(demo, args.listen), encoding="utf-8")
            return
        serve(demo, args.listen)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
