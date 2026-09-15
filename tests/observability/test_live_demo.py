from pathlib import Path

from profiling_lab.live_demo import Demo, prometheus_config


ROOT = Path(__file__).parents[2]


def test_demo_matches_the_b300_and_storage_topology() -> None:
    demo = Demo(ROOT / "observability" / "examples" / "live-demo")

    gpu = demo.metrics("gpu-node-0")
    storage = demo.metrics("storage-node-0")
    topology = demo.metrics("topology")
    config = prometheus_config(demo, "127.0.0.1:19110")

    assert len([sample for sample in gpu if sample.name == "profiling_gpu_utilization_percent"]) == 8
    assert len([sample for sample in storage if sample.name == "smartctl_device"]) == 4
    assert len([sample for sample in topology if sample.labels.get("role") == "B300"]) == 32
    assert len([sample for sample in topology if sample.labels.get("role") == "ssd"]) == 32
    assert config.count("__metrics_path__:") == 13
    assert "job_name: observability" in config
    assert "job_name: storage-smart" in config
