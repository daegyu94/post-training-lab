import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[2]
DASHBOARDS = (
    "spark-resources.json",
    "compute-communication.json",
    "data-storage.json",
)


def test_spark_dashboards_have_unique_uids_and_shared_cluster_filter() -> None:
    payloads = [
        json.loads((ROOT / "observability" / "examples" / "observability" / name).read_text())
        for name in DASHBOARDS
    ]

    assert [payload["uid"] for payload in payloads] == [
        "spark-profiling",
        "post-training-compute-communication",
        "post-training-data-storage",
    ]
    for payload in payloads:
        assert {item["name"] for item in payload["templating"]["list"]} >= {"cluster", "node"}
        assert all(
            panel.get("datasource", {}).get("uid") == "spark-prometheus"
            for panel in payload["panels"]
            if panel["type"] != "text"
        )
    assert "${node:queryparam}" in payloads[0]["links"][0]["url"]
    assert "${run_id:queryparam}" in payloads[2]["links"][0]["url"]
    matrix = payloads[1]["panels"][0]
    assert "profiling_gpu_sample_timestamp_seconds" in matrix["targets"][0]["expr"]
    assert matrix["transformations"][0]["options"]["rowField"] == "node"


def test_server_config_accepts_an_arbitrary_named_target_list(tmp_path: Path) -> None:
    script = ROOT / "observability" / "scripts" / "run_spark_observability.sh"
    environment = os.environ | {
        "CLUSTER_NAME": "next-cluster",
        "SPARK_TARGETS": "trainer-0=10.0.0.10,rollout-0=rollout.example",
        "SERVER_CONFIG_ONLY": "1",
        "OUTPUT_DIR": str(tmp_path / "monitoring"),
    }

    result = subprocess.run(
        ["bash", str(script), "server"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    config = (tmp_path / "monitoring" / "prometheus.yml").read_text()
    assert "targets: ['10.0.0.10:19100']" in config
    assert "targets: ['rollout.example:19100']" in config
    assert "cluster: next-cluster" in config
    assert "nodename: trainer-0" in config
    assert "nodename: rollout-0" in config
    assert {
        path.name for path in (tmp_path / "monitoring" / "dashboards").iterdir()
    } == set(DASHBOARDS)


def test_server_config_rejects_duplicate_target_names(tmp_path: Path) -> None:
    script = ROOT / "observability" / "scripts" / "run_spark_observability.sh"
    environment = os.environ | {
        "SPARK_TARGETS": "worker=10.0.0.10,worker=10.0.0.11",
        "SERVER_CONFIG_ONLY": "1",
        "OUTPUT_DIR": str(tmp_path / "monitoring"),
    }

    result = subprocess.run(
        ["bash", str(script), "server"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "node names must be unique" in result.stderr
