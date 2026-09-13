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
    storage_variables = {item["name"] for item in payloads[2]["templating"]["list"]}
    assert {"storage_system", "storage_node", "ssd"} <= storage_variables
    storage_titles = {panel["title"] for panel in payloads[2]["panels"]}
    assert {
        "SSDs with critical warnings",
        "Maximum SSD temperature",
        "Maximum endurance used",
        "Minimum available spare",
        "NVMe media errors",
        "Lifetime host bytes written by SSD",
        "SSD inventory and SMART status",
    } <= storage_titles
    matrix = payloads[1]["panels"][0]
    assert "profiling_gpu_sample_timestamp_seconds" in matrix["targets"][0]["expr"]
    assert matrix["transformations"][0]["options"]["rowField"] == "node"


def test_server_config_accepts_an_arbitrary_named_target_list(tmp_path: Path) -> None:
    script = ROOT / "observability" / "scripts" / "run_spark_observability.sh"
    environment = os.environ | {
        "CLUSTER_NAME": "next-cluster",
        "SPARK_TARGETS": "trainer-0=10.0.0.10,rollout-0=rollout.example",
        "STORAGE_TARGETS": "storage-0=10.0.1.10,storage-1=storage.example",
        "STORAGE_SYSTEM": "3fs",
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
    assert "job_name: storage-smart" in config
    assert "targets: ['10.0.1.10:19633']" in config
    assert "targets: ['storage.example:19633']" in config
    assert "storage_system: 3fs" in config
    assert "nodename: storage-0" in config
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


def test_storage_role_starts_smartctl_exporter_with_slow_polling(tmp_path: Path) -> None:
    script = ROOT / "observability" / "scripts" / "run_spark_observability.sh"
    smartctl = tmp_path / "smartctl"
    exporter = tmp_path / "smartctl_exporter"
    arguments = tmp_path / "arguments.txt"
    smartctl.write_text("#!/usr/bin/env bash\nexit 0\n")
    exporter.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$EXPORTER_ARGS"\n')
    smartctl.chmod(0o755)
    exporter.chmod(0o755)
    environment = os.environ | {
        "NODE_ADDR": "127.0.0.1",
        "SMARTCTL": str(smartctl),
        "SMARTCTL_EXPORTER": str(exporter),
        "EXPORTER_ARGS": str(arguments),
        "OUTPUT_DIR": str(tmp_path / "monitoring"),
    }

    result = subprocess.run(
        ["bash", str(script), "storage"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert arguments.read_text().splitlines() == [
        f"--smartctl.path={smartctl}",
        "--smartctl.interval=60s",
        "--web.listen-address=127.0.0.1:19633",
    ]


def test_storage_role_wraps_smartctl_with_sudo_when_forced(tmp_path: Path) -> None:
    # /dev/nvmeN (the admin-passthrough device SMART needs) stays root:root
    # 0600 even when the sibling block device is disk-group readable, so
    # smartctl_exporter gets "Permission denied" and reports no SMART fields
    # as a plain user (confirmed on real Spark hardware). SMARTCTL_SUDO=1
    # forces the sudo wrapper without depending on the test host's own sudo
    # configuration.
    script = ROOT / "observability" / "scripts" / "run_spark_observability.sh"
    smartctl = tmp_path / "smartctl"
    exporter = tmp_path / "smartctl_exporter"
    arguments = tmp_path / "arguments.txt"
    smartctl.write_text("#!/usr/bin/env bash\nexit 0\n")
    exporter.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$EXPORTER_ARGS"\n')
    smartctl.chmod(0o755)
    exporter.chmod(0o755)
    output_dir = tmp_path / "monitoring"
    environment = os.environ | {
        "NODE_ADDR": "127.0.0.1",
        "SMARTCTL": str(smartctl),
        "SMARTCTL_EXPORTER": str(exporter),
        "SMARTCTL_SUDO": "1",
        "EXPORTER_ARGS": str(arguments),
        "OUTPUT_DIR": str(output_dir),
    }

    result = subprocess.run(
        ["bash", str(script), "storage"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    wrapper = output_dir / "smartctl-sudo"
    assert arguments.read_text().splitlines()[0] == f"--smartctl.path={wrapper}"
    assert os.access(wrapper, os.X_OK)
    wrapper_text = wrapper.read_text()
    assert "sudo -n" in wrapper_text
    assert str(smartctl) in wrapper_text


def test_storage_role_skips_sudo_when_smartctl_already_has_permission(tmp_path: Path) -> None:
    script = ROOT / "observability" / "scripts" / "run_spark_observability.sh"
    smartctl = tmp_path / "smartctl"
    exporter = tmp_path / "smartctl_exporter"
    arguments = tmp_path / "arguments.txt"
    smartctl.write_text(
        '#!/usr/bin/env bash\n'
        'case "$1" in\n'
        '  --scan) echo "/dev/nvme0 -d nvme # comment" ;;\n'
        '  -i) exit 0 ;;\n'
        '  *) exit 0 ;;\n'
        'esac\n'
    )
    exporter.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$EXPORTER_ARGS"\n')
    smartctl.chmod(0o755)
    exporter.chmod(0o755)
    environment = os.environ | {
        "NODE_ADDR": "127.0.0.1",
        "SMARTCTL": str(smartctl),
        "SMARTCTL_EXPORTER": str(exporter),
        "EXPORTER_ARGS": str(arguments),
        "OUTPUT_DIR": str(tmp_path / "monitoring"),
    }

    result = subprocess.run(
        ["bash", str(script), "storage"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert arguments.read_text().splitlines()[0] == f"--smartctl.path={smartctl}"


def test_dashboards_keep_matrix_and_freshness_scopes_separate() -> None:
    """A fresh rank/cluster must not mask another rank's stale or colliding cell."""
    for name in DASHBOARDS:
        payload = json.loads((ROOT / "observability/examples/observability" / name).read_text())
        variables = {v["name"]: v for v in payload["templating"]["list"]}
        for panel in payload["panels"]:
            for target in panel.get("targets", []):
                expr = target["expr"]
                assert 'cluster=~"$cluster"' in expr
                if "profiling_topology_" in expr:
                    assert "max by (cluster," in expr
                    assert '$node' not in expr  # The publisher need not be the selected node.
                if "$training_max_age" in expr:
                    assert variables["training_max_age"]["current"]["value"] == "300"
                    assert "and on(cluster, instance, run_id, framework, node, rank, local_rank)" in expr
                if "profiling_gpu_" in expr and "sample age" not in panel["title"].lower():
                    assert "profiling_gpu_sample_timestamp_seconds" in expr
                    assert "< 30" in expr
            if panel["title"] == "GPU allocation matrix":
                expr = panel["targets"][0]["expr"]
                assert '"cluster", "instance", "run_id", "framework", "rank"' in expr
                assert "$training_max_age" in expr
            if panel["title"] in {"Compute topology matrix", "Storage topology matrix"}:
                expr = panel["targets"][0]["expr"]
                assert '"cluster", "source"' in expr and '"cluster", "destination"' in expr
                options = panel["transformations"][0]["options"]
                assert (options["rowField"], options["columnField"]) == ("source_key", "destination_key")
            if panel["title"] == "Training sample age by rank":
                assert "max(" not in panel["targets"][0]["expr"]
                assert "$training_max_age" not in panel["targets"][0]["expr"]
            if panel["type"] == "stat":
                assert all(t.get("instant") for t in panel["targets"])
