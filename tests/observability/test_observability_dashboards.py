import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[2]
DASHBOARDS = (
    "run-overview.json",
    "compute-communication.json",
    "data-storage.json",
)


def test_observability_dashboards_have_unique_uids_and_shared_cluster_filter() -> None:
    payloads = [
        json.loads((ROOT / "observability" / "examples" / "observability" / name).read_text())
        for name in DASHBOARDS
    ]

    assert [payload["uid"] for payload in payloads] == [
        "observability-overview",
        "post-training-compute-communication",
        "post-training-data-storage",
    ]
    for payload in payloads:
        assert {item["name"] for item in payload["templating"]["list"]} >= {"cluster", "node"}
        assert all(
            panel.get("datasource", {}).get("uid") == "observability-prometheus"
            for panel in payload["panels"]
            if panel["type"] != "text"
        )
    assert "${node:queryparam}" in payloads[0]["links"][0]["url"]
    assert any(link["title"] == "Run Logs" for link in payloads[0]["links"])
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

    logs = json.loads(
        (ROOT / "observability/examples/observability/run-logs.json").read_text()
    )
    assert logs["uid"] == "post-training-run-logs"
    assert logs["panels"][0]["datasource"]["uid"] == "observability-loki"
    assert "| unpack | run_id=~" in logs["panels"][0]["targets"][0]["expr"]


def test_server_config_accepts_an_arbitrary_named_target_list(tmp_path: Path) -> None:
    script = ROOT / "observability" / "scripts" / "run_observability.sh"
    environment = os.environ | {
        "CLUSTER_NAME": "next-cluster",
        "OBSERVABILITY_TARGETS": "trainer-0=10.0.0.10,rollout-0=rollout.example",
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
    script = ROOT / "observability" / "scripts" / "run_observability.sh"
    environment = os.environ | {
        "OBSERVABILITY_TARGETS": "worker=10.0.0.10,worker=10.0.0.11",
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


def test_server_log_config_provisions_loki_and_dashboard(tmp_path: Path) -> None:
    script = ROOT / "observability" / "scripts" / "run_observability.sh"
    loki = tmp_path / "loki"
    loki.write_text("#!/usr/bin/env bash\nexit 0\n")
    loki.chmod(0o755)
    output = tmp_path / "monitoring"
    environment = os.environ | {
        "CLUSTER_NAME": "spark-cluster",
        "OBSERVABILITY_TARGETS": "spark1=10.0.0.10,spark2=10.0.0.11",
        "ENABLE_LOGS": "1",
        "LOKI": str(loki),
        "LOKI_LISTEN_ADDR": "192.168.0.1",
        "SERVER_CONFIG_ONLY": "1",
        "OUTPUT_DIR": str(output),
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
    assert "http_listen_address: 192.168.0.1" in (output / "loki.yaml").read_text()
    assert "retention_period: 168h" in (output / "loki.yaml").read_text()
    assert "uid: observability-loki" in (
        output / "provisioning/datasources/default.yaml"
    ).read_text()
    assert "url: http://192.168.0.1:13100" in (
        output / "provisioning/datasources/default.yaml"
    ).read_text()
    assert (output / "dashboards/run-logs.json").is_file()


def test_node_log_config_accepts_multiple_local_workload_roots(tmp_path: Path) -> None:
    script = ROOT / "observability" / "scripts" / "run_observability.sh"
    alloy = tmp_path / "alloy"
    alloy.write_text("#!/usr/bin/env bash\nexit 0\n")
    alloy.chmod(0o755)
    trl = tmp_path / "trl"
    verl = tmp_path / "verl"
    trl.mkdir()
    verl.mkdir()
    output = tmp_path / "monitoring"
    environment = os.environ | {
        "NODE_ADDR": "127.0.0.1",
        "NODE_NAME": "spark1",
        "CLUSTER_NAME": "spark-cluster",
        "LOKI_PUSH_URL": "http://192.168.0.1:13100/loki/api/v1/push",
        "OBSERVABILITY_LOG_ROOTS": f"trl={trl},verl={verl}",
        "ALLOY": str(alloy),
        "NODE_CONFIG_ONLY": "1",
        "OUTPUT_DIR": str(output),
    }

    result = subprocess.run(
        ["bash", str(script), "node"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    config = (output / "alloy.alloy").read_text()
    assert f'{trl}/*/logs/**/*.log' in config
    assert f'{verl}/*/logs/**/*.log' in config
    assert 'workload = "trl"' in config
    assert 'workload = "verl"' in config
    assert 'labels = ["filename", "run_id", "log_file"]' in config
    assert 'ignore_older_than = "24h"' in config


def test_node_log_config_rejects_nfs_roots(tmp_path: Path) -> None:
    script = ROOT / "observability" / "scripts" / "run_observability.sh"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    findmnt = bin_dir / "findmnt"
    findmnt.write_text("#!/usr/bin/env bash\necho nfs4\n")
    findmnt.chmod(0o755)
    root = tmp_path / "logs"
    root.mkdir()
    result = subprocess.run(
        ["bash", str(script), "node"],
        cwd=ROOT,
        env=os.environ
        | {
            "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
            "NODE_ADDR": "127.0.0.1",
            "LOKI_PUSH_URL": "http://192.168.0.1:13100/loki/api/v1/push",
            "OBSERVABILITY_LOG_ROOTS": f"trl={root}",
            "NODE_CONFIG_ONLY": "1",
            "OUTPUT_DIR": str(tmp_path / "monitoring"),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "log root must be node-local" in result.stderr


def test_storage_role_starts_smartctl_exporter_with_slow_polling(tmp_path: Path) -> None:
    script = ROOT / "observability" / "scripts" / "run_observability.sh"
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
    # as a plain user. SMARTCTL_SUDO=1
    # forces the sudo wrapper without depending on the test host's own sudo
    # configuration.
    script = ROOT / "observability" / "scripts" / "run_observability.sh"
    smartctl = tmp_path / "smartctl"
    exporter = tmp_path / "smartctl_exporter"
    arguments = tmp_path / "arguments.txt"
    smartctl.write_text("#!/usr/bin/env bash\nexit 0\n")
    exporter.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$EXPORTER_ARGS"\n')
    smartctl.chmod(0o755)
    exporter.chmod(0o755)
    sudo = tmp_path / "sudo"
    sudo.write_text('#!/usr/bin/env bash\nshift\nexec "$@"\n')
    sudo.chmod(0o755)
    output_dir = tmp_path / "monitoring"
    environment = os.environ | {
        "NODE_ADDR": "127.0.0.1",
        "SMARTCTL": str(smartctl),
        "SMARTCTL_EXPORTER": str(exporter),
        "SMARTCTL_SUDO": "1",
        "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
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
    script = ROOT / "observability" / "scripts" / "run_observability.sh"
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
                    assert "and on(cluster, instance, run_id, producer, role, worker_id, node, local_rank)" in expr
                if "profiling_gpu_" in expr and "sample age" not in panel["title"].lower():
                    assert "profiling_gpu_sample_timestamp_seconds" in expr
                    assert "< 30" in expr
            if panel["title"] == "GPU allocation matrix":
                expr = panel["targets"][0]["expr"]
                assert '"cluster", "instance", "run_id", "producer", "role", "worker_id"' in expr
                assert "$training_max_age" in expr
            if panel["title"] in {"Compute topology matrix", "Storage topology matrix"}:
                expr = panel["targets"][0]["expr"]
                assert '"cluster", "source"' in expr and '"cluster", "destination"' in expr
                options = panel["transformations"][0]["options"]
                assert (options["rowField"], options["columnField"]) == ("source_key", "destination_key")
            if panel["title"] == "Training sample age by worker":
                assert "max(" not in panel["targets"][0]["expr"]
                assert "$training_max_age" not in panel["targets"][0]["expr"]
            if panel["type"] == "stat":
                assert all(t.get("instant") for t in panel["targets"])


def test_storage_role_fails_before_exporter_when_sudo_denied(tmp_path: Path) -> None:
    smartctl = tmp_path / "smartctl"
    exporter = tmp_path / "exporter"
    sudo = tmp_path / "sudo"
    marker = tmp_path / "started"
    smartctl.write_text("#!/bin/sh\nexit 0\n")
    sudo.write_text("#!/bin/sh\nexit 1\n")
    exporter.write_text('#!/bin/sh\ntouch "$MARKER"\n')
    for path in (smartctl, exporter, sudo):
        path.chmod(0o755)
    env = os.environ | {
        "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        "SMARTCTL": str(smartctl),
        "SMARTCTL_EXPORTER": str(exporter),
        "SMARTCTL_SUDO": "1",
        "NODE_ADDR": "127.0.0.1",
        "OUTPUT_DIR": str(tmp_path / "output"),
        "MARKER": str(marker),
    }
    result = subprocess.run(
        ["bash", str(ROOT / "observability/scripts/run_observability.sh"), "storage"],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "preflight failed" in result.stderr
    assert not marker.exists()


def test_healthy_ssd_count_preserves_zero_without_faking_missing_data() -> None:
    dashboard = json.loads(
        (ROOT / "observability/examples/observability/data-storage.json").read_text()
    )
    panel = next(p for p in dashboard["panels"] if p["title"] == "SSDs with critical warnings")
    expr = panel["targets"][0]["expr"]
    assert expr.startswith("sum(") and "!= bool 0" in expr
    assert "or vector(0)" not in expr
