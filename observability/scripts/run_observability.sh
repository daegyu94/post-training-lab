#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
role="${1:?Use node, storage, or server}"
system="$(uname -s)"
[[ "$system" == Linux ]] || { echo "Unsupported operating system: $system (expected Linux)" >&2; exit 2; }
machine="$(uname -m)"
case "$machine" in
  aarch64|arm64) release_arch=arm64 ;;
  x86_64|amd64) release_arch=amd64 ;;
  *) echo "Unsupported architecture: $machine (expected ARM64 or x86_64)" >&2; exit 2 ;;
esac
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
tools_dir="${TOOLS_DIR:-$HOME/.local/share/observability-tools}"
output_dir="${OUTPUT_DIR:-$PWD/artifacts/observability/monitoring-$(hostname)}"
mkdir -p "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"
pids=()
cleanup() {
  trap - EXIT INT TERM
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
  if [[ "$role" == node ]]; then rm -f "$output_dir/textfile/gpu.prom" "$output_dir/textfile/observatory.prom"; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
start_smartctl_exporter() {
  : "${NODE_ADDR:?Set NODE_ADDR to this storage node management address}"
  local exporter smartctl_path smartctl_cmd sudo_mode needs_sudo first_device scan_output
  exporter="${SMARTCTL_EXPORTER:-$tools_dir/smartctl_exporter-0.14.0.linux-$release_arch/smartctl_exporter}"
  if [[ ! -x "$exporter" ]]; then
    echo "smartctl_exporter not found or not executable: $exporter" >&2
    exit 1
  fi
  if ! smartctl_path="$(command -v "${SMARTCTL:-smartctl}")"; then
    echo "smartctl is required for SSD health collection" >&2
    exit 1
  fi
  smartctl_cmd="$smartctl_path"
  sudo_mode="${SMARTCTL_SUDO:-auto}"
  case "$sudo_mode" in
    auto|0|1) ;;
    *) echo "SMARTCTL_SUDO must be auto, 0, or 1" >&2; exit 2 ;;
  esac
  needs_sudo=0
  if [[ "$sudo_mode" == 1 ]]; then
    needs_sudo=1
  elif [[ "$sudo_mode" == auto ]]; then
    # NVMe SMART needs an admin-passthrough ioctl on the controller char
    # device (/dev/nvmeN), which stays root:root mode 0600 regardless of the
    # sibling block device's group (/dev/nvmeXn1, disk-group readable).
    # A plain user can receive "Permission denied" and no SMART fields.
    if ! scan_output="$("$smartctl_path" --scan 2>/dev/null)"; then
      needs_sudo=1
    fi
    first_device="$(awk 'NR==1{print $1}' <<< "$scan_output")"
    if [[ -n "$first_device" ]] && ! "$smartctl_path" -i "$first_device" >/dev/null 2>&1; then
      needs_sudo=1
    fi
  fi
  if [[ "$needs_sudo" == 1 ]]; then
    if ! command -v sudo >/dev/null 2>&1; then
      echo "smartctl needs root for NVMe SMART queries but sudo is unavailable; set SMARTCTL_SUDO=0 or grant access another way" >&2
      exit 1
    fi
    if ! sudo -n "$smartctl_path" --scan >/dev/null 2>&1; then
      echo "passwordless sudo smartctl preflight failed; check sudo permissions before starting SSD health collection" >&2
      exit 1
    fi
    smartctl_cmd="$output_dir/smartctl-sudo"
    printf '#!/usr/bin/env bash\nexec sudo -n %q "$@"\n' "$smartctl_path" > "$smartctl_cmd"
    chmod 0755 "$smartctl_cmd"
  fi
  "$exporter" \
    --smartctl.path="$smartctl_cmd" \
    --smartctl.interval="${SMARTCTL_INTERVAL:-60s}" \
    --web.listen-address="$NODE_ADDR:${SMARTCTL_PORT:-19633}" \
    > "$output_dir/smartctl-exporter.log" 2>&1 &
  pids+=("$!")
}
if [[ "$role" == node ]]; then
  : "${NODE_ADDR:?Set NODE_ADDR to this node management address}"
  if [[ -n "${LOKI_PUSH_URL:-}" || -n "${OBSERVABILITY_LOG_ROOTS:-}" ]]; then
    : "${LOKI_PUSH_URL:?Set LOKI_PUSH_URL when OBSERVABILITY_LOG_ROOTS is set}"
    : "${OBSERVABILITY_LOG_ROOTS:?Set OBSERVABILITY_LOG_ROOTS when LOKI_PUSH_URL is set}"
    cluster_name="${CLUSTER_NAME:-observability-cluster}"
    node_name="${NODE_NAME:-$(hostname -s)}"
    if [[ ! "$cluster_name" =~ ^[A-Za-z0-9_.-]+$ || ! "$node_name" =~ ^[A-Za-z0-9_.-]+$ ]]; then
      echo "CLUSTER_NAME and NODE_NAME must contain only letters, digits, dots, underscores, or hyphens" >&2
      exit 2
    fi
    if [[ ! "$LOKI_PUSH_URL" =~ ^https?://[A-Za-z0-9_.:-]+/[^\"[:space:]]*$ ]]; then
      echo "LOKI_PUSH_URL must be an HTTP(S) URL without spaces or quotes" >&2
      exit 2
    fi
    IFS=',' read -r -a log_roots <<< "$OBSERVABILITY_LOG_ROOTS"
    cat > "$output_dir/alloy.alloy" <<EOF
logging {
  level = "info"
}

loki.source.file "workloads" {
  targets = [
EOF
    seen_workloads=""
    for entry in "${log_roots[@]}"; do
      workload="${entry%%=*}"
      root="${entry#*=}"
      if [[ "$workload" == "$entry" || ! "$workload" =~ ^[A-Za-z0-9_.-]+$ || "$root" != /* || ! -d "$root" ]]; then
        echo "OBSERVABILITY_LOG_ROOTS entries must be workload=/existing/absolute/local/path" >&2
        exit 2
      fi
      if [[ " $seen_workloads " == *" $workload "* ]]; then
        echo "OBSERVABILITY_LOG_ROOTS workload names must be unique: $workload" >&2
        exit 2
      fi
      fs_type="$(findmnt -T "$root" -n -o FSTYPE)"
      if [[ "$fs_type" == nfs || "$fs_type" == nfs4 ]]; then
        echo "log root must be node-local, not $fs_type: $root" >&2
        exit 2
      fi
      seen_workloads+=" $workload"
      escaped_root="${root//\\/\\\\}"
      escaped_root="${escaped_root//\"/\\\"}"
      cat >> "$output_dir/alloy.alloy" <<EOF
    { __path__ = "$escaped_root/*/logs/**/*.log", cluster = "$cluster_name", node = "$node_name", workload = "$workload" },
EOF
    done
    cat >> "$output_dir/alloy.alloy" <<EOF
  ]
  forward_to = [loki.relabel.run_path.receiver]
  file_match {
    enabled = true
    ignore_older_than = "${ALLOY_IGNORE_OLDER_THAN:-24h}"
    sync_period = "2s"
  }
}

loki.relabel "run_path" {
  forward_to = [loki.process.pack_metadata.receiver]
  rule {
    source_labels = ["filename"]
    regex = "^.*/([^/]+)/logs/(.*)$"
    target_label = "run_id"
    replacement = "\$1"
  }
  rule {
    source_labels = ["filename"]
    regex = "^.*/([^/]+)/logs/(.*)$"
    target_label = "log_file"
    replacement = "\$2"
  }
}

loki.process "pack_metadata" {
  forward_to = [loki.write.controller.receiver]
  stage.pack {
    labels = ["filename", "run_id", "log_file"]
  }
}

loki.write "controller" {
  endpoint {
    url = "$LOKI_PUSH_URL"
  }
}
EOF
    alloy="${ALLOY:-$tools_dir/alloy-linux-$release_arch}"
    [[ -x "$alloy" ]] || { echo "Alloy not found or not executable: $alloy" >&2; exit 1; }
    "$alloy" validate "$output_dir/alloy.alloy"
    if [[ "${NODE_CONFIG_ONLY:-0}" == 1 ]]; then exit 0; fi
  fi
  mkdir -p "$output_dir/textfile"
  "$tools_dir/node_exporter-1.9.1.linux-$release_arch/node_exporter" \
    --web.listen-address="$NODE_ADDR:19100" \
    --collector.textfile.directory="$output_dir/textfile" > "$output_dir/node-exporter.log" 2>&1 &
  pids+=("$!")
  if [[ "${ENABLE_SSD_HEALTH:-0}" == 1 ]]; then
    start_smartctl_exporter
  fi
  "${PYTHON:-python3}" -m profiling_lab.telemetry \
    --output "$output_dir/gpu-$(date -u +%Y%m%dT%H%M%S).jsonl" \
    --textfile-dir "$output_dir/textfile" --duration "${DURATION:-900}" &
  pids+=("$!")
  if [[ -n "${OBSERVATORY_METRICS_DIR:-}" ]]; then
    "${PYTHON:-python3}" -m observatory_metrics.textfile \
      --metrics-dir "$OBSERVATORY_METRICS_DIR" \
      --textfile-dir "$output_dir/textfile" --interval "${OBSERVATORY_METRICS_INTERVAL:-2}" &
    pids+=("$!")
  fi
  if [[ -n "${TOPOLOGY_DIR:-}" ]]; then
    "${PYTHON:-python3}" -m profiling_lab.topology_textfile \
      --topology-dir "$TOPOLOGY_DIR" --textfile-dir "$output_dir/textfile" \
      --interval "${TOPOLOGY_INTERVAL:-10}" &
    pids+=("$!")
  fi
  if [[ -n "${LOKI_PUSH_URL:-}" ]]; then
    mkdir -p "$output_dir/alloy-data"
    "$alloy" run --disable-reporting --storage.path="$output_dir/alloy-data" \
      --server.http.listen-addr=127.0.0.1:12345 "$output_dir/alloy.alloy" \
      > "$output_dir/alloy.log" 2>&1 &
    pids+=("$!")
  fi
elif [[ "$role" == storage ]]; then
  start_smartctl_exporter
elif [[ "$role" == server ]]; then
  cluster_name="${CLUSTER_NAME:-observability-cluster}"
  if [[ ! "$cluster_name" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "CLUSTER_NAME must contain only letters, digits, dots, underscores, or hyphens" >&2
    exit 2
  fi
  if [[ "${DEMO_LIVE:-0}" == 1 ]]; then
    targets=()
  else
    : "${OBSERVABILITY_TARGETS:?Set OBSERVABILITY_TARGETS to comma-separated node=address targets}"
    IFS=',' read -r -a targets <<< "$OBSERVABILITY_TARGETS"
  fi
  mkdir -p "$output_dir/provisioning/datasources" "$output_dir/provisioning/dashboards" "$output_dir/dashboards"
  if [[ "${DEMO_LIVE:-0}" == 1 ]]; then
    demo_addr="${DEMO_ADDR:-127.0.0.1}"
    demo_port="${DEMO_PORT:-19110}"
    demo_topology_dir="${DEMO_TOPOLOGY_DIR:-$PWD/examples/live-demo}"
    "${PYTHON:-python3}" -m profiling_lab.live_demo \
      --listen "$demo_addr:$demo_port" --topology-dir "$demo_topology_dir" \
      --write-prometheus-config "$output_dir/prometheus.yml"
    if [[ "${SERVER_CONFIG_ONLY:-0}" != 1 ]]; then
      "${PYTHON:-python3}" -m profiling_lab.live_demo \
        --listen "$demo_addr:$demo_port" --topology-dir "$demo_topology_dir" &
      pids+=("$!")
    fi
  else
  cat > "$output_dir/prometheus.yml" <<EOF
global:
  scrape_interval: 2s
scrape_configs:
  - job_name: observability
    static_configs:
EOF
  seen_nodes=""
  for target in "${targets[@]}"; do
    node="${target%%=*}"
    address="${target#*=}"
    if [[ "$node" == "$target" || ! "$node" =~ ^[A-Za-z0-9_.-]+$ || ! "$address" =~ ^[A-Za-z0-9_.-]+$ ]]; then
      echo "OBSERVABILITY_TARGETS entries must be node=address with letters, digits, dots, underscores, or hyphens" >&2
      exit 2
    fi
    if [[ " $seen_nodes " == *" $node "* ]]; then
      echo "OBSERVABILITY_TARGETS node names must be unique: $node" >&2
      exit 2
    fi
    seen_nodes+=" $node"
    cat >> "$output_dir/prometheus.yml" <<EOF
      - targets: ['$address:19100']
        labels:
          cluster: $cluster_name
          nodename: $node
EOF
  done
  cat >> "$output_dir/prometheus.yml" <<EOF
    relabel_configs:
      - source_labels: [nodename]
        target_label: instance
EOF
  if [[ -n "${STORAGE_TARGETS:-}" ]]; then
    storage_system="${STORAGE_SYSTEM:-local}"
    if [[ ! "$storage_system" =~ ^[A-Za-z0-9_.-]+$ ]]; then
      echo "STORAGE_SYSTEM must contain only letters, digits, dots, underscores, or hyphens" >&2
      exit 2
    fi
    IFS=',' read -r -a storage_targets <<< "$STORAGE_TARGETS"
    cat >> "$output_dir/prometheus.yml" <<EOF
  - job_name: storage-smart
    scrape_interval: 60s
    static_configs:
EOF
    seen_storage_nodes=""
    for target in "${storage_targets[@]}"; do
      node="${target%%=*}"
      address="${target#*=}"
      if [[ "$node" == "$target" || ! "$node" =~ ^[A-Za-z0-9_.-]+$ || ! "$address" =~ ^[A-Za-z0-9_.-]+$ ]]; then
        echo "STORAGE_TARGETS entries must be node=address with letters, digits, dots, underscores, or hyphens" >&2
        exit 2
      fi
      if [[ " $seen_storage_nodes " == *" $node "* ]]; then
        echo "STORAGE_TARGETS node names must be unique: $node" >&2
        exit 2
      fi
      seen_storage_nodes+=" $node"
      cat >> "$output_dir/prometheus.yml" <<EOF
      - targets: ['$address:${SMARTCTL_PORT:-19633}']
        labels:
          cluster: $cluster_name
          nodename: $node
          storage_system: $storage_system
EOF
    done
    cat >> "$output_dir/prometheus.yml" <<EOF
    relabel_configs:
      - source_labels: [nodename]
        target_label: instance
EOF
  fi
  fi
  cat > "$output_dir/provisioning/datasources/default.yaml" <<EOF
apiVersion: 1
datasources:
  - name: Prometheus
    uid: observability-prometheus
    type: prometheus
    access: proxy
    url: http://127.0.0.1:19090
    isDefault: true
EOF
  if [[ "${ENABLE_LOGS:-0}" == 1 ]]; then
    loki_listen_addr="${LOKI_LISTEN_ADDR:-127.0.0.1}"
    if [[ ! "$loki_listen_addr" =~ ^[A-Za-z0-9_.:-]+$ ]]; then
      echo "LOKI_LISTEN_ADDR must be an IP address or hostname" >&2
      exit 2
    fi
    cat >> "$output_dir/provisioning/datasources/default.yaml" <<EOF
  - name: Loki
    uid: observability-loki
    type: loki
    access: proxy
    url: http://$loki_listen_addr:13100
EOF
    cat > "$output_dir/loki.yaml" <<EOF
auth_enabled: false
server:
  http_listen_address: $loki_listen_addr
  http_listen_port: 13100
  grpc_listen_address: 127.0.0.1
  grpc_listen_port: 0
common:
  instance_addr: 127.0.0.1
  path_prefix: $output_dir/loki-data
  storage:
    filesystem:
      chunks_directory: $output_dir/loki-data/chunks
      rules_directory: $output_dir/loki-data/rules
  replication_factor: 1
  ring:
    kvstore:
      store: inmemory
schema_config:
  configs:
    - from: 2020-10-24
      store: tsdb
      object_store: filesystem
      schema: v13
      index:
        prefix: index_
        period: 24h
compactor:
  working_directory: $output_dir/loki-data/compactor
  retention_enabled: true
  delete_request_store: filesystem
limits_config:
  retention_period: ${LOKI_RETENTION:-168h}
analytics:
  reporting_enabled: false
EOF
    loki="${LOKI:-$tools_dir/loki-linux-$release_arch}"
    [[ -x "$loki" ]] || { echo "Loki not found or not executable: $loki" >&2; exit 1; }
    "$loki" -config.file="$output_dir/loki.yaml" -verify-config=true
  fi
  cat > "$output_dir/provisioning/dashboards/default.yaml" <<EOF
apiVersion: 1
providers:
  - name: Observability
    type: file
    options:
      path: $output_dir/dashboards
EOF
  cp examples/observability/{run-overview,compute-communication,data-storage}.json "$output_dir/dashboards/"
  if [[ "${ENABLE_LOGS:-0}" == 1 ]]; then
    cp examples/observability/run-logs.json "$output_dir/dashboards/"
  fi
  if [[ "${SERVER_CONFIG_ONLY:-0}" == 1 ]]; then exit 0; fi
  "$tools_dir/prometheus-3.5.0.linux-$release_arch/prometheus" \
    --config.file="$output_dir/prometheus.yml" --storage.tsdb.path="$output_dir/prometheus-data" \
    --storage.tsdb.retention.time=1d --web.listen-address=127.0.0.1:19090 > "$output_dir/prometheus.log" 2>&1 &
  pids+=("$!")
  if [[ "${ENABLE_LOGS:-0}" == 1 ]]; then
    "$loki" -config.file="$output_dir/loki.yaml" > "$output_dir/loki.log" 2>&1 &
    pids+=("$!")
  fi
  export GF_AUTH_ANONYMOUS_ENABLED=true GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer
  if [[ -n "${GRAFANA_ADMIN_PASSWORD:-}" ]]; then
    export GF_SECURITY_ADMIN_PASSWORD="$GRAFANA_ADMIN_PASSWORD"
  fi
  export GF_SERVER_HTTP_ADDR=127.0.0.1 GF_SERVER_HTTP_PORT=13000
  export GF_PATHS_DATA="$output_dir/grafana-data" GF_PATHS_LOGS="$output_dir/grafana-logs"
  export GF_PATHS_PROVISIONING="$output_dir/provisioning"
  "$tools_dir/grafana-v12.1.0/bin/grafana" server \
    --homepath="$tools_dir/grafana-v12.1.0" > "$output_dir/grafana.log" 2>&1 &
  pids+=("$!")
else
  echo 'Use node, storage, or server' >&2
  exit 2
fi
# Exit and clean up siblings when one service exits; external timeout bounds the lab.
wait -n "${pids[@]}"
