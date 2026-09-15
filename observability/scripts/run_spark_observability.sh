#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
tools_dir="${TOOLS_DIR:-$HOME/.local/share/profiling-lab-tools}"
output_dir="${OUTPUT_DIR:-$PWD/artifacts/spark/monitoring-$(hostname)}"
mkdir -p "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"
role="${1:?Use node, storage, or server}"
pids=()
cleanup() {
  trap - EXIT INT TERM
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
  if [[ "$role" == node ]]; then rm -f "$output_dir/textfile/gpu.prom" "$output_dir/textfile/framework.prom"; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
start_smartctl_exporter() {
  : "${NODE_ADDR:?Set NODE_ADDR to this storage node management address}"
  local exporter smartctl_path smartctl_cmd sudo_mode needs_sudo first_device scan_output
  exporter="${SMARTCTL_EXPORTER:-$tools_dir/smartctl_exporter-0.14.0.linux-arm64/smartctl_exporter}"
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
    # Observed directly on a Spark node: smartctl_exporter reports
    # "Permission denied" and exports zero SMART fields as a plain user.
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
  mkdir -p "$output_dir/textfile"
  "$tools_dir/node_exporter-1.9.1.linux-arm64/node_exporter" \
    --web.listen-address="$NODE_ADDR:19100" \
    --collector.textfile.directory="$output_dir/textfile" > "$output_dir/node-exporter.log" 2>&1 &
  pids+=("$!")
  if [[ "${ENABLE_SSD_HEALTH:-0}" == 1 ]]; then
    start_smartctl_exporter
  fi
  "${PYTHON:-python3}" -m profiling_lab.spark_telemetry \
    --output "$output_dir/gpu-$(date -u +%Y%m%dT%H%M%S).jsonl" \
    --textfile-dir "$output_dir/textfile" --duration "${DURATION:-900}" &
  pids+=("$!")
  if [[ -n "${FRAMEWORK_METRICS_DIR:-}" ]]; then
    "${PYTHON:-python3}" -m profiling_lab.framework_metrics_textfile \
      --metrics-dir "$FRAMEWORK_METRICS_DIR" \
      --textfile-dir "$output_dir/textfile" --interval "${FRAMEWORK_METRICS_INTERVAL:-2}" &
    pids+=("$!")
  fi
  if [[ -n "${TOPOLOGY_DIR:-}" ]]; then
    "${PYTHON:-python3}" -m profiling_lab.topology_textfile \
      --topology-dir "$TOPOLOGY_DIR" --textfile-dir "$output_dir/textfile" \
      --interval "${TOPOLOGY_INTERVAL:-10}" &
    pids+=("$!")
  fi
elif [[ "$role" == storage ]]; then
  start_smartctl_exporter
elif [[ "$role" == server ]]; then
  cluster_name="${CLUSTER_NAME:-spark-cluster}"
  if [[ ! "$cluster_name" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "CLUSTER_NAME must contain only letters, digits, dots, underscores, or hyphens" >&2
    exit 2
  fi
  if [[ "${DEMO_LIVE:-0}" == 1 ]]; then
    spark_targets=()
  elif [[ -n "${SPARK_TARGETS:-}" ]]; then
    IFS=',' read -r -a spark_targets <<< "$SPARK_TARGETS"
  else
    : "${SPARK1_ADDR:?Set SPARK_TARGETS or SPARK1_ADDR and SPARK2_ADDR}"
    : "${SPARK2_ADDR:?Set SPARK_TARGETS or SPARK1_ADDR and SPARK2_ADDR}"
    spark_targets=("spark1=$SPARK1_ADDR" "spark2=$SPARK2_ADDR")
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
  - job_name: spark
    static_configs:
EOF
  seen_nodes=""
  for target in "${spark_targets[@]}"; do
    node="${target%%=*}"
    address="${target#*=}"
    if [[ "$node" == "$target" || ! "$node" =~ ^[A-Za-z0-9_.-]+$ || ! "$address" =~ ^[A-Za-z0-9_.-]+$ ]]; then
      echo "SPARK_TARGETS entries must be node=address with letters, digits, dots, underscores, or hyphens" >&2
      exit 2
    fi
    if [[ " $seen_nodes " == *" $node "* ]]; then
      echo "SPARK_TARGETS node names must be unique: $node" >&2
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
    uid: spark-prometheus
    type: prometheus
    access: proxy
    url: http://127.0.0.1:19090
    isDefault: true
EOF
  cat > "$output_dir/provisioning/dashboards/default.yaml" <<EOF
apiVersion: 1
providers:
  - name: Spark Lab
    type: file
    options:
      path: $output_dir/dashboards
EOF
  cp examples/observability/{spark-resources,compute-communication,data-storage}.json "$output_dir/dashboards/"
  if [[ "${SERVER_CONFIG_ONLY:-0}" == 1 ]]; then exit 0; fi
  "$tools_dir/prometheus-3.5.0.linux-arm64/prometheus" \
    --config.file="$output_dir/prometheus.yml" --storage.tsdb.path="$output_dir/prometheus-data" \
    --storage.tsdb.retention.time=1d --web.listen-address=127.0.0.1:19090 > "$output_dir/prometheus.log" 2>&1 &
  pids+=("$!")
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
