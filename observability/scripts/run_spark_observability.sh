#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
tools_dir="${TOOLS_DIR:-$HOME/.local/share/profiling-lab-tools}"
output_dir="${OUTPUT_DIR:-$PWD/artifacts/spark/monitoring-$(hostname)}"
mkdir -p "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"
role="${1:?Use node or server}"
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
if [[ "$role" == node ]]; then
  : "${NODE_ADDR:?Set NODE_ADDR to this node management address}"
  mkdir -p "$output_dir/textfile"
  "$tools_dir/node_exporter-1.9.1.linux-arm64/node_exporter" \
    --web.listen-address="$NODE_ADDR:19100" \
    --collector.textfile.directory="$output_dir/textfile" > "$output_dir/node-exporter.log" 2>&1 &
  pids+=("$!")
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
elif [[ "$role" == server ]]; then
  : "${SPARK1_ADDR:?Set SPARK1_ADDR to the spark1 management address}"
  : "${SPARK2_ADDR:?Set SPARK2_ADDR to the spark2 management address}"
  : "${GRAFANA_ADMIN_PASSWORD:?Set a non-default Grafana password}"
  mkdir -p "$output_dir/provisioning/datasources" "$output_dir/provisioning/dashboards" "$output_dir/dashboards"
  cat > "$output_dir/prometheus.yml" <<EOF
global:
  scrape_interval: 2s
scrape_configs:
  - job_name: spark
    static_configs:
      - targets: ['$SPARK1_ADDR:19100', '$SPARK2_ADDR:19100']
        labels:
          cluster: spark-cluster
EOF
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
  cp examples/observability/spark-resources.json "$output_dir/dashboards/"
  "$tools_dir/prometheus-3.5.0.linux-arm64/prometheus" \
    --config.file="$output_dir/prometheus.yml" --storage.tsdb.path="$output_dir/prometheus-data" \
    --storage.tsdb.retention.time=1d --web.listen-address=127.0.0.1:19090 > "$output_dir/prometheus.log" 2>&1 &
  pids+=("$!")
  export GF_SECURITY_ADMIN_PASSWORD="$GRAFANA_ADMIN_PASSWORD"
  export GF_SERVER_HTTP_ADDR=127.0.0.1 GF_SERVER_HTTP_PORT=13000
  export GF_PATHS_DATA="$output_dir/grafana-data" GF_PATHS_LOGS="$output_dir/grafana-logs"
  export GF_PATHS_PROVISIONING="$output_dir/provisioning"
  "$tools_dir/grafana-v12.1.0/bin/grafana" server \
    --homepath="$tools_dir/grafana-v12.1.0" > "$output_dir/grafana.log" 2>&1 &
  pids+=("$!")
else
  echo 'Use node or server' >&2
  exit 2
fi
# Exit and clean up siblings when one service exits; external timeout bounds the lab.
wait -n "${pids[@]}"
