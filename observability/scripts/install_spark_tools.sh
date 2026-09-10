#!/usr/bin/env bash
set -euo pipefail
[[ "$(uname -m)" == aarch64 ]] || { echo 'Run this on a Spark ARM64 node.' >&2; exit 1; }
tools_dir="${TOOLS_DIR:-$HOME/.local/share/profiling-lab-tools}"
mkdir -p "$tools_dir"
cd "$tools_dir"
download() {
  local url="$1" destination="$2"
  curl -fL --retry 3 --retry-all-errors --continue-at - --output "$destination" "$url"
}
download https://github.com/prometheus/node_exporter/releases/download/v1.9.1/node_exporter-1.9.1.linux-arm64.tar.gz node_exporter.tar.gz
tar xzf node_exporter.tar.gz
if [[ "${1:-node}" == server ]]; then
  download https://github.com/prometheus/prometheus/releases/download/v3.5.0/prometheus-3.5.0.linux-arm64.tar.gz prometheus.tar.gz
  tar xzf prometheus.tar.gz
  download https://dl.grafana.com/oss/release/grafana-12.1.0.linux-arm64.tar.gz grafana.tar.gz
  tar xzf grafana.tar.gz
fi
sha256sum ./*.tar.gz > downloaded-archives.sha256
