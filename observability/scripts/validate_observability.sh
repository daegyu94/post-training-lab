#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON:-python}"
compose_dir="$repo_root/examples/observability"
target_dir="${TARGET_DIR:-$compose_dir/targets}"
output_dir="${OUTPUT_DIR:-$repo_root/artifacts/observability-validation}"
prometheus_url="${PROMETHEUS_URL:-http://127.0.0.1:9090}"
grafana_url="${GRAFANA_URL:-http://127.0.0.1:3000}"
validation_timeout="${VALIDATION_TIMEOUT:-60}"
require_targets_up="${REQUIRE_TARGETS_UP:-0}"

if ! command -v "$python_bin" >/dev/null 2>&1; then
  printf 'Python executable not found: %s\n' "$python_bin" >&2
  exit 1
fi

"$python_bin" -m profiling_lab.observability check-targets \
  --target-dir "$target_dir"

if ! command -v docker >/dev/null 2>&1; then
  printf 'Docker is required to validate the monitoring stack.\n' >&2
  exit 1
fi
if ! docker compose version >/dev/null 2>&1; then
  printf 'Docker Compose v2 is required.\n' >&2
  exit 1
fi
if [[ "$require_targets_up" != "0" && "$require_targets_up" != "1" ]]; then
  printf 'REQUIRE_TARGETS_UP must be 0 or 1.\n' >&2
  exit 1
fi

export VALIDATION_DOCKER_VERSION
export VALIDATION_COMPOSE_VERSION
VALIDATION_DOCKER_VERSION="$(docker --version)"
VALIDATION_COMPOSE_VERSION="$(docker compose version)"

(
  cd "$compose_dir"
  docker compose config >/dev/null
  docker compose up -d
)

validation_args=(
  --target-dir "$target_dir"
  --prometheus-url "$prometheus_url"
  --grafana-url "$grafana_url"
  --output "$output_dir/summary.json"
  --timeout "$validation_timeout"
)
if [[ "$require_targets_up" == "1" ]]; then
  validation_args+=(--require-targets-up)
fi

set +e
"$python_bin" -m profiling_lab.observability validate-stack "${validation_args[@]}"
validation_status=$?
set -e

printf 'Monitoring services remain running after validation.\n'
printf 'Stop them with: cd %s && docker compose down\n' "$compose_dir"
exit "$validation_status"
