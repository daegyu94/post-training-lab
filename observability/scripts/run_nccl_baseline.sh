#!/usr/bin/env bash
set -euo pipefail

: "${NCCL_TEST_BINARY:?Set NCCL_TEST_BINARY to all_reduce_perf}"
: "${HOSTS:?Set HOSTS to a comma-separated allocated host list}"
: "${GPUS_PER_NODE:?Set GPUS_PER_NODE to the participating GPU count per node}"

output_dir="${OUTPUT_DIR:-artifacts/nccl-baseline}"
min_bytes="${MIN_BYTES:-8}"
max_bytes="${MAX_BYTES:-8G}"
factor="${SIZE_FACTOR:-2}"

IFS=',' read -r -a hosts <<<"$HOSTS"
node_count="${#hosts[@]}"
total_ranks="$((node_count * GPUS_PER_NODE))"
mkdir -p "$output_dir"

{
  printf 'hosts=%s\n' "$HOSTS"
  printf 'node_count=%s\n' "$node_count"
  printf 'gpus_per_node=%s\n' "$GPUS_PER_NODE"
  printf 'total_ranks=%s\n' "$total_ranks"
  printf 'binary=%s\n' "$NCCL_TEST_BINARY"
  printf 'min_bytes=%s\n' "$min_bytes"
  printf 'max_bytes=%s\n' "$max_bytes"
  printf 'size_factor=%s\n' "$factor"
} >"$output_dir/manifest.env"

mpirun \
  -np "$total_ranks" \
  -N "$GPUS_PER_NODE" \
  --host "$HOSTS" \
  "$NCCL_TEST_BINARY" \
  -b "$min_bytes" \
  -e "$max_bytes" \
  -f "$factor" \
  -g 1 \
  2>&1 | tee "$output_dir/all-reduce.log"
