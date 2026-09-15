#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
: "${PYTHON:?Set PYTHON to the node-local CUDA Python executable}"
: "${NODE_RANK:?Set NODE_RANK to this node's distributed rank}"
: "${MASTER_ADDR:?Set MASTER_ADDR to rank 0's data-interface address}"
: "${PROFILE_RUN_ID:?Set the same unique PROFILE_RUN_ID on both nodes}"
mode="${1:-baseline}"
case "$mode" in baseline|capture|collective) ;; *) echo 'Use baseline, capture, or collective' >&2; exit 2;; esac
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
output_dir="${OUTPUT_DIR:-artifacts/observability/$PROFILE_RUN_ID/$mode}"
mkdir -p "$output_dir"
"$PYTHON" - <<'PY'
import os
import subprocess
from pathlib import Path

memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
available_gib = int(memory['MemAvailable'].split()[0]) / 1024**2
minimum_gib = float(os.environ.get('MIN_AVAILABLE_GIB', '8'))
if available_gib < minimum_gib:
    raise SystemExit(f'Insufficient shared memory: MemAvailable={available_gib:.2f} GiB; require {minimum_gib} GiB for this small lab. Check free -h and swap; do not stop unrelated jobs.')
processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
if processes:
    raise SystemExit(f'GPU compute processes already active: {processes}. Run this baseline after they finish.')
PY
args=(examples/pytorch/ddp_profile_demo.py --run-id "$PROFILE_RUN_ID" --steps "${STEPS:-24}"
      --output-dir "$output_dir" --trace-dir "$output_dir/traces" --profile-ranks "")
if [[ "$mode" == capture ]]; then
  args[${#args[@]}-1]="0,1"
elif [[ "$mode" == collective ]]; then
  args=(examples/pytorch/collective_baseline.py --output-dir "$output_dir")
fi
{
  date -u +%FT%TZ
  git -c safe.directory="$PWD" rev-parse HEAD
  git -c safe.directory="$PWD" diff --stat
  hostname
  "$PYTHON" -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.nccl.version())'
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
  free -h
  nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv
  ip -br -4 addr
  timedatectl show -p NTPSynchronized || true
  env | sort | sed -n '/^NCCL_/p; /^OMP_NUM_THREADS=/p'
} > "$output_dir/node-$NODE_RANK-manifest.txt"
timeout --signal=TERM "${RUN_TIMEOUT:-300}" "$PYTHON" -m torch.distributed.run \
  --nnodes="${NNODES:-2}" --nproc-per-node=1 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port="${MASTER_PORT:-29670}" \
  "${args[@]}" 2>&1 | tee "$output_dir/node-$NODE_RANK.log"
