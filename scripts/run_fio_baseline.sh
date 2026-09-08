#!/usr/bin/env bash
set -euo pipefail

: "${FIO_DIRECTORY:?Set FIO_DIRECTORY to the filesystem under test}"

output_dir="${OUTPUT_DIR:-artifacts/fio-baseline}"
export FIO_RUNTIME="${FIO_RUNTIME:-60}"
export FIO_SIZE="${FIO_SIZE:-16G}"
export FIO_IOENGINE="${FIO_IOENGINE:-libaio}"
mkdir -p "$output_dir" "$FIO_DIRECTORY"

"${FIO_BINARY:-fio}" examples/baselines/checkpoint.fio \
  --output-format=json \
  --output="$output_dir/checkpoint.json"

printf 'fio retained its test file at %s/profiling-lab-checkpoint.bin\n' "$FIO_DIRECTORY"
printf 'Remove that exact file manually after confirming no further read test is needed.\n'
