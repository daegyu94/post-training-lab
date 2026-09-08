#!/usr/bin/env bash
# Export userspace library/build settings for the native Spark ARM64 stack.
# Source this file after selecting the interpreter for the current node.
set -euo pipefail

: "${PYTHON:?Set PYTHON to the selected ARM64 virtual-environment interpreter}"

PYTHON_SITE="$("${PYTHON}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
PYTHON_EXT_SUFFIX="$(
  "${PYTHON}" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX") or "")'
)"

for library_dir in "${PYTHON_SITE}/nvidia/nccl/lib" "${PYTHON_SITE}/nvidia/cudnn/lib"; do
  if [[ -d "${library_dir}" ]]; then
    export LD_LIBRARY_PATH="${library_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
done

# Set PYTHON_HEADERS to an extracted userspace include directory when a native
# helper or Transformer Engine wheel must be built. Never install system
# packages from this script.
if [[ -n "${PYTHON_HEADERS:-}" ]]; then
  export CPATH="${PYTHON_HEADERS}${CPATH:+:${CPATH}}"
fi

# Megatron helper wheels need the exact CPython extension suffix. Preserve an
# explicitly supplied LIBEXT, otherwise pass the current interpreter suffix to
# make through MAKEFLAGS.
export LIBEXT="${LIBEXT:-${PYTHON_EXT_SUFFIX}}"
export MAKEFLAGS="${MAKEFLAGS:-} LIBEXT=${LIBEXT}"

# The optional NCCL EP extension is unavailable on the verified Torch tree.
# This is a build-time switch; changing it after installing a wheel does not
# add the compiled extension to that wheel.
export NVTE_WITH_NCCL_EP="${NVTE_WITH_NCCL_EP:-0}"

echo "[spark-runtime] python=${PYTHON}"
echo "[spark-runtime] python_site=${PYTHON_SITE}"
echo "[spark-runtime] libext=${LIBEXT} nvte_nccl_ep=${NVTE_WITH_NCCL_EP}"
