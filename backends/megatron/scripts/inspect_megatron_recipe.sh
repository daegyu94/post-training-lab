#!/usr/bin/env bash
set -euo pipefail

.venv/bin/python -m megatron_lab.inspect_recipe "$@"
