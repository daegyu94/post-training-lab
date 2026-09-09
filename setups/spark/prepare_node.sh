#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ptl_home="${PTL_HOME:-$HOME/.local/ptl}"

sudo apt-get update
sudo apt-get install -y \
  build-essential cmake coreutils git libaio-dev ninja-build pkg-config \
  python3 python3-dev python3-venv util-linux

printf 'spark soft memlock 33554432\nspark hard memlock 33554432\n' \
  | sudo tee /etc/security/limits.d/90-post-training-lab.conf >/dev/null

install -d "$ptl_home/cache"
for backend in trl megatron; do
  python3 -m venv "$repo_root/backends/$backend/.venv"
done

echo "Prepared $ptl_home"
echo "Reconnect over SSH before checking: ulimit -l"
