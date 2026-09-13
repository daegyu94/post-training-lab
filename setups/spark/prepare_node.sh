#!/usr/bin/env bash
set -euo pipefail

ptl_home="${PTL_HOME:-$HOME/.local/ptl}"

sudo apt-get update
sudo apt-get install -y \
  build-essential cmake coreutils libaio-dev ninja-build pkg-config \
  python3 python3-dev python3-venv util-linux

printf 'spark soft memlock 33554432\nspark hard memlock 33554432\n' \
  | sudo tee /etc/security/limits.d/90-post-training-lab.conf >/dev/null

if getent group docker >/dev/null && ! id -nG "$USER" | grep -qw docker; then
  sudo usermod -aG docker "$USER"
  echo "Added $USER to the docker group; reconnect over SSH for it to take effect"
fi

install -d "$ptl_home/cache" "$ptl_home/venvs"
for backend in trl megatron; do
  python3 -m venv "$ptl_home/venvs/$backend"
done

echo "Prepared $ptl_home"
echo "Reconnect over SSH before checking: ulimit -l"
