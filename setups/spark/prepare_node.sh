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

install -d "$ptl_home/cache" "$ptl_home/venvs"
for backend in trl megatron; do
  python3 -m venv "$ptl_home/venvs/$backend"
done

# repo_root is an NFS-shared checkout (same worktree as controller), so it is
# typically owned by the controller user's UID; Git refuses to operate on a
# repository owned by another UID unless it is marked safe. This edits the
# config file directly rather than invoking `git config`, since Spark nodes
# never run Git commands (see AGENTS.md).
safe_directory_line="	directory = $repo_root"
if ! grep -qxF "$safe_directory_line" "$HOME/.gitconfig" 2>/dev/null; then
  printf '[safe]\n%s\n' "$safe_directory_line" >> "$HOME/.gitconfig"
fi

echo "Prepared $ptl_home"
echo "Reconnect over SSH before checking: ulimit -l"
