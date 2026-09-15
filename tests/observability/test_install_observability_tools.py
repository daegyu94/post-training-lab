import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize(
    ("machine", "release_arch"),
    [("aarch64", "arm64"), ("x86_64", "amd64")],
)
def test_installer_selects_release_architecture(
    tmp_path: Path, machine: str, release_arch: str
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "uname").write_text(
        f'#!/bin/sh\ncase "$1" in -s) echo Linux ;; -m) echo {machine} ;; esac\n'
    )
    (bin_dir / "curl").write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$CURL_LOG"\n'
        'while test "$#" -gt 0; do\n'
        '  if test "$1" = --output; then shift; : > "$1"; fi\n'
        '  shift\n'
        'done\n'
    )
    (bin_dir / "tar").write_text("#!/bin/sh\nexit 0\n")
    for path in bin_dir.iterdir():
        path.chmod(0o755)

    curl_log = tmp_path / "curl.log"
    result = subprocess.run(
        ["bash", str(ROOT / "observability/scripts/install_observability_tools.sh"), "server"],
        env=os.environ
        | {
            "CURL_LOG": str(curl_log),
            "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
            "TOOLS_DIR": str(tmp_path / "tools"),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    urls = curl_log.read_text()
    assert f"node_exporter-1.9.1.linux-{release_arch}.tar.gz" in urls
    assert f"smartctl_exporter-0.14.0.linux-{release_arch}.tar.gz" in urls
    assert f"prometheus-3.5.0.linux-{release_arch}.tar.gz" in urls
    assert f"grafana-12.1.0.linux-{release_arch}.tar.gz" in urls
