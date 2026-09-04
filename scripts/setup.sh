#!/usr/bin/env bash
set -euo pipefail

python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

dataset_dir="${DATASET_DIR:-data/ultrachat_200k}"

python - "$dataset_dir" <<'PY'
from pathlib import Path
import sys

from huggingface_hub import snapshot_download

destination = Path(sys.argv[1])
snapshot_download(
    repo_id="HuggingFaceH4/ultrachat_200k",
    repo_type="dataset",
    local_dir=destination,
    allow_patterns=["data/train_sft-*.parquet", "data/test_sft-*.parquet"],
)

data_dir = destination / "data"
for split in ("train_sft", "test_sft"):
    if not list(data_dir.glob(f"{split}-*.parquet")):
        raise SystemExit(f"Missing {split} parquet files under {data_dir}")

print(f"UltraChat dataset is ready at {data_dir}")
PY
