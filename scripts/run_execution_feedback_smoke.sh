#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-$ROOT_DIR/output/execution-feedback-smoke}"
ENGINE="${ENGINE:-docker}"
PYTHON="${PYTHON:-python}"

mkdir -p "$OUTPUT_DIR"
cd "$ROOT_DIR"

"$PYTHON" -m execution_feedback.prepare --limit-per-split 4 --output-dir "$OUTPUT_DIR/data"

"$PYTHON" - "$OUTPUT_DIR/data/train.jsonl" "$OUTPUT_DIR/train-candidates.jsonl" <<'PY'
import json
from pathlib import Path
import sys

source, destination = map(Path, sys.argv[1:])
with source.open(encoding="utf-8") as rows, destination.open("w", encoding="utf-8") as output:
    for line in rows:
        task = json.loads(line)
        output.write(json.dumps({"task_id": task["task_id"], "candidate_id": "pass", "response": task["reference_solution"]}) + "\n")
        output.write(json.dumps({"task_id": task["task_id"], "candidate_id": "fail", "response": "def broken(*args):\n    return None\n"}) + "\n")
PY

"$PYTHON" -m execution_feedback.evaluate \
  --tasks "$OUTPUT_DIR/data/tasks.jsonl" \
  --candidates "$OUTPUT_DIR/train-candidates.jsonl" \
  --output "$OUTPUT_DIR/train-evaluations.jsonl" \
  --engine "$ENGINE" --workers 2

"$PYTHON" -m execution_feedback.feedback \
  --tasks "$OUTPUT_DIR/data/tasks.jsonl" \
  --evaluations "$OUTPUT_DIR/train-evaluations.jsonl" \
  --output-dir "$OUTPUT_DIR/feedback"

"$PYTHON" - "$OUTPUT_DIR/feedback/manifest.json" <<'PY'
import json
from pathlib import Path
import sys

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if manifest["filtered_sft_count"] != 4 or manifest["dpo_pair_count"] != 4:
    raise SystemExit(f"unexpected smoke result: {manifest}")
PY

echo "smoke artifacts: $OUTPUT_DIR"
