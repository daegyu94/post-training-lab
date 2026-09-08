# Getting Started

The controller runner validates a bounded experiment and prints its plan by default.
It only contacts Spark nodes when `--execute` is supplied.

## Configure the Spark setup

Copy the ignored local setup and replace every placeholder with the paths from the target nodes.

```bash
cp setups/spark/local.example.json setups/spark/local.json
```

The setup names the two nodes, the unified checkout, the backend specific Python interpreters, node local model snapshots, prepared data and output roots.
Model and dataset revisions live in the experiment file and must be immutable 40-hex revisions.

## Inspect and run an experiment

Run from the repository root.

```bash
python experiments/run.py \
  --backend trl \
  --setup setups/spark/local.json \
  --experiment experiments/trl/smoke.json \
  --output artifacts/runs/trl-smoke
```

This dry-run validates topology, backend settings and path mappings, then prints the controller commit and both configuration hashes.
It does not check remote file availability.
After reviewing the plan, add `--execute` to run the selected backend launcher.
The default remote timeout is 900 seconds; use `--timeout` to select another positive duration.

```bash
python experiments/run.py \
  --backend trl \
  --setup setups/spark/local.json \
  --experiment experiments/trl/smoke.json \
  --output artifacts/runs/trl-smoke \
  --execute
```

The output directory contains `manifest.json` and one `rank-<n>.log` per node.
An existing controller output directory or node output for the same run ID is refused so a completed run cannot be silently overwritten.

## Presets

- `experiments/trl/smoke.json` runs the two-node DDP LoRA smoke.
- `experiments/trl/single-node-smoke.json` runs the same TRL code with one node.
- `experiments/megatron/smoke.json` runs the two-node full-parameter Megatron smoke.

The runner does not replace backend launchers or claim GPU verification when a run has only been dry-run or partially completed.
