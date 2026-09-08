# Getting Started

Start by preparing a Spark setup file, then inspect an experiment before running it.
The controller coordinates the run; the Spark nodes perform the training.
By default, the runner prints a plan and only contacts the nodes when you add `--execute`.

The controller needs Python 3.10+ and OpenSSH.
Use `python3` in the examples if the controller has no `python` command.

## Configure the Spark setup

Copy the setup example to an ignored local file and replace every placeholder with the paths from the target nodes.

```bash
cp setups/spark/local.example.json setups/spark/local.json
```

The setup file describes where each node runs the experiment and finds its inputs.
It lists the checkout, backend-specific Python interpreters, local model snapshots, prepared data and output directories.
Create each node’s `output_root` before launching and install each backend in its own Python environment using its [TRL](backends/trl/spark-cluster.md) or [Megatron](backends/megatron/spark-cluster.md) guide.
Keep the remote checkout clean and at the same commit as the controller.

Model and dataset revisions belong in the experiment file.
Use immutable 40-character hexadecimal commit IDs so another run can load the same versions.

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

Choose the preset that matches the backend and node count you want to check.
These short runs check that the execution path works; they do not measure model quality.

- `experiments/trl/smoke.json` runs the two-node DDP LoRA smoke.
- `experiments/trl/single-node-smoke.json` runs the same TRL code with one node.
- `experiments/megatron/smoke.json` runs the two-node full-parameter Megatron smoke.
- `experiments/megatron/resume-smoke.json` adds checkpoint reload and one resumed optimizer step.

For repeated feature comparisons, use the [measurement guide](experiments/repeated-measurements.md).

The runner does not replace backend launchers or claim GPU verification when a run has only been dry-run or partially completed.
