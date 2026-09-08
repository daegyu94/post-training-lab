# Repeated Megatron measurements

Use `experiments/benchmarks.py` to compare Megatron settings under the same model, data and training conditions.
It runs each comparison through `experiments/run.py` and saves the configuration, logs and measurements.
The default is a dry run: remote training starts only with `--execute`.

## Choose the measurement budget

The default plan is intended for repeated measurements; the shorter configuration checks that the plan can run at a lower cost.
A cell is one comparison condition, such as overlap at micro-batch size 1.
Each cell has variants, such as overlap off and on.

| Setting | Default plan | Short validation plan |
| --- | --- | --- |
| Optimizer steps per measured run | 64 | 16 |
| Measured repeats per variant | 4 | 2 |
| Initial steps excluded within each run | 8 | 4 |
| Checkpoint intervals | 16 and 32 | 4 and 8 |

Cell names retain the default interval values even when CLI options change them.
Use the effective values saved in each run configuration when interpreting results.

## What the plan compares

The eight cells compare overlap at micro-batches 1 and 2, recompute at sequence lengths 2048 and 4096, sequence parallelism at both lengths with TP=2, and sync/async checkpointing at two save intervals.
The plan fixes Qwen 0.5B and `HuggingFaceH4/no_robots` revisions, seed 42, BF16 full fine-tuning, global batch size 4 and the Megatron train stage.

Length comparisons set `PAD_TO_MAX_LENGTH=true` to make the tensors actually reach 2048 or 4096 tokens.
Changing only a truncation limit would not change the tensor width for short examples.
Megatron SFT leaves padding disabled by default; direct runs can opt in with `--pad-to-max-length`.

## Inspect and execute

1. Prepare the Spark setup file as described in the [getting-started guide](../getting-started.md).
   Keep local paths and host names in the gitignored setup file.
   Multi-node checkpoint reload requires the same shared `output_root` on participating nodes.
2. Inspect the expanded plan from the repository root:

```bash
python experiments/benchmarks.py \
  --setup setups/spark/local.json \
  --output results/benchmarks-dry-run
```

3. Run the short validation configuration when the plan and node paths are ready:

```bash
python experiments/benchmarks.py \
  --setup setups/spark/local.json \
  --output results/benchmarks-16x2 \
  --steps 16 --repeats 2 --within-run-warmup 4 \
  --checkpoint-intervals 4 8 --execute
```

Each run gets a unique output name containing the benchmark name, cell, variant and run index.
This prevents a new run from reusing another run's claimed output directory.

## Read the output

The output directory contains the expanded plan, incremental `records.jsonl`, per-run manifests and logs, raw measurement JSONL, checksums and a final manifest.
Use the manifests to connect each measurement to its settings and source commit.
Warmup runs, failed runs and runs after a variant's first failure are excluded from summaries.

A measured run is accepted only if every requested step is present, loss and gradient values are finite, no steps were skipped, and the runner exits successfully.
The summaries describe steady-step timing, whole-run time and peak CUDA allocated/reserved memory.
They do not establish model quality or a general speedup.

Checkpoint timing is reported separately from training-step timing.
Event seconds are summed within each rank, then the maximum rank total is reported for save-call enqueue work and blocking finalization separately.
These timings do not establish that a checkpoint survives a crash.

The [integration measurement record](../verification/integration-20260908/benchmarks/README.md) contains the executed short plan and raw logs.
It also records the selective-recompute configuration failure, so that comparison is not complete.
