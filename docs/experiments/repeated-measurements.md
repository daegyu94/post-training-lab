# Repeated Megatron measurements

`experiments/benchmarks.py` drives the bounded Megatron feature matrix through `experiments/run.py`.

The driver is dry-run by default and requires an explicit `--execute` before it starts remote training.

The default plan uses 64 training steps, four measured repeats, eight within-run warmup steps, and checkpoint intervals 16 and 32.
The interval values in cell names identify the default plan; CLI overrides change the effective values recorded in each run configuration.

The short validation configuration is `--steps 16 --repeats 2 --within-run-warmup 4 --checkpoint-intervals 4 8`.

The eight factorial cells cover overlap at micro-batches 1 and 2, recompute at lengths 2048 and 4096, sequence parallelism at both lengths with TP2, and sync or async checkpointing at intervals 16 and 32.

Warmup records, failed records, and records after a variant's first failure are excluded from summaries.

Each run gets a unique output basename containing the benchmark output name, cell, variant, and run index, so a remote runner claim cannot be silently reused.

The output directory contains the expanded plan, an incremental `records.jsonl`, per-run runner manifests and logs, raw measurement JSONL, checksums, and a final manifest.

Measurement summaries report descriptive steady-step timings, wall time, peak CUDA allocation and reservation, and cumulative checkpoint event time.

Checkpoint event seconds are summed within each rank and the maximum rank total is reported separately for save-call enqueue work and blocking finalization.

Native Megatron iteration logs are accepted only when every requested step is present, finite, non-skipped, and accompanied by a successful runner exit.

The plan compares pinned Qwen 0.5B and `HuggingFaceH4/no_robots` data with seed 42, BF16 full fine-tuning, global batch size 4, and Megatron train stage.

Multi-node checkpoint reload requires the participating nodes to use the same shared `output_root`.

The records provide timing and finite loss or gradient evidence; they do not claim a quality improvement or a speedup.

Example dry run:

```bash
python experiments/benchmarks.py \
  --setup setups/spark/local.json \
  --output results/benchmarks-dry-run
```

Example execution with the short smoke budget:

```bash
python experiments/benchmarks.py \
  --setup setups/spark/local.json \
  --output results/benchmarks-16x2 \
  --steps 16 --repeats 2 --within-run-warmup 4 \
  --checkpoint-intervals 4 8 --execute
```

The setup file may contain local absolute paths and host names and should remain gitignored.

The recompute and sequence length cells set `PAD_TO_MAX_LENGTH=true` so their 2048 and 4096 settings change tensor width instead of only truncating examples.

Megatron SFT keeps padding disabled by default and accepts `--pad-to-max-length` for explicit opt-in.

The [integration measurement record](../verification/integration-20260908/benchmarks/README.md) contains the executed short plan, raw logs, and the selective-recompute configuration failure.
