# Runner Architecture

The runner combines a machine setup file with an experiment file, then invokes the selected backend.
This separation lets a new environment reuse the existing training code.
The diagram shows where validation happens and how work reaches the Spark nodes.

```text
setups/spark/local.json       experiments/<backend>/*.json
          |                              |
          +------------+-----------------+
                       v
              experiments/run.py
                       |
          +------------+-----------------+
          |                              |
     dry-run plan                 one SSH process per rank
                                         |
                              backends/<backend>/scripts/
```

Setup files contain hosts, checkout locations, Python interpreters, model directories, data directories, output roots and hardware environment variables.
Experiment files contain the backend, Spark topology and training environment.
The runner joins these inputs only at execution time, keeping model and learning options out of machine setup.

Before execution, the runner checks whether the requested combination is supported.
It validates the backend and setup, node count, one GPU process per node, fixed revisions, environment variable names and training stages.
For Megatron, it also checks that the parallelism and batch sizes divide correctly.

The controller starts rank processes concurrently with `ssh` in `BatchMode`.
Each remote command changes to `backends/<backend>`, exports the selected node environment and invokes that backend's existing launcher.
Megatron sources `scripts/spark_runtime_env.sh` before launch.

Each run has its own output directory and a file identifying its processes (pidfile).
The launcher is wrapped with `timeout --signal=TERM --kill-after=30s`.
If a rank fails, the controller asks only the remaining run-specific pid groups to terminate, then waits for the SSH processes before writing final statuses.
This cleanup targets only the processes belonging to that run.

The run manifest makes the execution traceable.
`manifest.json` records the backend, run ID, setup and experiment SHA-256 hashes, controller commit, per-rank host, command, log path and exit status.

## Support and Evidence

Use the table to distinguish accepted settings from paths that have actually run on GPUs.
A successful small-model run applies to that recorded configuration, not every model using the same backend.

| Backend / setup | Implementation | Evidence boundary |
| --- | --- | --- |
| TRL / Spark, one or two nodes, DDP | shared launcher with topology settings | 0.5B LoRA two-node layout regression and single-node common runner smoke passed |
| TRL / Spark, FSDP2 or DeepSpeed | base/train stages; tuned/all rejected | [earlier configuration results](backends/trl/training-verification.md), including failures; no blanket runtime support claim |
| Megatron / Spark, TP/PP/EP compatible with world size | base/train/tuned and optional resume workflow | 0.5B full SFT two-node save/reload and 2 → 3 step common runner resume smoke passed |
| Megatron feature variants | explicit experiment candidates | 0.5B overlap, full recompute, sequence parallel and checkpoint repeats recorded; selective rejected by current configuration; resume equivalence and durability remain separate |
| Other setups | no common runner adapter yet | rejected before launch |
| Legacy TRL RTX workflow | source and historical guide retained | outside this integration's Spark revalidation scope |

Actual commits, settings and results are in the [integration verification record](verification/integration-20260908/README.md).
An accepted configuration is not itself a completed GPU experiment.
