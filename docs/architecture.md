# Runner Architecture

The repository separates machine setup, experiment intent and backend code.

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

Before execution it checks the backend/setup match, node count, one GPU process per node, immutable revisions, backend environment names, stage support and Megatron parallelism and batch divisibility.

The controller starts rank processes concurrently with `ssh` in `BatchMode`.
Each remote command changes to `backends/<backend>`, exports the selected node environment and invokes that backend's existing launcher.
Megatron sources `scripts/spark_runtime_env.sh` before launch.

Every remote run gets a deterministic run-specific output directory and pidfile.
The launcher is wrapped with `timeout --signal=TERM --kill-after=30s`.
If a rank fails, the controller asks only the remaining run-specific pid groups to terminate, then waits for the SSH processes before writing final statuses.
No broad process kill is used.

`manifest.json` records the backend, run ID, setup and experiment SHA-256 hashes, controller commit, per-rank host, command, log path and exit status.

## Support and Evidence

| Backend / setup | Implementation | Evidence boundary |
| --- | --- | --- |
| TRL / Spark, one or two nodes, DDP | shared launcher with topology settings | 0.5B LoRA two-node layout regression passed; runner-specific results are recorded separately |
| TRL / Spark, FSDP2 or DeepSpeed | base/train stages; tuned/all rejected | [earlier configuration results](backends/trl/training-verification.md), including failures; no blanket runtime support claim |
| Megatron / Spark, TP/PP/EP compatible with world size | base/train/tuned and optional resume workflow | 0.5B full SFT two-node save/reload layout regression passed |
| Megatron feature variants | explicit experiment candidates | repeated performance, resume correctness and durability are separate checks |
| Other setups | no common runner adapter yet | rejected before launch |
| Legacy TRL RTX workflow | source and historical guide retained | outside this integration's Spark revalidation scope |

Actual commits, settings and results are in the [integration verification record](verification/integration-20260908/README.md).
An accepted configuration is not itself a completed GPU experiment.
