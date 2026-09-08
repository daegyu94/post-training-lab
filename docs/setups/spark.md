# Spark Setup Contract

Create `setups/spark/local.json` from `local.example.json` and keep the local file ignored by Git.
The example intentionally uses placeholder paths; configure actual node paths only in the local file.

## Required fields

The setup file describes node locations and runtime settings.
Training options, such as batch size and model revision, belong in the experiment file.

| Level | Fields | Purpose |
| --- | --- | --- |
| Setup | `setup: "spark"`, `master_addr`, `master_port`, `nodes` | Name the setup and connect one or two nodes |
| Node | `host`, `checkout`, `python.trl`, `python.megatron` | Locate the checkout and backend environments |
| Node | `model_dirs`, `data_dir`, `output_root` | Locate model snapshots, prepared data and outputs |
| Setup or node | Optional `env` | Supply hardware and runtime variables |

The node checkout should contain the unified repository, and the runner executes from its `backends/<backend>` directory so existing relative launcher paths keep their meaning.

## Environment ownership

Setup environment is restricted to hardware and runtime variables: `NCCL*`, `OMP*`, `HF_*`, `TOKENIZERS_*`, `PYTHON_HEADERS` and `CPATH`.
Training settings belong in the experiment file.
The runner owns `NODE_RANK`, `PYTHON`, `NNODES`, `NPROC_PER_NODE`, rendezvous variables, `MODEL_DIR`, `DATA_DIR` and `OUTPUT_DIR`; experiment files cannot override them.

## Paths and launch checks

Use a node-local model snapshot on every participating node and a shared NFS directory for prepared data and output when a multi-node checkpoint is saved or reloaded.
The runner maps the experiment's `MODEL_ID` to each node's `model_dirs` entry and uses the node's `data_dir` and `output_root` for that rank; the example uses the same placeholders to represent shared NFS paths.

Create each configured `output_root` before running an experiment.
Both nodes must use a clean checkout at the controller commit.
NFS can delay peer session visibility; the runner allows up to 70 seconds, bounded by the run timeout, before rejecting an unowned output directory.
