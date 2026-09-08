# Spark Setup Contract

Create `setups/spark/local.json` from `local.example.json` and keep the local file ignored by Git.
The example intentionally uses placeholder paths; configure actual node paths only in the local file.

The setup object has `setup: "spark"`, `master_addr`, `master_port`, optional common `env`, and one or two `nodes`.
Each node supplies `host`, `checkout`, `python.trl`, `python.megatron`, `model_dirs`, `data_dir`, `output_root` and optional `env`.
The node checkout should contain the unified repository, and the runner executes from its `backends/<backend>` directory so existing relative launcher paths keep their meaning.

Setup environment is restricted to hardware and runtime variables: `NCCL*`, `OMP*`, `HF_*`, `TOKENIZERS_*`, `PYTHON_HEADERS` and `CPATH`.
Training settings belong in the experiment file.
The runner owns `NODE_RANK`, `PYTHON`, `NNODES`, `NPROC_PER_NODE`, rendezvous variables, `MODEL_DIR`, `DATA_DIR` and `OUTPUT_DIR`; experiment files cannot override them.

Use a node-local model snapshot on every participating node and a shared NFS directory for prepared data and output when a multi-node checkpoint is saved or reloaded.
The runner maps the experiment's `MODEL_ID` to each node's `model_dirs` entry and uses the node's `data_dir` and `output_root` for that rank; the example uses the same placeholders to represent shared NFS paths.
