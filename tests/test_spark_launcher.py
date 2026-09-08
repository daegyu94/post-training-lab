import json
import os
import subprocess
from pathlib import Path


def test_launcher_uses_explicit_torchrun_and_forwards_topology(tmp_path: Path) -> None:
    calls = tmp_path / "calls.jsonl"
    fake = tmp_path / "python"
    fake.write_text("#!/usr/bin/env python3\nimport json, os, sys\nopen(%r, 'a').write(json.dumps({'argv': sys.argv[1:], 'rank': os.environ.get('NODE_RANK')})+'\\n')\n" % str(calls), encoding="utf-8")
    fake.chmod(0o755)
    model = tmp_path / "model"
    model.mkdir()
    data = tmp_path / "data"
    data.mkdir()
    env = {**os.environ, "PYTHON": str(fake), "NODE_RANK": "1", "NNODES": "2", "NPROC_PER_NODE": "1", "MASTER_ADDR": "spark1", "MODEL_DIR": str(model), "MODEL_REVISION": "a" * 40, "DATASET_REVISION": "b" * 40, "DATA_DIR": str(data), "OUTPUT_DIR": str(tmp_path / "out"), "STAGE": "base"}
    subprocess.run(["bash", "scripts/run_spark_cluster.sh"], cwd=Path(__file__).parents[1], env=env, check=True, capture_output=True, text=True)
    call = json.loads(calls.read_text().splitlines()[0])
    assert "torch.distributed.run" in call["argv"]
    assert "--standalone" not in call["argv"]
    assert call["rank"] == "1"
    payload = call["argv"][call["argv"].index("sft_lab.spark_train") + 1 :]
    assert payload.count("base") == 1
    assert payload[payload.index("--distributed-backend") + 1] == "ddp"
    assert payload[-2:] == ["--stage", "base"]


def test_launcher_resolves_default_node_local_hub_snapshot(tmp_path: Path) -> None:
    calls = tmp_path / "calls.jsonl"
    revision = "a" * 40
    snapshot = tmp_path / "hub" / "models--Qwen--Qwen3-30B-A3B" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    fake = tmp_path / "python"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "if 'sft_lab.spark_config' in sys.argv:\n"
        "    print(os.environ['EXPECTED_SNAPSHOT'])\n"
        "else:\n"
        "    open(os.environ['CALLS'], 'a').write(json.dumps(sys.argv[1:]) + '\\n')\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    data = tmp_path / "data"
    data.mkdir()
    env = {
        **os.environ,
        "PYTHON": str(fake),
        "NODE_RANK": "0",
        "NNODES": "1",
        "MASTER_ADDR": "127.0.0.1",
        "MODEL_REVISION": revision,
        "DATASET_REVISION": "b" * 40,
        "DATA_DIR": str(data),
        "OUTPUT_DIR": str(tmp_path / "out"),
        "STAGE": "base",
        "EXPECTED_SNAPSHOT": str(snapshot),
        "CALLS": str(calls),
    }
    env.pop("MODEL_DIR", None)
    subprocess.run(
        ["bash", "scripts/run_spark_cluster.sh"],
        cwd=Path(__file__).parents[1],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    call = json.loads(calls.read_text(encoding="utf-8").splitlines()[0])
    payload = call[call.index("sft_lab.spark_train") + 1 :]
    assert payload[payload.index("--model-dir") + 1] == str(snapshot)


def test_launcher_forwards_fsdp2_and_deepspeed_profiles(tmp_path: Path) -> None:
    calls = tmp_path / "calls.jsonl"
    fake = tmp_path / "python"
    fake.write_text("#!/usr/bin/env python3\nimport json, sys\nopen(%r, 'a').write(json.dumps(sys.argv[1:])+'\\n')\n" % str(calls), encoding="utf-8")
    fake.chmod(0o755)
    model = tmp_path / "model"
    data = tmp_path / "data"
    model.mkdir()
    data.mkdir()
    base_env = {
        **os.environ,
        "PYTHON": str(fake),
        "NODE_RANK": "0",
        "NNODES": "2",
        "NPROC_PER_NODE": "1",
        "MASTER_ADDR": "spark1",
        "MODEL_DIR": str(model),
        "MODEL_REVISION": "a" * 40,
        "DATASET_REVISION": "b" * 40,
        "DATA_DIR": str(data),
        "STAGE": "train",
    }
    subprocess.run(
        ["bash", "scripts/run_spark_cluster.sh"],
        cwd=Path(__file__).parents[1],
        env={**base_env, "DISTRIBUTED_BACKEND": "fsdp2", "OUTPUT_DIR": str(tmp_path / "fsdp")},
        check=True,
        capture_output=True,
        text=True,
    )
    profile = tmp_path / "zero3.json"
    profile.write_text("{}", encoding="utf-8")
    subprocess.run(
        ["bash", "scripts/run_spark_cluster.sh"],
        cwd=Path(__file__).parents[1],
        env={
            **base_env,
            "DISTRIBUTED_BACKEND": "deepspeed",
            "DEEPSPEED_CONFIG": str(profile),
            "OUTPUT_DIR": str(tmp_path / "zero3"),
        },
        check=True,
        capture_output=True,
        text=True,
    )
    fsdp_call, deepspeed_call = [json.loads(line) for line in calls.read_text().splitlines()]
    fsdp_payload = fsdp_call[fsdp_call.index("sft_lab.spark_train") + 1 :]
    deepspeed_payload = deepspeed_call[deepspeed_call.index("sft_lab.spark_train") + 1 :]
    assert fsdp_payload[fsdp_payload.index("--distributed-backend") + 1] == "fsdp2"
    assert "--deepspeed-config" not in fsdp_payload
    assert deepspeed_payload[deepspeed_payload.index("--distributed-backend") + 1] == "deepspeed"
    assert deepspeed_payload[deepspeed_payload.index("--deepspeed-config") + 1] == str(profile)


def test_launcher_rejects_unverified_sharded_reload_workflow(tmp_path: Path) -> None:
    model = tmp_path / "model"
    data = tmp_path / "data"
    model.mkdir()
    data.mkdir()
    env = {
        **os.environ,
        "NODE_RANK": "0",
        "MASTER_ADDR": "spark1",
        "MODEL_DIR": str(model),
        "MODEL_REVISION": "a" * 40,
        "DATASET_REVISION": "b" * 40,
        "DATA_DIR": str(data),
        "OUTPUT_DIR": str(tmp_path / "out"),
        "DISTRIBUTED_BACKEND": "fsdp2",
        "STAGE": "all",
    }
    result = subprocess.run(
        ["bash", "scripts/run_spark_cluster.sh"],
        cwd=Path(__file__).parents[1],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "sharded export/reload is not yet verified" in result.stderr
