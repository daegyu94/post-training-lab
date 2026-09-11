# Spark Cluster Setup

실행 환경은 controller 한 대와 NVIDIA DGX Spark 노드 `spark1`, `spark2`로 구성됩니다.
Controller는 Git checkout·설정·실행 기록을 관리하고 GPU 학습 process를 실행하지 않습니다.
각 Spark 노드는 자체 CUDA Python 환경, node-local 모델 snapshot과 node-local 학습 데이터를 가지며 실제 rank process를 실행합니다.

```text
                           SSH control plane
 +--------------------+  ssh spark@spark1  +--------------------+
 | controller         | -----------------> | spark1             |
 |                    |  ssh spark@spark2  | rank 0 / GPU work  |
 | experiments/run.py | -----------------> +--------------------+
 | controller logs    |                              |
 +--------------------+                              | rendezvous + NCCL
          |                                          | (MASTER_ADDR and
          | shared NFS: /path/to/shared              |  configured interface)
          |                                          v
          |                              +--------------------+
          +----------------------------> | spark2             |
             same shared files           | rank 1 / GPU work  |
                                         +--------------------+
```

`experiments/run.py --execute`는 controller에서 각 노드로 SSH 접속해 launcher를 시작합니다.
학습 process는 `MASTER_ADDR`·`MASTER_PORT`로 rendezvous한 뒤 NCCL이 setup에 지정된 interface를 사용합니다.
물리 NIC, IP 대역, RoCE·InfiniBand 구성은 환경마다 다르므로 `local.json`의 host, rendezvous 주소와 `NCCL_*` 변수를 실제 구성에 맞춥니다.

이 PoC에는 별도 storage cluster가 없어 **노드가 학습 중 직접 읽는 무거운 자원(모델·데이터)은 NFS를 거치지 않고 각 노드에 둡니다** — 이유는 [30B NVMe 실습](../../labs/nvme-30b/README.md#training-data-storage)을 따릅니다.

같은 공유 파일도 controller와 Spark 노드에서 다른 절대 경로로 보일 수 있으므로, 설정의 `checkout`·`model_dirs`·`data_dir`·`output_root`에는 **명령을 실행하는 노드에서 보이는** 절대 경로를 적습니다.
`output_root`는 모든 backend가 같은 경로를 쓸 때 문자열로, 서로 다른 저장장치를 쓸 때는 `trl`·`megatron` 키를 가진 객체로 지정합니다.

```json
"output_root": {
  "trl": "/path/to/trl-run-results",
  "megatron": "/path/to/megatron-run-results"
}
```

<a id="prepare-each-spark-node"></a>

## Prepare Each Spark Node

Git checkout은 따로 clone하지 않고 controller의 NFS 공유 디렉터리를 그대로 씁니다.
두 노드와 controller가 같은 `.git`을 보므로 별도 pull·sync 없이 controller에서 커밋한 상태가 즉시 반영됩니다.

**가상환경과 cache는 반드시 checkout 밖의 node-local 경로에 둡니다.**
checkout 안에 두면 컴파일된 native extension이 NFS를 통해 다른 노드와 공유되어 버립니다.

| 용도 | 권장 경로 |
| --- | --- |
| Git checkout | NFS 공유 디렉터리의 저장소 경로 |
| TRL 가상환경 | `$HOME/.local/ptl/venvs/trl` |
| Megatron 가상환경 | `$HOME/.local/ptl/venvs/megatron` |
| Hugging Face cache | `$HOME/.local/ptl/cache/huggingface` |
| Torch extension cache | `$HOME/.local/ptl/cache/torch_extensions` |

TRL과 Megatron의 고정 의존성이 다르므로 하나의 공용 가상환경을 공유하지 않습니다.
학습 결과와 NVMe offload 파일은 용량과 수명이 다르므로 이 디렉터리가 아니라 setup의 `output_root`에 둡니다.

각 노드에서 다음을 한 번 실행합니다.
이 스크립트는 compiler, Python headers, `libaio-dev`, `ninja-build` 등 native extension과 DeepSpeed async I/O에 필요한 system package를 설치하고, checkout 밖에 두 가상환경을 만들며, `spark` 사용자의 memlock 한도를 32GiB로 설정합니다.

```bash
cd "/path/to/shared/post-training-lab"
./setups/spark/prepare_node.sh
```

`sudo` 권한이 필요하며 완료 후 새 SSH session으로 다시 접속해야 memlock 설정이 적용됩니다.

```bash
test "$(ulimit -l)" -ge 33554432
```

**스크립트는 backend Python package를 설치하지 않습니다.**
TRL은 [TRL Spark environment](../../docs/backends/trl.md#prepare-the-spark-environment), Megatron은 [Megatron Spark environment](../../docs/backends/megatron.md#spark-environment)를 따릅니다.
