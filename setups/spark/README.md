# Spark Cluster Setup

Controller는 Git·설정·실행 기록과 Prometheus·Grafana·Loki를 관리하고, NVIDIA DGX Spark `spark1`·`spark2`는 학습 rank와 metric·log collector를 실행합니다.
각 노드는 자체 CUDA Python 환경과 node-local 모델·데이터를 사용합니다.

```text
                           SSH control plane
 +--------------------+  ssh spark@spark1  +--------------------+
 | controller         | -----------------> | spark1             |
 |                    |  ssh spark@spark2  | rank 0 / GPU work  |
 | experiments/run.py | -----------------> +--------------------+
 | Prometheus/Grafana |                              |
 | Loki               |                              |
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

`experiments/run.py --execute`가 SSH로 launcher를 시작하면 rank들은 `MASTER_ADDR`·`MASTER_PORT`로 rendezvous하고 지정 interface에서 NCCL로 통신합니다.
`local.json`의 host·주소·`NCCL_*`는 실제 NIC·RoCE·InfiniBand 구성에 맞춥니다.

별도 storage cluster가 없는 이 PoC에서는 **모델·데이터와 checkpoint를 node-local에 둡니다**.
공유 NFS checkout은 코드 실행에만 사용합니다.
Prometheus와 Grafana의 실행 데이터는 NFS checkout 밖의 controller-local 경로에 둡니다.

## Start Cluster Telemetry

각 Spark node에는 먼저 telemetry node 도구를 `$HOME/telemetry-tools`에 설치합니다.
그 뒤 controller에서 setup의 `host`와 `checkout`을 사용해 node collector와 monitoring server를 함께 실행합니다.

```bash
TOOLS_DIR=$HOME/telemetry-tools \
OUTPUT_DIR=$HOME/telemetry-data \
python scripts/run_telemetry_cluster.py --setup setups/spark/local.json
```

기본 수집 시간은 1시간이며 `--duration`으로 초 단위로 바꿀 수 있습니다.
`Ctrl+C`를 누르거나 process 하나가 종료되면 controller server와 모든 원격 collector를 함께 정리합니다.
새 setup도 `setup`, `nodes[].host`, `nodes[].checkout`을 제공하면 같은 명령을 사용할 수 있습니다.

공유 mount 경로는 host마다 다를 수 있으므로 `checkout`·`model_dirs`·`data_dir`·`output_root`에는 **실행 노드 기준** 절대 경로를 씁니다.
`output_root`는 공통 경로 문자열 또는 backend별 객체로 지정합니다.

```json
"output_root": {
  "trl": "/path/to/trl-run-results",
  "megatron": "/path/to/megatron-run-results"
}
```

<a id="prepare-each-spark-node"></a>

## Prepare Each Spark Node

Controller의 NFS checkout을 공유하므로 노드에서 별도 clone·pull·sync는 하지 않습니다.
**가상환경·cache는 checkout 밖의 node-local 경로에 두어** native extension이 노드 간 공유되지 않게 합니다.

| 용도 | 권장 경로 |
| --- | --- |
| Git checkout | NFS 공유 디렉터리의 저장소 경로 |
| TRL 가상환경 | `$HOME/.local/ptl/venvs/trl` |
| Megatron 가상환경 | `$HOME/.local/ptl/venvs/megatron` |
| Hugging Face cache | `$HOME/.local/ptl/cache/huggingface` |
| Torch extension cache | `$HOME/.local/ptl/cache/torch_extensions` |

TRL과 Megatron의 고정 의존성이 다르므로 하나의 공용 가상환경을 공유하지 않습니다.
학습 결과와 NVMe offload 파일은 용량과 수명이 다르므로 이 디렉터리가 아니라 setup의 `output_root`에 둡니다.

각 노드에서 한 번 실행하면 compiler·Python headers·`libaio-dev`·`ninja-build`, 두 가상환경과 `spark`의 32GiB memlock 한도를 준비합니다.

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
