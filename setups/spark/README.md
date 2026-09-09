# NVIDIA DGX Spark Cluster Setup

이 저장소의 실행 환경은 controller 한 대와 NVIDIA DGX Spark 노드 `spark1`, `spark2`로 구성됩니다.
Controller는 Git checkout, 설정, 실행 기록을 관리하고 GPU 학습 process를 실행하지 않습니다.
각 Spark 노드는 자체 CUDA Python 환경과 node-local 모델 snapshot을 가지며 실제 rank process를 실행합니다.

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

`experiments/run.py --execute`는 controller에서 각 노드로 SSH 접속해 해당 노드의 launcher를 시작합니다.
분산 학습 process는 `MASTER_ADDR`와 `MASTER_PORT`로 rendezvous한 뒤 NCCL이 setup에 지정된 통신 interface를 사용합니다.
물리 NIC, IP 대역, RoCE·InfiniBand 구성은 노드 환경마다 다르므로 `local.json`의 host, rendezvous 주소와 `NCCL_*` 변수를 실제 구성에 맞춰 설정합니다.

`/path/to/shared`는 NFS 공유 디렉터리를 나타내는 예시 절대 경로이며 실제 환경의 경로로 바꿉니다.
같은 공유 파일도 controller와 Spark 노드에서 서로 다른 절대 경로로 보일 수 있으므로 각 노드에서 보이는 경로를 사용합니다.
설정 파일의 `checkout`, `model_dirs`, `data_dir`, `output_root`에는 명령을 실행하는 노드에서 보이는 절대 경로를 적습니다.
`output_root`는 모든 backend가 같은 경로를 쓸 때 문자열로 지정하고, 서로 다른 저장장치를 쓸 때는 `trl`과 `megatron` 경로를 가진 객체로 지정합니다.

```json
"output_root": {
  "trl": "/path/to/trl-run-results",
  "megatron": "/path/to/megatron-run-results"
}
```

## Prepare Each Spark Node

Git checkout은 별도로 clone하지 않고 controller의 NFS 공유 디렉터리(`/path/to/shared/post-training-lab`)를 그대로 사용합니다.
두 Spark 노드와 controller가 같은 `.git`을 보므로 별도 pull·sync 없이 controller에서 커밋한 상태가 즉시 반영됩니다.
가상환경과 cache는 checkout 밖의 node-local 경로 `$HOME/.local/ptl` 아래에 모으는 것을 권장합니다.
가상환경을 checkout 안에 두면 컴파일된 native extension이 NFS를 통해 다른 노드와 공유되어 버리므로 반드시 checkout 밖에 둡니다.
학습 결과와 NVMe offload 파일은 용량과 수명이 다르므로 이 디렉터리가 아니라 setup의 `output_root`에 둡니다.

| 용도 | 권장 경로 |
| --- | --- |
| Git checkout | NFS 공유 디렉터리의 저장소 경로 (예: `/path/to/shared/post-training-lab`) |
| TRL 가상환경 | `$HOME/.local/ptl/venvs/trl` |
| Megatron 가상환경 | `$HOME/.local/ptl/venvs/megatron` |
| Hugging Face cache | `$HOME/.local/ptl/cache/huggingface` |
| Torch extension cache | `$HOME/.local/ptl/cache/torch_extensions` |

TRL과 Megatron의 고정 의존성이 다르므로 하나의 공용 가상환경을 공유하지 않습니다.
Launcher는 각 backend의 가상환경을 기본으로 사용하고 cache 환경변수도 위 경로로 설정합니다.

NFS로 공유되는 checkout은 controller 사용자와 다른 UID로 접근하므로 Git이 "dubious ownership"으로 거부할 수 있습니다.
각 Spark 노드의 `spark` 사용자 `~/.gitconfig`에 해당 checkout 경로를 `safe.directory`로 등록해야 하며, 이 등록은 Git 저장소 상태를 바꾸지 않는 순수 설정 파일 편집이므로 `git` 명령을 실행하지 않고 파일에 직접 추가합니다(AGENTS.md의 "Spark 노드에서 Git 명령을 실행하지 않는다" 규칙 유지).

각 노드에서 다음 스크립트를 한 번 실행합니다.
이 스크립트는 compiler, Python headers, `libaio-dev`, `ninja-build` 등 native extension과 DeepSpeed async I/O에 필요한 system package를 설치하고, checkout 밖에 두 가상환경을 만들고, NFS checkout에 대한 `safe.directory` 설정을 추가합니다.
또한 TRL 30B NVMe 실습에 필요한 `spark` 사용자의 memlock 한도를 32GiB로 설정합니다.

```bash
cd "/path/to/shared/post-training-lab"
./setups/spark/prepare_node.sh
```

`sudo` 권한이 필요하며 완료 후 새 SSH session으로 다시 접속해야 memlock 설정이 적용됩니다.

```bash
ulimit -l
test "$(ulimit -l)" -ge 33554432
```

스크립트는 backend Python package를 설치하지 않습니다.
TRL은 [TRL Spark environment](../../docs/backends/trl.md#prepare-the-spark-environment)를 따르고, Megatron은 [Megatron Spark environment](../../docs/backends/megatron.md#spark-environment)의 현재 설치 제한을 확인합니다.
