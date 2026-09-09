# Getting Started

NVIDIA DGX Spark 노드에 환경과 입력을 준비한 뒤 controller에서 작은 TRL 분산 SFT를 시작합니다.
최종 목표는 `base`, `train`, `tuned` 단계의 완료와 결과 판정입니다.
실제 학습은 controller가 아니라 설정한 Spark 노드에서 수행합니다.
Controller와 `spark1`·`spark2`의 역할, SSH 실행 흐름, NFS 경로는 [Spark cluster setup](../setups/spark/README.md)을 확인합니다.

## Check the Execution Plan

Controller에는 Python 3.10 이상과 Git이 필요합니다.
아래 명령은 저장소 루트에서 실행해 setup과 experiment를 결합한 실행 계획을 출력합니다.
예제 setup에는 실제 환경과 무관한 예시 경로가 들어 있습니다.
이 검토 단계는 SSH로 접속하거나 원격 파일의 존재를 확인하지 않습니다.

```bash
python experiments/run.py \
  --backend trl \
  --setup setups/spark/local.example.json \
  --experiment experiments/trl/smoke.json \
  --output results/trl-smoke
```

종료 코드 0과 JSON 계획이 예상 결과입니다.
`backend`가 `trl`이고 두 rank의 환경변수와 출력 경로가 있는지 확인합니다.
명령은 출력 디렉터리를 만들지 않지만 이미 존재하는 경로를 지정하면 거부합니다.
계획 출력은 모델·데이터·GPU·SSH 준비를 증명하지 않습니다.

## Prepare the Nodes

공통 runner는 한 개 또는 두 개 노드와 노드당 process 하나를 지원합니다.
각 참여 노드에 다음을 준비합니다.

- Controller와 같은 commit의 깨끗한 Git checkout
- ARM64·CUDA 환경에 맞는 백엔드별 Python
- 동일 revision의 완전한 node-local 모델 snapshot
- 같은 JSONL과 manifest가 보이는 데이터 디렉터리
- 미리 생성한 쓰기 가능한 출력 부모 디렉터리

먼저 [Spark cluster setup](../setups/spark/README.md#prepare-each-spark-node)의 공통 준비 스크립트로 system package와 backend별 `.venv`를 만듭니다.
그다음 [TRL](backends/trl.md#prepare-the-spark-environment) 또는 [Megatron](backends/megatron.md#spark-environment)의 Python 의존성을 해당 `.venv`에 설치합니다.
두 backend requirements는 서로 다르므로 하나의 가상환경을 공유하지 않습니다.
Megatron은 기존 Spark 환경에서 2노드 smoke를 다시 통과했지만, 새 환경에 전체 의존성을 처음부터 설치하는 과정은 아직 검증되지 않았습니다.

Controller에는 OpenSSH client가 필요하고 노드에는 Git, Bash, GNU `timeout`, `setsid`가 필요합니다.
각 노드에 대한 `ssh -o BatchMode=yes '<ssh-host>' true`가 비대화형으로 성공하는지 확인합니다.
Rendezvous 주소·포트와 NCCL 통신 경로도 노드 사이에서 접근 가능해야 합니다.

## Prepare the Inputs

제공된 네 smoke preset은 실행 경로를 빠르게 확인하기 위해 다음 입력을 고정해서 사용합니다.

| 입력 | 고정 버전 |
| --- | --- |
| 모델 | `Qwen/Qwen2.5-0.5B-Instruct` revision `7ae557604adf67be50417f59c2c2f167def9a775` |
| 데이터 | `HuggingFaceH4/no_robots` revision `e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b` |

먼저 모델을 **각 Spark 노드에 따로** 준비합니다.
모델 snapshot은 가중치, tokenizer와 설정 파일이 모두 들어 있는 디렉터리입니다.
한 노드의 cache만 채우면 다른 노드에서는 모델을 읽을 수 없으며 setup 스크립트가 대신 다운로드하지도 않습니다.

각 노드에서 사용할 backend의 node-local 가상환경을 활성화하고 아래 명령을 실행합니다.
명령이 출력하는 snapshot 절대 경로를 해당 노드의 `model_dirs`에 기록합니다.

```bash
cd "/path/to/shared/post-training-lab/backends/trl"  # 또는 backends/megatron
. "$HOME/.local/ptl/venvs/trl/bin/activate"  # 또는 venvs/megatron
python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download(
    repo_id="Qwen/Qwen2.5-0.5B-Instruct",
    revision="7ae557604adf67be50417f59c2c2f167def9a775",
))
PY
```

다음으로 [공개 데이터 준비](datasets.md#public-data)에 따라 No Robots 학습 예제 8개와 검증 예제 2개를 만듭니다.
모델 snapshot과 마찬가지로 학습 데이터도 각 노드의 `data_dir`을 node-local 경로로 둡니다.
한 노드에서 준비한 뒤 그 결과물(`training.jsonl`, `validation.jsonl`, `manifest.json`)을 다른 노드에 그대로 복사합니다 — 각 노드가 Hugging Face Hub에서 독립적으로 다시 만들면 venv 간 라이브러리 버전 차이로 내용이 미묘하게 갈릴 수 있고, `validate_dataset_manifest()`는 `dataset_id`·`revision`만 비교해 이런 내용물 수준의 차이는 잡지 못합니다.

## Configure the Setup

저장소 루트에서 예제를 복사하고 환경에 맞게 수정합니다.
기존 파일이 있다면 덮어쓰지 않고 내용을 먼저 확인합니다.

```bash
cp -n setups/spark/local.example.json setups/spark/local.json
```

`local.json`은 Git에서 제외됩니다.
예제 host와 `/path/to/...` 값은 실제 실행 전에 모두 교체합니다.

| 필드 | 입력할 값 |
| --- | --- |
| `master_addr`, `master_port` | 첫 참여 노드의 rendezvous 주소·포트 |
| `nodes[].host` | SSH host |
| `nodes[].checkout` | 해당 노드의 저장소 절대 경로 |
| `nodes[].python.trl`, `nodes[].python.megatron` | 백엔드 Python 절대 경로 |
| `nodes[].model_dirs` | 모델 ID → snapshot 절대 경로 |
| `nodes[].data_dir` | JSONL과 manifest 디렉터리 (모델 snapshot처럼 node-local 경로 권장) |
| `nodes[].output_root` | 미리 생성한 출력 부모 디렉터리 또는 `trl`·`megatron`별 부모 디렉터리 |
| `env`, `nodes[].env` | 허용된 NCCL·OMP·HF 등 환경별 변수 |

현재 검증은 TRL만 실행해도 `python` 사전에 두 백엔드 key를 요구합니다.
실제 실행은 선택한 백엔드 Python만 사용합니다.
공유 저장소 경로는 controller가 아니라 각 노드에서 보이는 경로를 적습니다.
분산 checkpoint 재로딩에는 필요한 모든 rank shard가 보여야 합니다.

## Run the Smoke

첫 명령의 setup을 `setups/spark/local.json`으로 바꾸어 계획을 다시 확인합니다.
그다음 controller의 저장소 루트에서 아래 명령을 실행합니다.
이 명령은 SSH로 GPU 작업을 시작하고 원격 결과와 controller 로그를 생성합니다.

```bash
python experiments/run.py \
  --backend trl \
  --setup setups/spark/local.json \
  --experiment experiments/trl/smoke.json \
  --output results/trl-smoke \
  --timeout 900 \
  --execute
```

재실행할 때는 새 `--output`을 지정합니다.
한 노드만 사용하려면 `experiments/trl/single-node-smoke.json`으로 바꿉니다.
다른 preset과 반복 측정은 [Experiments](experiments.md)를 확인합니다.

## Verify the Result

Controller 출력의 `manifest.json`에서 최종 `status`가 `passed`이고 모든 rank가 정상 종료했는지 확인합니다.
같은 디렉터리의 `rank-<n>.log`에는 원격 실행 로그가 남습니다.
백엔드 summary와 가중치는 자동 복사되지 않으며 계획의 원격 출력 경로에서 확인합니다.

TRL은 `summary-base.json`, `summary-train.json`, `summary-tuned.json`과 adapter를 확인합니다.
Finite loss와 실제 optimizer step을 확인하고 `tuned`가 별도 process에서 저장물을 읽었는지 대조합니다.
판정 범위와 과거 결과는 [Verification](verification.md)에 정리합니다.

## Troubleshooting

| 증상 | 확인 | 다음 조치 |
| --- | --- | --- |
| 원격 시작 거부 | checkout commit·dirty 상태 | 사용자 변경을 보존하며 같은 깨끗한 commit 준비 |
| Rendezvous 대기·timeout | 모든 rank 로그, 주소·포트·SSH·NCCL | 원인 해결 후 새 출력으로 재실행 |
| Dataset revision 불일치 | 실험과 manifest의 ID·revision | 입력과 실험을 함께 맞춤 |
| `exceeds max_length` | 해당 예제의 토큰 길이 | 길이·메모리 예산을 검토하고 실험 변경 |
| 출력 claim 실패 | 부모 경로·권한·공유 mount | 부모 경로 준비 후 새 출력 사용 |
| CUDA OOM | rank 로그, 다른 작업, 메모리 계측 | 다른 작업을 임의 종료하지 않고 소형 preset부터 재확인 |

Timeout이나 실패 후 정리 오류가 남으면 manifest와 process 상태를 확인합니다.
Runner의 정리 대상은 해당 run 소유 process이며 다른 작업 종료를 복구 절차로 사용하지 않습니다.
