# Getting Started

먼저 controller에서 GPU 없이 계획을 확인하고, Spark 노드에 환경과 입력을 준비한 뒤 작은 TRL SFT를 실행합니다.
최종 목표는 `base`, `train`, `tuned` 단계의 완료와 결과 판정입니다.
실제 학습은 controller가 아니라 설정한 Spark 노드에서 수행합니다.

## Check a Plan First

Controller에는 Python 3.10 이상과 Git이 필요합니다.
아래 명령은 저장소 루트에서 실행하며 Python 표준 라이브러리만 사용합니다.
예제 setup의 경로는 자리표시자이지만 dry-run은 SSH나 원격 파일 검사를 하지 않으므로 계획 확인에 사용할 수 있습니다.

```bash
python experiments/run.py \
  --backend trl \
  --setup setups/spark/local.example.json \
  --experiment experiments/trl/smoke.json \
  --output artifacts/runs/trl-smoke
```

종료 코드 0과 JSON 계획이 예상 결과입니다.
`backend`가 `trl`이고 두 rank의 환경변수와 출력 경로가 있는지 확인합니다.
명령은 출력 디렉터리를 만들지 않지만 이미 존재하는 경로를 지정하면 거부합니다.
Dry-run 성공은 모델·데이터·GPU·SSH가 준비됐다는 뜻이 아닙니다.
GPU 없는 변환도 확인하려면 [가상 서비스 기록 예제](datasets.md#reviewed-service-traces)를 실행합니다.

## Prepare the Nodes

공통 runner는 한 개 또는 두 개 노드와 노드당 process 하나를 지원합니다.
각 참여 노드에 다음을 준비합니다.

- Controller와 같은 commit의 깨끗한 Git checkout
- ARM64·CUDA 환경에 맞는 백엔드별 Python
- 동일 revision의 완전한 node-local 모델 snapshot
- 같은 JSONL과 manifest가 보이는 데이터 디렉터리
- 미리 생성한 쓰기 가능한 출력 부모 디렉터리

설치는 [TRL](backends/trl.md#spark-environment) 또는 [Megatron](backends/megatron.md#spark-environment)을 따릅니다.
두 requirements는 서로 다른 환경을 대상으로 하며 일반 `scripts/setup.sh`가 Spark 설치를 완결한다고 가정하지 않습니다.
특히 Megatron은 새 환경의 전체 의존성 설치가 검증되지 않았다는 제한이 있습니다.

Controller에는 OpenSSH client가 필요하고 노드에는 Git, Bash, GNU `timeout`, `setsid`가 필요합니다.
각 노드에 대한 `ssh -o BatchMode=yes '<ssh-host>' true`가 비대화형으로 성공하는지 확인합니다.
Rendezvous 주소·포트와 NCCL 통신 경로도 노드 사이에서 접근 가능해야 합니다.

## Prepare the Inputs

제공된 smoke 네 개는 `Qwen/Qwen2.5-0.5B-Instruct` revision `7ae557604adf67be50417f59c2c2f167def9a775`와 No Robots revision `e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b`를 사용합니다.
모델은 각 노드에 가중치와 tokenizer를 포함한 동일 snapshot을 준비하고 절대 경로를 setup에 기록합니다.
한 노드의 cache만 채워서는 안 되며 setup 스크립트는 모델 다운로드를 보장하지 않습니다.

각 노드에서 백엔드 환경의 Python을 지정한 뒤 다음 예시로 고정 모델을 내려받을 수 있습니다.
명령은 Hugging Face Hub에 접속하고 node-local cache에 가중치·tokenizer를 저장합니다.
출력된 snapshot 절대 경로를 해당 노드의 `model_dirs`에 사용합니다.

```bash
export PYTHON='<backend-python>'
"$PYTHON" - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download(
    repo_id="Qwen/Qwen2.5-0.5B-Instruct",
    revision="7ae557604adf67be50417f59c2c2f167def9a775",
))
PY
```

[공개 데이터 준비](datasets.md#public-data)로 No Robots 8 train / 2 validation 예제를 생성합니다.
JSONL과 manifest를 모든 참여 노드에서 읽을 수 있게 하고 `data_dir`은 해당 노드 기준 경로로 지정합니다.

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
| `nodes[].data_dir` | JSONL과 manifest 디렉터리 |
| `nodes[].output_root` | 미리 생성한 출력 부모 디렉터리 |
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
  --output artifacts/runs/trl-smoke \
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
