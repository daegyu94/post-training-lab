# Getting Started

Controller에서 작은 TRL 분산 SFT를 시작하고 `spark1`·`spark2`의 학습 결과를 확인합니다.
순서는 계획 확인 → 노드·입력 준비 → setup 작성 → smoke 실행 → 판정입니다.
역할·SSH·NFS 구성은 [Spark cluster setup](../setups/spark/README.md)을 따릅니다.

## 1. Check the Execution Plan

Controller에는 Python 3.10 이상과 Git이 필요합니다.
아래 명령은 setup과 experiment를 결합한 계획만 출력하며 원격으로 접속하지 않습니다.

```bash
python experiments/run.py \
  --backend trl \
  --setup setups/spark/local.example.json \
  --experiment experiments/trl/smoke.json \
  --output results/trl-smoke
```

Exit 0과 JSON의 `backend: trl`, 두 rank의 환경변수·출력 경로를 확인합니다.
출력 디렉터리를 생성하지는 않지만 기존 경로는 거부하며, 모델·데이터·GPU·SSH 준비는 별도로 확인해야 합니다.

## 2. Prepare the Nodes

공통 runner는 1\~2개 노드와 노드당 process 하나를 지원합니다.
각 참여 노드에 다음이 필요합니다.

- Controller와 같은 commit의 깨끗한 Git checkout (NFS 공유 checkout을 그대로 사용)
- ARM64·CUDA 환경에 맞는 백엔드별 node-local Python 가상환경
- 동일 revision의 완전한 node-local 모델 snapshot
- 같은 JSONL과 manifest가 보이는 node-local 데이터 디렉터리
- 미리 생성한 쓰기 가능한 출력 부모 디렉터리

먼저 각 노드에서 [공통 준비 스크립트](../setups/spark/README.md#prepare-each-spark-node)로 system package와 백엔드별 가상환경을 만듭니다.
그다음 [TRL](backends/trl.md#prepare-the-spark-environment) 또는 [Megatron](backends/megatron.md#spark-environment)의 Python 의존성을 해당 가상환경에 설치합니다.
두 백엔드의 고정 의존성이 다르므로 하나의 가상환경을 공유하지 않습니다.

Controller에는 OpenSSH client가, 노드에는 Bash·GNU `timeout`·`setsid`가 필요합니다.
각 노드에 대해 `ssh -o BatchMode=yes '<ssh-host>' true`가 비대화형으로 성공하는지 확인합니다.
Rendezvous 주소·포트와 NCCL 통신 경로도 노드 사이에서 접근 가능해야 합니다.

## 3. Prepare the Inputs

제공된 smoke preset은 아래 입력을 고정해 실행 경로를 빠르게 확인합니다.

| 입력 | 고정 버전 |
| --- | --- |
| 모델 | `Qwen/Qwen2.5-0.5B-Instruct` revision `7ae557604adf67be50417f59c2c2f167def9a775` |
| 데이터 | `HuggingFaceH4/no_robots` revision `e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b` |

모델은 setup 스크립트가 다운로드하지 않으므로 **각 노드에서** 백엔드 가상환경을 활성화하고 snapshot을 준비합니다.
출력된 절대 경로를 해당 노드의 `model_dirs`에 기록합니다.

```bash
. "$HOME/.local/ptl/venvs/trl/bin/activate"  # 또는 venvs/megatron
python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download(
    repo_id="Qwen/Qwen2.5-0.5B-Instruct",
    revision="7ae557604adf67be50417f59c2c2f167def9a775",
))
PY
```

No Robots 원본을 준비한 뒤 Qwen2.5 tokenizer 기준 2048 token 이하의 학습 4개·검증 2개를 `experiments/prepare_checkpoint_cohort.py`로 각 노드에 선택합니다.
두 backend의 `data_dir`는 이 고정 cohort를 가리켜야 하며, Megatron은 초과 행을 자동으로 자르지 않습니다.

## 4. Configure the Setup

저장소 루트에서 예제를 복사하고 환경에 맞게 수정합니다.
`local.json`은 Git에서 제외되며 예제의 host와 `/path/to/...` 값은 실행 전에 모두 교체합니다.

```bash
cp -n setups/spark/local.example.json setups/spark/local.json
```

| 필드 | 입력할 값 |
| --- | --- |
| `master_addr`, `master_port` | 첫 참여 노드의 rendezvous 주소·포트 |
| `nodes[].host` | SSH host |
| `nodes[].checkout` | 해당 노드에서 보이는 저장소 절대 경로 |
| `nodes[].python.trl`, `nodes[].python.megatron` | 백엔드 Python 절대 경로 |
| `nodes[].model_dirs` | 모델 ID → snapshot 절대 경로 |
| `nodes[].data_dir` | JSONL·manifest 디렉터리 (node-local 권장) |
| `nodes[].output_root` | 미리 생성한 출력 부모 디렉터리, 또는 `trl`·`megatron`별 객체 |
| `env`, `nodes[].env` | 허용된 환경변수 ([Architecture](architecture.md#configuration-contract)) |

> TRL만 실행하더라도 `nodes[].python`에는 `trl`과 `megatron` **두 key가 모두** 있어야 검증을 통과합니다.
> 실제 실행에는 선택한 백엔드의 Python만 사용합니다.

경로는 controller가 아니라 **명령을 실행하는 노드에서 보이는** 절대 경로를 적습니다.
공유 checkout은 코드 배포에만 사용합니다.
Checkpoint는 node-local output에 저장하며 분산 restore·resume은 shared checkpoint storage가 준비될 때까지 TODO입니다.

## 5. Run the Smoke

1단계 명령의 setup을 `setups/spark/local.json`으로 바꿔 계획을 다시 확인한 뒤 `--execute`를 붙입니다.
이 명령은 SSH로 GPU 작업을 시작합니다.

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
한 노드만 쓰려면 `experiments/trl/single-node-smoke.json`으로 바꿉니다.
다른 preset과 반복 측정은 [Experiments](experiments.md)를 따릅니다.

## 6. Verify the Result

Controller 출력의 `manifest.json`에서 최종 `status`가 `passed`이고 모든 rank가 정상 종료했는지 확인합니다.
같은 디렉터리의 `rank-<n>.log`에 원격 실행 로그가 남습니다.
백엔드 summary와 가중치는 controller로 자동 복사되지 않으므로 계획에 적힌 원격 출력 경로에서 확인합니다.

TRL의 `summary-train.json`과 저장물을 아래 기준으로 확인합니다.

| 확인 대상 | 성공 조건 | 증명하지 않는 것 |
| --- | --- | --- |
| Dry-run | 설정 검증·계획 생성·exit 0 | 원격 파일, GPU, SSH 준비 |
| Controller 실행 | manifest의 `passed`, 모든 rank exit 0 | 학습 품질 |
| 학습 | 예상 optimizer step, finite loss·gradient, 저장물 | 장기 수렴, 다른 모델의 메모리 적합성 |
| Async 저장 | 필요한 shard와 pending save finalization | fsync·장애 후 durability |

모든 rank 로그와 모델·데이터 revision, 실제 선택한 입력, topology, seed, 환경 버전을 함께 확인합니다.
Sharded backend의 optimizer-step 증거를 전체 parameter checksum 검증으로 읽지 않습니다.
현재 지원 상태와 30B 검증 범위는 [Experiments](experiments.md)를 따릅니다.

## Troubleshooting

| 증상 | 확인 | 다음 조치 |
| --- | --- | --- |
| 원격 시작 거부 | checkout commit·dirty 상태 | 사용자 변경을 보존하며 같은 깨끗한 commit 준비 |
| Rendezvous 대기·timeout | 모든 rank 로그, 주소·포트·SSH·NCCL | 원인 해결 후 새 출력으로 재실행 |
| Dataset revision 불일치 | 실험과 manifest의 ID·revision | 입력과 실험을 함께 맞춤 |
| `exceeds max_length` | 해당 예제의 토큰 길이 | 길이·메모리 예산 검토 후 실험 변경 |
| 출력 claim 실패 | 부모 경로·권한·공유 mount | 부모 경로 준비 후 새 출력 사용 |
| CUDA OOM | rank 로그, 다른 작업, 메모리 계측 | 다른 작업을 임의 종료하지 않고 소형 preset부터 재확인 |
| `CERTIFICATE_VERIFY_FAILED` / `client has been closed` | venv의 CA 번들 | `SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt` 지정 |

정리 오류가 남으면 manifest와 해당 run의 process 상태를 확인하며, 다른 작업은 종료하지 않습니다.
