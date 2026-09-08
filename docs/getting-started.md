# Getting Started

이 문서는 controller에서 Spark backend를 처음 실행하는 최소 경로를 설명합니다.
Controller는 설정을 검증하고 SSH로 원격 launcher를 시작합니다.
실제 학습 process는 설정한 Spark node에서 실행됩니다.

## 사전 조건

Controller에는 Python 3.10 이상과 OpenSSH client가 필요합니다.
각 Spark node에는 같은 commit의 repository checkout, backend별 Python 환경, 모델 snapshot과 dataset directory가 필요합니다.
Runner는 GPU당 process 하나만 지원하며 setup과 experiment는 한 개 또는 두 개 node를 사용할 수 있습니다.

TRL Spark 환경은 backends/trl/requirements-spark.txt를 사용합니다.
Megatron Spark 환경은 backends/megatron/requirements-spark.txt를 사용하며 CUDA Torch를 먼저 준비해야 합니다.
실제 버전 제약은 각 requirements 파일을 확인합니다.

## Setup 파일 만들기

Repository root에서 예제 setup을 복사합니다.

```bash
cp setups/spark/local.example.json setups/spark/local.json
```

local.json에는 node별 실제 경로와 host를 입력합니다.
이 파일은 Git에 추가하지 않도록 ignore 설정되어 있습니다.

| 필드 | 의미 |
| --- | --- |
| master_addr, master_port | distributed rendezvous 주소 |
| nodes[].host | SSH host |
| nodes[].checkout | 해당 node의 repository checkout |
| nodes[].python.trl, nodes[].python.megatron | backend별 Python executable |
| nodes[].model_dirs | MODEL_ID별 node-local model snapshot |
| nodes[].data_dir | prepared JSONL이 보이는 경로 |
| nodes[].output_root | run별 output directory의 부모 경로 |
| env | NCCL*, OMP*, HF_* 같은 hardware/runtime 변수 |

두 node를 사용할 때 checkout은 같은 commit이어야 합니다.
모델 snapshot은 참여하는 모든 node에 있어야 합니다.
Checkpoint를 공유해서 재로딩할 때는 두 node에서 같은 output_root를 가리켜야 합니다.

## 데이터와 모델 준비

학습 데이터는 [Datasets](datasets.md)의 backend별 명령으로 준비합니다.
Experiment의 MODEL_REVISION과 DATASET_REVISION에는 40자리 immutable commit SHA를 사용합니다.
Setup file의 model_dirs key는 experiment의 MODEL_ID와 정확히 일치해야 합니다.

## Dry-run 실행

다음 명령은 node에 접속하지 않고 setup과 experiment를 검증하고 plan을 출력합니다.
Output directory가 이미 존재하면 runner는 덮어쓰지 않고 실패합니다.

```bash
python experiments/run.py \
  --backend trl \
  --setup setups/spark/local.json \
  --experiment experiments/trl/smoke.json \
  --output artifacts/runs/trl-smoke
```

실행 전 plan에서 backend, node 수, remote output path와 environment를 확인합니다.
--backend 값은 experiment의 backend와 같아야 합니다.

## 원격 실행

Dry-run 결과를 확인한 뒤 같은 명령에 --execute를 추가합니다.
기본 timeout은 900초이며 양의 정수 --timeout으로 바꿀 수 있습니다.

```bash
python experiments/run.py \
  --backend trl \
  --setup setups/spark/local.json \
  --experiment experiments/trl/smoke.json \
  --output artifacts/runs/trl-smoke \
  --execute
```

Controller output에는 manifest.json과 rank별 rank-<n>.log가 생성됩니다.
Manifest에는 설정 hash, controller commit, host, remote command, exit status와 remote checkout 상태가 기록됩니다.

## 사용할 preset

| 파일 | 용도 |
| --- | --- |
| experiments/trl/single-node-smoke.json | TRL one-node DDP LoRA smoke |
| experiments/trl/smoke.json | TRL two-node DDP LoRA smoke |
| experiments/megatron/smoke.json | Megatron two-node full SFT smoke |
| experiments/megatron/resume-smoke.json | Megatron checkpoint load와 resumed step smoke |

Smoke는 실행 경로와 수치 안정성을 확인하는 짧은 실행입니다.
Smoke 결과만으로 모델 품질, 장기 수렴 또는 일반적인 throughput을 주장하지 않습니다.

## 실패 시 확인할 항목

먼저 manifest.json의 controller_commit, rank별 status와 log를 확인합니다.
Remote checkout이 dirty이거나 controller와 commit이 다르면 runner가 시작을 거부합니다.
모델 snapshot, dataset manifest, output 권한과 SSH BatchMode 접속도 확인합니다.
Megatron은 world size, TP/PP/EP와 batch size의 나눗셈 조건을 추가로 검사합니다.
