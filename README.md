# TRL Post-Training Lab

이 브랜치는 TRL 기반 SFT의 데이터 준비, 학습, 평가와 checkpoint·adapter 저장을 실습합니다.
Single-GPU QLoRA부터 두 노드의 BF16 LoRA 및 full-parameter 학습 경로까지 다룹니다.
현재 학습 구현은 SFT이며, DPO와 RL trainer는 포함하지 않습니다.

공통 하드웨어 구성과 Setup 1·2 소개는 [main의 PoC Setups](https://github.com/daegyu94/post-training-lab/blob/main/README.md#poc-setups)에서 확인하세요.

## Setup 1: Single-GPU QLoRA

한 GPU에서 QLoRA 학습과 adapter 저장·재로딩 흐름을 익히는 경로입니다.
`scripts/setup.sh`로 `.venv`와 UltraChat parquet을 준비하고, 모델 cache까지 준비한 뒤 `scripts/run_experiment.sh`를 실행합니다.
학습 전후 held-out 평가와 generation 비교 결과는 `results/qwen2.5-14b-qlora/summary.json`에, 학습한 adapter는 같은 디렉터리의 `adapter/`에 저장됩니다.

[Setup 1 가이드](docs/setup1.md)에서 설치 → 데이터·모델 준비 → 실행 → 결과 확인 → adapter 재로딩 순서로 진행하세요.
[실험 기록](docs/experiment-result.md)은 128개 training conversation, 15개 held-out conversation과 20 optimizer steps를 사용합니다.
이 짧은 실험의 loss 변화는 일반적인 모델 품질 향상을 입증하지 않습니다.

## Setup 2: Two-Node Spark SFT

두 Spark 노드에서 분산 SFT를 실행하고 finite loss, optimizer step, rank별 정상 종료와 저장 결과를 확인하는 경로입니다.
Controller는 실행을 조율하고 실제 학습은 `spark1`, `spark2`에서 수행합니다.
각 노드에 ARM64/CUDA 호환 Python 환경과 같은 revision의 모델 snapshot을 준비합니다.
모델 weight는 node-local cache에 두고, 공유 데이터와 결과는 Spark 노드에서 접근하는 NFS 경로를 사용합니다.

[Setup 2 가이드](docs/spark-cluster.md)에서 runtime·통신 확인 → 모델·데이터 준비 → 두 노드 실행 → 결과 확인 순서로 진행하세요.
Setup 1용 `scripts/setup.sh`는 `requirements.txt`를 설치하므로 Spark 환경 준비를 대신하지 않습니다.
DDP의 기본 workflow는 `base`, `train`, `tuned`를 각각 새 process로 실행하고 `summary-<stage>.json`과 rank logs를 남깁니다.
FSDP2와 DeepSpeed는 현재 `base` 또는 `train` stage만 지원합니다.

### Training Modes and Verification

`FINETUNING_MODE=lora|full`로 학습 대상을, `DISTRIBUTED_BACKEND=ddp|fsdp2|deepspeed`로 분산 방식을 선택합니다.
Full mode에서는 `OPTIMIZER=sgd|adamw`를 선택할 수 있으며 메모리 예산을 먼저 확인해야 합니다.
아래는 기존 문서의 실행 기록이며, 지원 옵션 전체가 모든 모델에서 검증됐다는 뜻은 아닙니다.

| 구성 | 기록된 검증 범위 |
| --- | --- |
| Qwen3/GLM 30B DDP LoRA | Two-node 1-step 학습, finite loss, sampled parameter update, adapter 저장 |
| Qwen2.5-0.5B DDP LoRA | Two-node 2-step 학습 및 별도 process adapter reload 평가 |
| 소형 모델 full-parameter FSDP2 | Two-node 1-step 학습과 sharded checkpoint 저장; reload 미검증 |
| DeepSpeed ZeRO-2/3 | Configuration·launcher unit test; Spark GPU runtime 미검증 |
| 30B full-parameter 및 30B sharded backend | 메모리 적합성과 학습 runtime 미검증 |

짧은 integration 결과를 장기 수렴, 품질 또는 throughput 결과로 해석하지 않습니다.
정확한 조건과 수치는 [Setup 2 실행 기록](docs/spark-cluster.md#verified-integration-smoke)을 참고하세요.

## Dataset Guides

[공개 데이터 가이드](docs/public-datasets.md)는 No Robots, Self-OSS, xLAM 등의 schema 변환과 revision·manifest를 설명합니다.
[사내 데이터 가이드](docs/internal-data-guide.md)는 승인된 service trace의 정제, split과 test 정답 분리를 설명합니다.
학습에 전달하는 옵션은 Setup 1의 `--dataset-jsonl-dir`, Setup 2의 `DATA_DIR`입니다.

## Repository Layout and Tests

| 경로 | 역할 |
| --- | --- |
| `sft_lab/train.py`, `sft_lab/infer.py` | Setup 1 학습과 adapter inference |
| `sft_lab/spark_train.py`, `sft_lab/spark_config.py` | Setup 2 학습과 설정·snapshot 검사 |
| `scripts/` | 설치, 데이터 준비와 실행 launcher |
| `configs/` | DeepSpeed ZeRO 설정 |
| `tests/` | 데이터, 설정과 launcher 검증 |

선택한 setup의 환경을 활성화한 뒤 저장소 루트에서 CPU unit test를 실행합니다.

```bash
python -m pytest -q
```

Model weight, dataset cache, checkpoint와 큰 실행 산출물은 Git에 저장하지 않습니다.
