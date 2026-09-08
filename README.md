# TRL Post-Training Lab

이 브랜치는 TRL 기반 SFT의 데이터 준비, 학습, 평가와 checkpoint·adapter 저장을 실습합니다.
Single-GPU QLoRA부터 두 노드의 BF16 LoRA 및 full-parameter 학습 경로까지 다룹니다.
현재 학습 구현은 SFT이며, DPO와 RL trainer는 포함하지 않습니다.
Python 패키지 `trl_lab`은 데이터 준비, 학습과 inference 진입점을 제공합니다.

공통 하드웨어 구성과 Setup 1·2 소개는 [main의 PoC Setups](https://github.com/daegyu94/post-training-lab/blob/main/README.md#poc-setups)에서 확인하세요.

데이터 선택·정제 기준은 [main의 Dataset Guides](https://github.com/daegyu94/post-training-lab/blob/main/README.md#dataset-guides), TRL 변환 명령과 학습 입력 옵션은 [데이터 준비](docs/dataset-preparation.md)를 참고하세요.

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
아래는 실제 실행한 구성별 결과이며, 지원 옵션 전체가 모든 모델에서 동작한다는 뜻은 아닙니다.
두 30B 모델에 공통으로 검증된 학습 경로는 DDP LoRA입니다.

| 구성 | 기록된 검증 범위 |
| --- | --- |
| Qwen3/GLM 30B DDP LoRA | 두 모델 모두 two-node 1-step, finite loss, nonzero sampled update, adapter 저장 재검증 완료 |
| Qwen3-30B FSDP2 LoRA | Two-node 1-step, finite loss, sharded model·optimizer checkpoint 저장 완료 |
| GLM-4.7-Flash FSDP2 LoRA | State-dict load의 `Tensor.device_mesh` AttributeError로 실패 |
| DeepSpeed ZeRO-2 | 소형 full SFT에서도 NVML memory query 오류로 실패 |
| DeepSpeed ZeRO-3 | 소형 full SFT는 성공; Qwen3/GLM 30B LoRA는 각각 초기화/첫 step 중 OOM |
| 30B full SFT | 개별 full SGD: spark1 Qwen3는 메모리 오류 관측 후 20분 시간 초과, spark2 GLM은 backward CUDA OOM; Qwen3 two-node full FSDP2도 OOM |
| 소형 모델 full-parameter FSDP2 | Sharded checkpoint 저장, export 후 별도 DDP process 평가 완료; 동일 backend 수치 일치·optimizer resume 미검증 |
| Qwen2.5-0.5B DDP LoRA | 기존 two-node 2-step 학습 및 별도 process adapter reload 평가 완료 |

짧은 integration 결과를 장기 수렴, 품질 또는 throughput 결과로 해석하지 않습니다.
이번 실행의 조건·수치·실패 원인은 [Training Verification](docs/training-verification.md)에 기록했습니다.
이전 실행은 [Setup 2 기존 기록](docs/spark-cluster.md#verified-integration-smoke)에 보존합니다.

## Repository Layout and Tests

| 경로 | 역할 |
| --- | --- |
| `trl_lab/train.py`, `trl_lab/infer.py` | Setup 1 학습과 adapter inference |
| `trl_lab/spark_train.py`, `trl_lab/spark_config.py` | Setup 2 학습과 설정·snapshot 검사 |
| `scripts/` | 설치, 데이터 준비와 실행 launcher |
| `configs/` | DeepSpeed ZeRO 설정 |
| `tests/` | 데이터, 설정과 launcher 검증 |

선택한 setup의 환경을 활성화한 뒤 저장소 루트에서 CPU unit test를 실행합니다.

```bash
python -m pytest -q
```

Model weight, dataset cache, checkpoint와 큰 실행 산출물은 Git에 저장하지 않습니다.
