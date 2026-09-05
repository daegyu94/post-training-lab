# Megatron Bridge Qwen2.5-7B SFT Experiment

이 브랜치는 Megatron Bridge와 Megatron Core를 사용해 `Qwen/Qwen2.5-7B-Instruct`에 LoRA SFT를 적용하고, 학습 전후 held-out loss를 비교하는 실험을 제공합니다. 기본 데이터는 `HuggingFaceH4/ultrachat_200k`이며, Hugging Face checkpoint는 AutoBridge를 통해 Megatron model provider로 불러옵니다.

## 요구 사항

- Linux와 NVIDIA CUDA GPU 1개
- BF16을 지원하는 GPU와 약 20GiB 이상의 사용 가능한 GPU memory
- Python virtual environment를 만들 수 있는 환경
- Hugging Face에서 model과 dataset을 내려받을 수 있는 연결
- model과 결과를 저장할 약 16GB 이상의 disk 공간

기본 구성은 sequence length 512, micro batch size 1, activation recomputation, TP=1, PP=1, CP=1, DP=1입니다. 실제 memory 사용량은 driver, CUDA, allocator 상태에 따라 달라지므로 짧은 실행부터 확인해야 합니다.

## 설치

```bash
git clone -b megatron-lab https://github.com/daegyu94/sft-lab.git
cd sft-lab
./scripts/setup.sh
```

## 모델과 공개 데이터 준비

Hugging Face 인증이 필요한 환경이라면 `.venv/bin/hf auth login`을 먼저 실행합니다.

```bash
./scripts/download_model.sh
./scripts/prepare_data.sh
```

| 항목 | 기본 경로 |
| --- | --- |
| Model | `models/Qwen2.5-7B-Instruct` |
| Train data | `data/ultrachat_200k/training.jsonl` |
| Validation data | `data/ultrachat_200k/validation.jsonl` |
| Result | `results/qwen2.5-7b-megatron-experiment` |

`prepare_data.sh`는 UltraChat의 서로 겹치지 않는 train과 evaluation split에서 작은 subset을 받아 Megatron Bridge가 읽는 conversational JSONL로 저장합니다.

## 사내 서비스 데이터 준비

사내 LLM 서비스 trace와 benchmark에서 train, validation, test 데이터를 만드는 기준과 실습은 [사내 LLM 서비스 데이터 가이드](docs/internal-data-guide.md)를 참고하세요. 실습 converter는 승인된 synthetic trace만 선택하고, session 단위 split, 중복 prompt 제거, test 정답 분리, manifest 생성을 수행합니다.

```bash
./scripts/prepare_service_data.sh
```

생성한 데이터로 실험하려면 `DATA_DIR=data/service-sft`를 지정합니다. `test.jsonl`의 `reference_answer`와 `grader`는 현재 학습 recipe에 전달되지 않습니다.

## 실험 실행

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/run_experiment.sh
```

경로와 일부 학습 조건은 environment variable로 바꿀 수 있습니다.

```bash
MODEL_DIR=<model-dir> \
DATA_DIR=<prepared-data-dir> \
OUTPUT_DIR=<output-dir> \
MAX_STEPS=5 \
CUDA_VISIBLE_DEVICES=0 \
./scripts/run_experiment.sh
```

실행 script는 pretrained checkpoint 평가, LoRA 학습과 checkpoint 저장, 저장한 LoRA checkpoint 재로딩 평가를 차례로 수행합니다. 결과 directory의 `base-eval.log`, `train.log`, `tuned-eval.log`, `summary.json`에서 각 단계를 확인할 수 있습니다.

## Megatron 개념 실습

GPU 없이 Megatron Bridge recipe와 TP, PP, CP, DP의 논리적 rank group을 살펴보려면 다음을 실행합니다.

```bash
./scripts/run_megatron_practice.sh
```

이 command는 checkpoint를 내려받거나 CUDA, NCCL, `torch.distributed`를 시작하지 않습니다. 자세한 설명은 [Megatron-LM, Megatron Core, Megatron Bridge 개요](docs/megatron-overview.md)를 참고하세요.

## 주요 구현

- `megatron_lab/config.py`: Qwen2.5-7B LoRA recipe와 `DirectHFSFTDatasetConfig`
- `megatron_lab/prepare_data.py`: UltraChat train·validation JSONL 준비
- `megatron_lab/prepare_service_data.py`: 검수된 서비스 trace 변환과 test 정답 분리
- `megatron_lab/sft.py`: Bridge 기반 학습과 evaluation stage
- `megatron_lab/compare.py`: 학습 전후 evaluation 결과 비교
- `megatron_lab/preflight.py`: GPU, model, dataset 사전 검사
- `megatron_lab/parallelism.py`: TP, PP, CP, DP rank group 시뮬레이션
- `scripts/run_experiment.sh`: 전체 실험 실행
