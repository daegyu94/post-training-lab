# Megatron Bridge Qwen2.5-7B SFT Lab

이 브랜치는 `Qwen/Qwen2.5-7B-Instruct`에 Megatron Bridge의 Qwen recipe로 LoRA supervised fine-tuning(SFT)을 적용하고, 학습 전후의 held-out loss와 checkpoint 재로딩 경로를 확인하는 단일 GPU 실습입니다.

실제 학습은 TP=1, PP=1, CP=1, DP=1로 실행됩니다. 따라서 이 브랜치는 Megatron 기반 model·dataset·checkpoint workflow를 검증하는 출발점이며, multi-GPU 성능이나 확장성을 측정하는 benchmark는 아닙니다.

## Requirements

- Linux와 NVIDIA GPU 1개
- CUDA를 사용할 수 있고 BF16을 지원하는 GPU
- Python virtual environment를 생성할 수 있는 환경
- 약 20GiB 이상의 GPU memory
- model과 결과를 위한 약 16GB 이상의 disk 공간
- Hugging Face에서 model과 dataset을 받을 수 있는 network 연결

기본 설정은 sequence length 512, micro batch size 1, global batch size 8, activation recomputation을 사용합니다. VRAM 20GiB는 preflight의 최소 기준이며, driver·CUDA·allocator 상태에 따라 더 많은 memory가 필요할 수 있습니다.

## Setup

```bash
git clone -b megatron https://github.com/daegyu94/post-training-lab.git
cd post-training-lab
./scripts/setup.sh
```

setup script는 `.venv`를 만들고 `requirements.txt`의 PyTorch, Megatron Bridge, dataset 도구를 설치합니다. 이미 environment가 있으면 같은 경로를 재사용합니다.

## Prepare the Model and Dataset

Hugging Face 인증이 필요한 환경이라면 먼저 로그인합니다.

```bash
.venv/bin/hf auth login
```

Qwen2.5-7B checkpoint와 UltraChat train/evaluation subset을 준비합니다.

```bash
./scripts/download_model.sh
./scripts/prepare_data.sh
```

| Item | Default |
| --- | --- |
| Model | `models/Qwen2.5-7B-Instruct` |
| Dataset | `HuggingFaceH4/ultrachat_200k` |
| Training subset | 32 conversations from `train_sft` |
| Evaluation subset | 8 conversations from `test_sft` |
| Prepared data | `data/ultrachat_200k/{training,validation}.jsonl` |
| Output | `results/qwen2.5-7b-megatron-experiment` |

`MODEL_DIR`, `DATA_DIR`, `TRAIN_SAMPLES`, `EVAL_SAMPLES`, `SEED`로 준비 경로와 subset을 바꿀 수 있습니다.

## Run the Experiment

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/run_experiment.sh
```

script는 preflight를 통과한 후 다음 세 stage를 각각 새 `torchrun` process로 실행합니다.

1. pretrained checkpoint의 held-out evaluation
2. 기본 5 optimizer step의 LoRA SFT와 checkpoint 저장
3. 저장한 checkpoint를 새 process에서 다시 읽은 held-out evaluation

학습 조건과 경로는 environment variable로 바꿀 수 있습니다.

```bash
MODEL_DIR=/path/to/Qwen2.5-7B-Instruct \
DATA_DIR=/path/to/prepared-data \
OUTPUT_DIR=/path/to/output \
MAX_STEPS=10 \
CUDA_VISIBLE_DEVICES=0 \
./scripts/run_experiment.sh
```

## Outputs and Verification

정상 실행 후 output directory는 다음 핵심 artifact를 포함합니다.

```text
results/qwen2.5-7b-megatron-experiment/
├── base-eval.log
├── train.log
├── tuned-eval.log
├── checkpoints/
└── summary.json
```

`summary.json`에서 다음을 확인합니다.

- `validation.checkpoint_reload_verified`가 `true`인지 확인합니다.
- `quality.tuned_eval_loss`와 `quality.tuned_perplexity`가 대응하는 base 값보다 낮은지 비교합니다.
- 세 log에 `nan`, 무한대, checkpoint load 오류가 없는지 확인합니다.

```bash
.venv/bin/python -m json.tool results/qwen2.5-7b-megatron-experiment/summary.json
```

짧은 subset과 5 step에서의 loss 변화는 workflow 검증 신호일 뿐, 일반적인 모델 품질이나 cluster-scale 성능을 의미하지 않습니다.

`summary.json`은 `schema_version: 1`과 `configuration`, `environment`, `quality`, `performance`, `artifacts`, `validation` section을 사용합니다. 현재 Megatron workflow는 안정적으로 추출하는 performance metric이 없으므로 `performance`는 빈 object로 기록합니다.

## CPU-only Concept Exercise

GPU 초기화, checkpoint download, NCCL 통신 없이 Bridge recipe와 논리적 parallel rank group을 살펴볼 수 있습니다.

```bash
./scripts/run_megatron_practice.sh
```

기본 예제는 TP=2, PP=2, CP=2, DP=2인 16개 논리 rank를 출력합니다. 실제 distributed process group을 만들지는 않습니다. 자세한 내용은 [Megatron stack overview](docs/megatron-overview.md)를 참고하세요.

## Repository Layout

- `megatron_lab/config.py`: Qwen2.5-7B LoRA recipe와 single-GPU 설정
- `megatron_lab/sft.py`: base, train, tuned stage 진입점
- `megatron_lab/prepare_data.py`: deterministic UltraChat subset 준비
- `megatron_lab/compare.py`: base와 reloaded-checkpoint loss 비교
- `megatron_lab/preflight.py`: GPU, dependency, input path 검사
- `megatron_lab/inspect_recipe.py`: GPU 초기화 없는 recipe 요약
- `megatron_lab/parallelism.py`: TP/PP/CP/DP rank group simulation
- `run_summary.py`: framework 공통 summary schema
- `scripts/run_experiment.sh`: end-to-end experiment orchestration
- `tests/`: data, log parsing, rank layout의 CPU unit tests

## Run the CPU Tests

model checkpoint나 GPU 없이 data selection, log parsing, parallel rank layout을 검증합니다.

```bash
.venv/bin/python -m pytest -q
```

## Limitations

이 branch는 실제 multi-GPU·multi-node launcher, distributed checkpoint scale test, throughput benchmark를 제공하지 않습니다. 학습 전후의 data·checkpoint·serving lifecycle은 `system-integration` branch에서 설계 문서로 설명하며, cluster resource 분석은 `profiling` branch에서 다룹니다.
