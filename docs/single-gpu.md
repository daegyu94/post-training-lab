# Setup1: Single-GPU Qwen2.5-7B SFT

Setup1은 단일 GPU에서 LoRA SFT와 학습 전후 평가, checkpoint reload 흐름을 익히는 실습입니다.
TP=1, PP=1, CP=1, DP=1이며 CLI에서는 `--setup single`에 해당합니다.
구성 비교는 [README](../README.md), 두 노드 실행은 [Setup2 guide](spark-cluster.md)를 참고하세요.

명령은 GPU 실행 노드의 repository root에서 실행합니다.
Controller는 개발과 실행 조율을 담당하며, Spark에서는 SSH로 접속해 `/home/spark/shared/post-training-lab`에서 작업합니다.
공유 repository가 있으면 clone 단계는 생략합니다.
Spark의 ARM64 환경은 [software prerequisites](spark-cluster.md#software-prerequisites)를 따라 준비합니다.
Setup1 스크립트가 사용하는 `.venv/bin/python`과 `.venv/bin/torchrun`에 호환 환경이 필요하며 아래 일반 설치 스크립트만으로 Spark 환경이 완성되지는 않습니다.

## Requirements

- Linux와 NVIDIA GPU 1개
- CUDA를 사용할 수 있고 BF16을 지원하는 GPU
- Python virtual environment를 생성할 수 있는 환경
- 약 20GiB 이상의 GPU memory
- model checkpoint용 약 16GB와 dataset·학습 checkpoint·log를 위한 추가 disk 공간
- Hugging Face에서 model과 dataset을 받을 수 있는 network 연결

기본 설정은 sequence length 512, micro batch size 1, global batch size 8, activation recomputation을 사용합니다.
VRAM 20GiB는 preflight의 최소 기준이며, driver·CUDA·allocator 상태에 따라 더 많은 memory가 필요할 수 있습니다.

## Install the Environment

```bash
git clone -b megatron https://github.com/daegyu94/post-training-lab.git
cd post-training-lab
./scripts/setup.sh
```

setup script는 `.venv`를 만들고 `requirements.txt`의 PyTorch, Megatron Bridge, dataset 도구를 설치합니다.
이미 environment가 있으면 같은 경로를 재사용합니다.

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

UltraChat 외 public source의 schema adapter와 canonical JSONL preparation은 [public dataset guide](public-datasets.md)를 참고하세요.
출력은 `DATA_DIR`와 `--train-data`/`--eval-data`로 연결할 수 있지만, dataset 호환성은 model/tokenizer loss-mask나 GPU training 검증을 의미하지 않습니다.

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
+-- base-eval.log
+-- train.log
+-- tuned-eval.log
+-- checkpoints/
+-- summary.json
```

`summary.json`에서 다음을 확인합니다.

- `validation.checkpoint_reload_verified`가 `true`인지 확인합니다.
- `quality.tuned_eval_loss`와 `quality.tuned_perplexity`가 대응하는 base 값보다 낮은지 비교합니다.
- 세 log에 `nan`, 무한대, checkpoint load 오류가 없는지 확인합니다.

```bash
.venv/bin/python -m json.tool results/qwen2.5-7b-megatron-experiment/summary.json
```

짧은 subset과 5 step에서의 loss 변화는 workflow 검증 신호일 뿐, 일반적인 모델 품질이나 cluster-scale 성능을 의미하지 않습니다.

`summary.json`은 `schema_version: 1`과 `configuration`, `environment`, `quality`, `performance`, `artifacts`, `validation` section을 사용합니다.
현재 Megatron workflow는 안정적으로 추출하는 performance metric이 없으므로 `performance`는 빈 object로 기록합니다.
