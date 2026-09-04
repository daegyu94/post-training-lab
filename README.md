# TRL QLoRA Lab

이 브랜치는 TRL로 `Qwen/Qwen2.5-14B-Instruct`를 `HuggingFaceH4/ultrachat_200k`의 대화에 맞춰 4-bit QLoRA fine-tuning하는 독립 실습입니다. 학습 전후의 동일한 held-out `test_sft` loss와 deterministic generation을 비교하고, 저장한 PEFT adapter를 새 process에서 다시 불러오는 단계까지 다룹니다.

## What You Will Verify

- NF4 4-bit base model 위에 LoRA adapter만 학습되는지 확인합니다.
- UltraChat의 `train_sft`와 `test_sft`를 섞지 않고 사용합니다.
- Qwen chat template에 generation mask를 추가해 assistant 응답 token만 loss에 포함합니다.
- 학습 전후 held-out loss와 perplexity를 같은 sample에서 비교합니다.
- adapter checkpoint 저장과 독립 inference 재로딩을 확인합니다.

## Requirements

- Linux와 NVIDIA CUDA GPU
- 약 18GiB 이상의 사용 가능한 GPU memory
- Python 3.10 이상
- 모델과 dataset을 받을 수 있는 Hugging Face Hub 연결 또는 미리 준비한 cache

검증한 기본 구성은 24GiB GPU 한 장, BF16 compute, sequence length 512, micro batch 1입니다. 짧은 smoke run은 학습 경로를 확인하기 위한 것이며 일반화 품질을 보장하지 않습니다.

## Setup

```bash
./scripts/setup.sh
```

설치 후 GPU와 quantization backend를 확인합니다.

```bash
.venv/bin/python -c "import torch, bitsandbytes; print(torch.cuda.get_device_name(0)); print(torch.cuda.is_bf16_supported())"
```

## Dataset

`setup.sh`는 의존성을 설치한 뒤 필요한 UltraChat SFT parquet split만 `data/ultrachat_200k/data`에 미리 내려받습니다. 이후 smoke run은 이 로컬 dataset을 사용합니다.

다른 디스크에 저장하려면 setup 시작 시 경로를 지정합니다.

```bash
DATASET_DIR=/mnt/datasets/ultrachat_200k ./scripts/setup.sh
```

## Smoke Run

```bash
./scripts/run_smoke.sh
```

기본 smoke run은 train 32개, evaluation 8개, optimizer step 5회를 사용합니다. 실제 학습 경향을 더 안정적으로 확인하려면 sample과 step을 늘립니다.

```bash
TRAIN_SAMPLES=128 EVAL_SAMPLES=32 MAX_STEPS=20 OUTPUT_DIR=results/qwen2.5-14b-qlora ./scripts/run_smoke.sh
```

`run_smoke.sh`는 기본 경로를 사용합니다. 다른 경로를 사용했다면 `DATASET_PARQUET_DIR=/mnt/datasets/ultrachat_200k/data ./scripts/run_smoke.sh`처럼 지정합니다.

## What Happens During a Run

`run_smoke.sh`는 다음 순서로 실행됩니다.

1. 로컬 UltraChat의 train/evaluation subset을 읽고, Qwen2.5-14B-Instruct와 tokenizer를 준비합니다.
2. 학습 전 base model의 held-out loss와 고정 prompt 두 개의 응답을 측정합니다.
3. NF4 4-bit base model 위에서 LoRA adapter만 학습합니다. 콘솔의 `loss`, `grad_norm`, `learning_rate`는 이 단계의 optimizer log입니다.
4. 같은 held-out subset으로 tuned model을 다시 평가하고, 같은 prompt의 응답을 생성합니다.
5. adapter, 마지막 checkpoint, 환경·품질·성능·생성 결과를 output directory에 저장합니다.

## Console Log Guide

| Console output | Meaning |
| --- | --- |
| `Loading checkpoint shards` progress | Qwen base model weight를 GPU에 준비하는 단계입니다. 처음에는 시간이 걸리고 GPU memory가 증가합니다. |
| `Filter` 또는 `Map` progress | UltraChat 대화를 검사하고 chat template와 assistant-only label을 준비하는 단계입니다. |
| `{'loss': ..., 'grad_norm': ..., 'learning_rate': ..., 'epoch': ...}` | 한 optimizer step의 학습 log입니다. `loss`는 해당 batch의 loss, `grad_norm`은 gradient 크기, `learning_rate`는 현재 cosine schedule의 learning rate입니다. |
| `{'train_runtime': ..., 'train_steps_per_second': ..., 'train_loss': ...}` | 학습 loop가 끝난 뒤의 처리 시간과 평균 training loss입니다. |
| 마지막 JSON object | 저장된 `summary.json`과 같은 내용입니다. 학습 전후 평가, generation, 성능, 환경 정보를 확인합니다. |

새 스크립트로 실행했다면 `train_gen-*.parquet` 다운로드 progress는 나타나지 않아야 합니다. 그 출력이 다시 보이면 이전 버전의 `run_smoke.sh`가 실행 중이거나 `sft_lab.train`을 직접 실행한 경우입니다.

## Did Training Work?

| Signal | What to check | Interpretation |
| --- | --- | --- |
| Held-out loss | `tuned_eval_loss < base_eval_loss` | 이 실습에서 학습 성공을 판단하는 주 지표입니다. 같은 평가 subset에서 낮아져야 합니다. |
| `loss_change_percent` | 음수 | held-out loss의 상대 변화입니다. 예를 들어 `-20`은 loss가 20% 감소했다는 뜻입니다. |
| Perplexity | `tuned_perplexity < base_perplexity` | loss를 사람이 비교하기 쉽게 변환한 보조 지표입니다. loss와 같은 방향으로 감소해야 합니다. |
| Step `loss` | 큰 폭의 발산이나 `nan` 없음 | batch마다 달라 단조 감소하지 않아도 됩니다. 학습 안정성 확인용이지 최종 품질 지표는 아닙니다. |
| `grad_norm` | 유한한 값으로 유지 | `nan` 또는 계속 커지는 값은 수치 불안정 가능성을 알립니다. 품질 향상을 직접 뜻하지는 않습니다. |
| Base/tuned generation | 고정 prompt의 응답 비교 | 형식·완결성의 정성 확인입니다. 두 예시만으로 일반화 품질을 결론내릴 수는 없습니다. |

따라서 완료된 run은 held-out loss 감소, 유한한 training log, adapter 저장과 재로딩 성공을 함께 확인합니다. 기본 5-step smoke run은 경로 검증용이며, 품질 경향은 20-step 예시와 [reproduced result](docs/reproduced-result.md)처럼 더 큰 subset에서 판단합니다.

## Read the Result

학습 중 출력되는 각 step의 `loss`는 서로 다른 batch에서 계산되므로 항상 감소할 필요는 없습니다. 학습 전후를 판단할 때는 완료 시 출력되는 `base_eval_loss`와 `tuned_eval_loss`를 비교합니다.

```bash
.venv/bin/python -m json.tool results/qwen2.5-14b-qlora-smoke/summary.json
```

`summary.json`에서 다음 항목을 확인합니다.

- `quality.base_eval_loss`와 `quality.tuned_eval_loss`: 같은 held-out assistant token의 loss입니다. tuned 값이 낮으면 이 subset에 대한 optimization이 진행됐다는 뜻입니다.
- `quality.loss_change_percent`, `base_perplexity`, `tuned_perplexity`: 학습 전후 품질 차이를 보조적으로 보여줍니다.
- `performance`: 학습 시간, optimizer step/s, sample/s, 최고 GPU 할당 메모리입니다.
- `generations.base`와 `generations.tuned`: 고정 prompt에 대한 학습 전후 응답 비교입니다. 정성 확인용이며 loss보다 강한 품질 근거는 아닙니다.

## Outputs

```text
results/qwen2.5-14b-qlora-smoke/
|-- adapter/       PEFT adapter와 tokenizer
|-- checkpoints/   Trainer checkpoint
`-- summary.json   구성, 환경, 학습 전후 품질, 처리량, generation
```

`summary.json`의 `base_eval_loss`와 `tuned_eval_loss`는 같은 held-out subset에서 assistant token만 측정합니다. 짧은 run에서 loss가 내려가면 선택한 sample에 대한 optimization path가 동작했다는 증거지만, 별도 benchmark의 지시 이행 능력이나 일반화 성능을 뜻하지는 않습니다.

## Adapter Reload

학습 process가 끝난 뒤 adapter를 새로 불러와 생성합니다.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m sft_lab.infer results/qwen2.5-14b-qlora-smoke/adapter --local-files-only
```

## Reproduced Result

이 저장소에서 직접 실행한 환경, 수치, 해석은 [reproduced result](docs/reproduced-result.md)에 기록합니다. 숫자만 비교하지 말고 sample 수, step, sequence length와 hardware가 같은지 함께 확인하세요.

## Design Notes

Qwen의 기본 chat template는 assistant mask를 반환하지 않으므로 그대로 `assistant_only_loss=True`를 켜면 학습 label이 비어 버릴 수 있습니다. `sft_lab.data.QWEN_ASSISTANT_MASK_TEMPLATE`은 assistant content와 `<|im_end|>`를 `{% generation %}` block으로 감싸며, 학습 시작 전에 실제 mask가 0이 아닌지 검사합니다.

QLoRA는 frozen 4-bit base weight를 직접 갱신하지 않고 attention과 MLP projection에 작은 low-rank matrix를 추가합니다. 이 구성은 full fine-tuning보다 memory를 줄이지만, adapter rank·dataset·step 수에 따라 결과가 크게 달라집니다.

## References

- [TRL SFTTrainer documentation](https://huggingface.co/docs/trl/sft_trainer): SFTTrainer, assistant-only loss, conversational dataset format, logged metric의 공식 기준입니다.

## Troubleshooting

- CUDA out-of-memory가 발생하면 `--max-length 384`로 낮추고 다른 GPU process를 종료합니다.
- Hub 접근이 불안정하면 모델과 dataset을 먼저 cache한 뒤 `--local-files-only`와 `--dataset-parquet-dir`를 사용합니다.
- loss가 `nan`이면 assistant mask 검사 오류가 없는지 확인하고 learning rate를 낮춥니다.
- `No *-*.parquet files` 오류가 나면 `data` 하위 directory를 지정했는지 확인합니다.
