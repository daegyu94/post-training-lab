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

## Smoke Run

```bash
./scripts/run_smoke.sh
```

기본 smoke run은 train 32개, evaluation 8개, optimizer step 5회를 사용합니다. 실제 학습 경향을 더 안정적으로 확인하려면 sample과 step을 늘립니다.

```bash
TRAIN_SAMPLES=128 EVAL_SAMPLES=32 MAX_STEPS=20 OUTPUT_DIR=results/qwen2.5-14b-qlora ./scripts/run_smoke.sh
```

Hub 연결 없이 이미 받은 parquet를 사용하려면 dataset snapshot의 `data` directory를 지정할 수 있습니다.

```bash
./scripts/run_smoke.sh --dataset-parquet-dir <ultrachat-snapshot>/data --local-files-only
```

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

## Troubleshooting

- CUDA out-of-memory가 발생하면 `--max-length 384`로 낮추고 다른 GPU process를 종료합니다.
- Hub 접근이 불안정하면 모델과 dataset을 먼저 cache한 뒤 `--local-files-only`와 `--dataset-parquet-dir`를 사용합니다.
- loss가 `nan`이면 assistant mask 검사 오류가 없는지 확인하고 learning rate를 낮춥니다.
- `No *-*.parquet files` 오류가 나면 `data` 하위 directory를 지정했는지 확인합니다.
