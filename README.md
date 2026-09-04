# TRL QLoRA Reproduced Result

이 브랜치는 TRL을 사용해 `Qwen/Qwen2.5-14B-Instruct`를 `HuggingFaceH4/ultrachat_200k`의 대화 데이터로 NF4 QLoRA fine-tuning한 결과를 재현하고 기록하기 위한 브랜치입니다.

재현 과정은 local dataset과 cached model을 사용한 학습, 학습 전후 held-out evaluation, deterministic generation 비교, 저장한 PEFT adapter의 독립 프로세스 재로딩으로 구성됩니다.

## Reproduced Result

재현 결과의 환경, configuration, 실행 명령, console output, metric, generation 비교, adapter reload 결과는 [docs/reproduced-result.md](docs/reproduced-result.md)에 기록되어 있습니다.

이 문서의 목적은 새 실험을 설계하는 것이 아니라, 기록된 configuration과 command를 동일하게 실행하여 결과를 다시 확인하는 것입니다.

## Requirements

- Linux와 NVIDIA CUDA GPU
- 약 18GiB 이상의 사용 가능한 GPU memory
- Python 3.10 이상
- Python virtual environment를 생성할 수 있는 환경
- `Qwen/Qwen2.5-14B-Instruct` model cache
- `HuggingFaceH4/ultrachat_200k` dataset을 다운로드할 수 있는 Hugging Face Hub 연결

검증한 구성은 NVIDIA RTX PRO 4000 Blackwell 24GiB GPU 한 장, BF16 compute, sequence length 512, micro batch 1입니다.

## Setup

의존성과 재현에 사용하는 UltraChat parquet 파일을 준비합니다.

```bash
./scripts/setup.sh
```

setup.sh는 Python virtual environment를 .venv에 만들고 requirements.txt의 의존성을 설치한 뒤 data/ultrachat_200k/data에 train_sft와 test_sft parquet split을 저장합니다.

다른 위치에 dataset을 저장하려면 다음처럼 실행합니다.

```bash
DATASET_DIR=<dataset-root> ./scripts/setup.sh
```

재현 command는 HF_HUB_OFFLINE=1, HF_DATASETS_OFFLINE=1, --local-files-only를 사용하므로 실행 전에 model과 dataset이 local cache 또는 지정한 local directory에 준비되어 있어야 합니다.

## Run the Reproduced Result

기록된 결과와 동일한 configuration으로 실행하려면 다음 command를 사용합니다.

```bash
./scripts/run_reproduced.sh
```

이 script는 다음 configuration을 사용합니다.

| Item | Value |
| --- | --- |
| Model | `Qwen/Qwen2.5-14B-Instruct` |
| Dataset | `HuggingFaceH4/ultrachat_200k` |
| Train subset | 128 conversations |
| Evaluation subset | 16 requests, validation 후 최대 15 conversations |
| Quantization | NF4 4-bit, double quantization, BF16 compute |
| LoRA | rank 16, alpha 32, dropout 0.05 |
| Sequence length | 512 |
| Effective batch | 8 |
| Optimizer steps | 20 |
| Learning rate | `2e-4` |
| Seed | 42 |

script가 수행하는 실제 Python command는 다음과 같습니다.

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  .venv/bin/python -m sft_lab.train \
  --dataset-parquet-dir data/ultrachat_200k/data \
  --local-files-only \
  --train-samples 128 \
  --eval-samples 16 \
  --max-steps 20 \
  --max-length 512 \
  --gradient-accumulation-steps 8 \
  --output-dir results/trl-branch-reproduction
```

GPU와 dataset 경로를 바꾸려면 script를 수정하지 않고 environment variable을 지정할 수 있습니다.

```bash
CUDA_VISIBLE_DEVICES=0 DATASET_PARQUET_DIR=<dataset-parquet-dir> OUTPUT_DIR=<output-dir> \
  ./scripts/run_reproduced.sh
```

## Output Files

실행이 완료되면 다음 결과가 생성됩니다.

```text
results/trl-branch-reproduction/
|-- adapter/       PEFT adapter와 tokenizer
|-- checkpoints/   Trainer checkpoint
\-- summary.json   configuration, environment, quality, performance, generation
```

summary.json은 Git에서 제외됩니다.

## Verify the Result

학습 전후의 핵심 결과는 다음 항목으로 확인합니다.

| Signal | Check | Meaning |
| --- | --- | --- |
| Held-out loss | `tuned_eval_loss < base_eval_loss` | 같은 held-out assistant token에서 loss가 낮아졌는지 확인합니다. |
| Loss change | `loss_change_percent < 0` | 학습 전후 held-out loss의 상대 변화를 확인합니다. |
| Perplexity | `tuned_perplexity < base_perplexity` | loss를 지수 변환한 보조 지표입니다. |
| Training log | `loss`와 `grad_norm`이 finite인지 확인 | 학습 과정의 수치 안정성을 확인합니다. |
| Adapter output | `adapter/`가 생성되었는지 확인 | PEFT adapter 저장이 완료되었는지 확인합니다. |

각 optimizer step의 loss는 서로 다른 training batch에서 계산되므로 학습 중 항상 감소할 필요는 없습니다. 최종 학습 효과는 동일한 held-out subset에서 측정한 base_eval_loss와 tuned_eval_loss를 비교하여 판단합니다.

```bash
.venv/bin/python -m json.tool results/trl-branch-reproduction/summary.json
```

기록된 실행의 주요 결과는 [docs/reproduced-result.md](docs/reproduced-result.md)의 Metrics section에서 확인할 수 있습니다.

## Reload the Adapter

학습 프로그램이 종료된 뒤 저장한 adapter를 새 Python 프로세스에서 다시 불러와 inference할 수 있습니다.

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
  .venv/bin/python -m sft_lab.infer \
  results/trl-branch-reproduction/adapter \
  --local-files-only \
  --prompt 'Give two practical tips for debugging an out-of-memory error during LLM training.'
```

이 command가 오류 없이 실행되고 답변을 출력하면 adapter 저장·재로딩 경로가 동작한 것입니다.

## Implementation Notes

Qwen의 기본 chat template만 사용하면 assistant-only loss를 위한 generation mask가 생성되지 않을 수 있습니다. `sft_lab.data.QWEN_ASSISTANT_MASK_TEMPLATE`은 assistant content와 `<|im_end|>`를 generation block으로 감싸며, `sft_lab.train`은 학습 시작 전에 실제 assistant mask가 생성되는지 확인합니다.

QLoRA는 frozen 4-bit base weight에 LoRA parameter만 추가하여 학습합니다. 따라서 이 결과에서 확인하는 adapter는 full model checkpoint가 아니라 원본 Qwen model과 결합해야 사용하는 PEFT adapter입니다.

## Limitations

이 결과는 128개 training conversation, 15개 held-out conversation, 20 optimizer steps로 실행한 짧은 재현 run입니다. held-out loss 감소와 adapter reload 성공은 구현된 학습 경로가 동작했음을 보여주지만, 일반적인 instruction-following 품질이나 benchmark 성능 향상을 의미하지는 않습니다.

## References

- [Reproduced Result](docs/reproduced-result.md)
- [TRL SFTTrainer documentation](https://huggingface.co/docs/trl/sft_trainer)
