# Reproduced Result

이 문서는 이 브랜치의 code로 실제 실행한 Qwen2.5-14B-Instruct QLoRA 결과를 기록합니다. 실행 산출물 전체는 Git에서 제외하고, 재현에 필요한 command와 작은 summary만 남깁니다.

## Environment

| Item | Value |
| --- | --- |
| Date | 2026-09-04 |
| GPU | NVIDIA RTX PRO 4000 Blackwell 24GiB 1장 |
| Driver | 580.173.02 |
| Python | 3.12.3 |
| PyTorch | 2.11.0+cu128 |
| Transformers | 4.57.6 |
| TRL | 1.12.0 |
| CUDA runtime | 12.8 |

## Configuration

| Item | Value |
| --- | --- |
| Model | `Qwen/Qwen2.5-14B-Instruct` |
| Dataset | `HuggingFaceH4/ultrachat_200k` |
| Train subset | 128 conversations, seed 42 |
| Evaluation subset | 요청 16개, 512-token 경계 안에 assistant label이 있는 15개 |
| Quantization | NF4 4-bit, double quantization, BF16 compute |
| LoRA | rank 16, alpha 32, dropout 0.05, attention·MLP projection |
| Sequence length | 512 |
| Micro/effective batch | 1 / 8 |
| Optimizer steps | 20 |
| Learning rate | `2e-4`, cosine schedule, warmup ratio 0.1 |

로컬 cache를 재사용한 명령은 다음과 같습니다. 일반 실행에서는 `--dataset-parquet-dir`와 `--local-files-only`를 생략할 수 있습니다.

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  .venv/bin/python -m sft_lab.train \
  --dataset-parquet-dir <ultrachat-snapshot>/data \
  --local-files-only \
  --train-samples 128 \
  --eval-samples 16 \
  --max-steps 20 \
  --max-length 512 \
  --gradient-accumulation-steps 8 \
  --output-dir results/trl-branch-reproduction
```

## Metrics

| Metric | Base | Tuned | Change |
| --- | ---: | ---: | ---: |
| Held-out assistant-token loss | 1.1978 | 0.8401 | -29.87% |
| Held-out perplexity | 3.3128 | 2.3165 | -30.08% |

| Performance metric | Result |
| --- | ---: |
| Mean training loss | 0.9498 |
| Training time | 163.89s |
| Optimizer throughput | 0.122 steps/s |
| Micro-sample throughput | 0.978 samples/s |
| Peak allocated GPU memory | 14.50GiB |

`train_sft`의 첫 logged loss는 1.5410이었고 마지막 logged loss는 0.8367이었습니다. 개별 batch 난이도가 달라 중간 값은 단조 감소하지 않았지만, 별도 `test_sft` subset의 loss가 29.87% 감소해 adapter가 단순히 training batch를 통과한 것보다 강한 optimization 증거를 보였습니다.

이 수치는 15개 held-out conversation만 사용한 짧은 run의 결과입니다. Qwen2.5-14B-Instruct 자체가 이미 instruction-tuned model이고 UltraChat과 비슷한 대화 분포에 익숙할 수 있으므로, 이를 일반적인 품질 향상이나 benchmark 점수로 해석해서는 안 됩니다.

## Generation Check

`Explain gradient accumulation in three concise bullet points.`에 대해 base와 tuned model 모두 세 항목을 생성했습니다. tuned model은 각 bullet을 더 짧게 끝냈고 96-token 한도 안에서 세 번째 항목까지 완결했습니다.

`Give two practical tips for debugging an out-of-memory error during LLM training.`에 대해 base output은 두 번째 항목에 도달하기 전에 96-token 한도에서 잘렸습니다. tuned output은 `Reduce the batch size`와 `Use gradient checkpointing` 두 항목을 한도 안에서 완결했습니다. 두 예시는 형식 준수가 개선된 방향을 보이지만 정성적 예시 두 개이므로 품질 판단의 주 근거는 held-out loss입니다.

## Reload Check

학습 process를 종료한 뒤 저장된 adapter를 새 Python process에서 다시 적재했습니다.

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
  .venv/bin/python -m sft_lab.infer \
  results/trl-branch-reproduction/adapter \
  --local-files-only \
  --prompt 'Give two practical tips for debugging an out-of-memory error during LLM training.'
```

명령은 exit code 0으로 끝났고 학습 직후와 같은 두 항목의 답을 생성했습니다. 따라서 결과는 memory에 남은 adapter에만 의존하지 않으며 저장·재로딩 경로까지 동작합니다.
