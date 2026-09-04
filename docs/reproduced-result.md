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

## Dataset Terms

- `train_sft`: 모델이 답변을 보고 파라미터를 조정하는 연습용 대화 묶음입니다.
- `test_sft`: 학습에는 사용하지 않고, 학습 전후 결과를 비교하는 시험용 대화 묶음입니다.

`test_sft`를 따로 두는 이유는 모델이 연습 문제만 외운 것인지, 학습에 쓰지 않은 대화에도 더 잘 답하는지 구분하기 위해서입니다.

`scripts/setup.sh`로 준비한 로컬 dataset과 model cache를 재사용해 실행한 명령은 다음과 같습니다. `run_smoke.sh`는 이 dataset 옵션을 자동으로 전달합니다.

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

## Console Walkthrough

아래는 위 명령을 실제로 실행했을 때의 핵심 출력만 남긴 예시입니다. progress bar의 중간 갱신은 생략했습니다.

### 1. Model Loading

~~~text
[INFO] Stage 1/6: Loading local dataset from data/ultrachat_200k/data
[INFO] Stage 1/6: Dataset ready: train=128, evaluation=16
[INFO] Stage 2/6: Loading tokenizer and 4-bit Qwen base model
Loading checkpoint shards: 100%|██████████| 8/8 [00:09<00:00, 1.23s/it]
~~~

터미널에서는 대괄호 안의 [INFO]가 청록색, [WARNING]이 노란색, [ERROR]가 빨간색으로 표시됩니다. NO_COLOR를 설정하거나 출력을 파일로 보낼 때는 색상 코드 없이 출력됩니다. Stage 1/6은 local parquet를 읽는 단계이고, Stage 2/6은 Qwen base model의 8개 weight shard를 GPU에 올리는 단계입니다. 이 단계에서는 GPU memory가 크게 증가하지만 아직 학습은 시작하지 않았습니다.

### 2. Held-out Evaluation Before Training

~~~text
[INFO] Stage 3/6: Evaluating the base model on held-out conversations
0%|          | 0/15 [00:00<?, ?it/s]
100%|██████████| 15/15 [00:04<00:00, 3.43it/s]
~~~

학습 전에 <code>test_sft</code>의 유효한 15개 대화로 base model을 평가합니다. 요청은 16개였지만, assistant label이 없는 대화는 제외되어 실제 평가는 15개가 사용됐습니다.

### 3. Training Logs

~~~text
[INFO] Stage 4/6: Training the QLoRA adapter for 20 optimizer steps
{'loss': 1.541, 'grad_norm': 0.7578, 'learning_rate': 0.0, 'epoch': 0.07}
{'loss': 0.9418, 'grad_norm': 0.3242, 'learning_rate': 0.000194, 'epoch': 0.35}
{'loss': 0.8363, 'grad_norm': 0.2002, 'learning_rate': 0.00000152, 'epoch': 1.35}
{'train_runtime': 158.9003, 'train_samples_per_second': 1.007, 'train_steps_per_second': 0.126, 'train_loss': 0.9498}
~~~

각 <code>loss</code>는 서로 다른 training batch의 값이라 중간에 오르내릴 수 있습니다. 여기서는 <code>nan</code> 없이 끝났고 gradient norm도 유한하게 유지됐으므로 학습은 수치적으로 안정적이었습니다. 마지막 <code>train_loss</code>는 training data의 평균값이며, 학습 품질의 최종 판단에는 아래 held-out 평가를 사용합니다.

### 4. Final Summary

~~~text
[INFO] Stage 5/6: Evaluating the tuned model and generating comparison responses
[INFO] Stage 6/6: Saving adapter, checkpoint, and summary under results/trl-branch-reproduction
[INFO] Complete: Summary written to results/trl-branch-reproduction/summary.json
{
  "effective_train_samples": 114,
  "effective_eval_samples": 15,
  "base_eval_loss": 1.1978040933609009,
  "tuned_eval_loss": 0.8403604030609131,
  "loss_change_percent": -29.841581964964053
}
~~~

마지막 JSON object는 <code>summary.json</code>에도 저장됩니다. <code>tuned_eval_loss</code>가 <code>base_eval_loss</code>보다 낮고 변화율이 음수이므로, 학습에 쓰지 않은 이 15개 대화에서는 adapter 적용 뒤 예측이 개선됐다고 해석합니다.

## Metrics

| Metric | Base | Tuned | Change |
| --- | ---: | ---: | ---: |
| Held-out assistant-token loss | 1.1978 | 0.8404 | -29.84% |
| Held-out perplexity | 3.3128 | 2.3172 | -30.05% |

| Performance metric | Result |
| --- | ---: |
| Mean training loss | 0.9498 |
| Training time | 159.15s |
| Optimizer throughput | 0.126 steps/s |
| Micro-sample throughput | 1.007 samples/s |
| Peak allocated GPU memory | 14.50GiB |

`train_sft`의 첫 logged loss는 1.5410이었고 마지막 logged loss는 0.8363이었습니다. 개별 batch 난이도가 달라 중간 값은 단조 감소하지 않았지만, 별도 `test_sft` subset의 loss가 29.84% 감소해 adapter가 단순히 training batch를 통과한 것보다 강한 optimization 증거를 보였습니다.

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
