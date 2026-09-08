# Experiment Result

이 문서는 이 브랜치에서 실행한 Qwen2.5-14B-Instruct QLoRA 결과를 기록합니다.
전체 산출물은 Git에서 제외하고 실험에 필요한 명령과 요약만 남깁니다.

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

실험에 사용한 명령은 다음과 같습니다.

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
  --output-dir results/trl-branch-experiment
```

## Console Walkthrough

아래는 위 명령을 실제로 실행했을 때의 핵심 출력만 남긴 예시입니다.
progress bar의 중간 갱신은 생략했습니다.

### 1. Model Loading

~~~text
[INFO] Stage 1/6: Loading local dataset from data/ultrachat_200k/data
[INFO] Stage 1/6: Dataset ready: train=128, evaluation=16
[INFO] Stage 2/6: Loading tokenizer and 4-bit Qwen base model
Loading checkpoint shards: 100%|██████████| 8/8 [00:09<00:00, 1.23s/it]
~~~

Stage 1/6에서는 local parquet를 읽고, Stage 2/6에서는 Qwen base model을 GPU에 불러옵니다.

### 2. Held-out Evaluation Before Training

~~~text
[INFO] Stage 3/6: Evaluating the base model on held-out conversations
0%|          | 0/15 [00:00<?, ?it/s]
100%|██████████| 15/15 [00:04<00:00, 3.43it/s]
~~~

학습 전에 <code>test_sft</code>의 유효한 15개 대화로 base model을 평가합니다.
요청은 16개였지만, assistant label이 없는 대화는 제외되어 실제 평가는 15개가 사용됐습니다.

### 3. Training Logs

~~~text
[INFO] Stage 4/6: Training the QLoRA adapter for 20 optimizer steps
{'loss': 1.541, 'grad_norm': 0.7578, 'learning_rate': 0.0, 'epoch': 0.07}
{'loss': 0.9418, 'grad_norm': 0.3242, 'learning_rate': 0.000194, 'epoch': 0.35}
{'loss': 0.8363, 'grad_norm': 0.2002, 'learning_rate': 0.00000152, 'epoch': 1.35}
{'train_runtime': 158.9003, 'train_samples_per_second': 1.007, 'train_steps_per_second': 0.126, 'train_loss': 0.9498}
~~~

각 <code>loss</code>는 서로 다른 training batch의 값이라 중간에 오르내릴 수 있습니다.
여기서는 <code>nan</code> 없이 끝났고 gradient norm도 유한하게 유지됐으므로 학습은 수치적으로 안정적이었습니다.
마지막 <code>train_loss</code>는 training data의 평균값이며, 학습 품질의 최종 판단에는 아래 held-out 평가를 사용합니다.

### 4. Final Summary

~~~text
[INFO] Stage 5/6: Evaluating the tuned model and generating comparison responses
[INFO] Stage 6/6: Saving adapter, checkpoint, and summary under results/trl-branch-experiment
[INFO] Complete: Summary written to results/trl-branch-experiment/summary.json
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

`train_sft`의 첫 번째 loss는 1.5410, 마지막 loss는 0.8363이었습니다.
batch별 loss는 단조 감소하지 않았지만, 별도 `test_sft` subset의 loss가 29.84% 감소해 학습에 사용하지 않은 data에서도 optimization이 진행됐음을 확인했습니다.

평가는 held-out conversation 15개만 사용했으므로 이 결과를 일반적인 품질 향상이나 benchmark 성능으로 해석할 수는 없습니다.

## Generation Check

학습 전후의 답변 형식을 비교하기 위해 두 모델에 같은 질문을 입력했습니다.
답변 길이는 최대 96 token으로 제한했습니다.

- `Explain gradient accumulation in three concise bullet points.`는 gradient accumulation을 세 항목으로 설명하라는 질문입니다. 두 모델 모두 세 항목을 만들었지만, base model은 마지막 항목을 끝내기 전에 길이 제한에 도달했습니다. tuned model은 세 항목을 모두 끝까지 작성했습니다.
- `Give two practical tips for debugging an out-of-memory error during LLM training.`은 GPU memory 부족 문제를 해결할 방법 두 가지를 묻는 질문입니다. base model은 두 번째 방법을 쓰기 전에 길이 제한에 도달했습니다. tuned model은 batch size 줄이기와 gradient checkpointing 사용하기를 모두 제시했습니다.

이 두 결과에서는 tuned model이 요구된 항목 수에 맞춰 더 짧고 완결된 답변을 생성했습니다.
다만 질문이 두 개뿐이므로 일반적인 품질 향상을 입증하지는 않으며, 학습 효과는 위의 held-out loss를 기준으로 판단합니다.

## Reload Check

학습 프로그램을 종료한 뒤 새 Python 프로세스에서 원본 Qwen 모델과 저장한 adapter를 다시 불러와 결합했습니다.
다음 명령은 결합된 모델에 학습 때 사용한 질문을 다시 입력합니다.

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
  .venv/bin/python -m sft_lab.infer \
  results/trl-branch-experiment/adapter \
  --local-files-only \
  --prompt 'Give two practical tips for debugging an out-of-memory error during LLM training.'
```

명령은 오류 없이 끝났고 학습 직후와 같은 내용의 답변을 생성했습니다.
이를 통해 저장한 adapter를 원본 모델에 다시 붙여 추론할 수 있음을 확인했습니다.
