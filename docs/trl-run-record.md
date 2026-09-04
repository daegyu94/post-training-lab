# TRL Run Record

이 문서는 이 repository의 TRL workflow로 실행한 Qwen2.5-14B-Instruct QLoRA SFT 결과를 기록합니다. 전체 실행 산출물은 Git에서 제외하고, 실행 configuration과 검증에 필요한 작은 summary만 남깁니다.

## Environment

| Item | Value |
| --- | --- |
| Date | 2026-09-04 |
| GPU | NVIDIA RTX PRO 4000 Blackwell 24GiB 1장 |
| Python | 3.12.3 |
| PyTorch | 2.11.0+cu128 |
| TRL | 1.12.0 |
| CUDA runtime | 12.8 |

## Configuration

| Item | Value |
| --- | --- |
| Model | `Qwen/Qwen2.5-14B-Instruct` |
| Dataset | `HuggingFaceH4/ultrachat_200k` |
| Train / evaluation subset | 128 / 15 conversations |
| Quantization | NF4 4-bit, double quantization, BF16 compute |
| LoRA | rank 16, alpha 32, dropout 0.05 |
| Sequence length | 512 |
| Effective batch | 8 |
| Optimizer steps | 20 |
| Learning rate | `2e-4` |

## Result

| Metric | Base | Tuned | Change |
| --- | ---: | ---: | ---: |
| Held-out assistant-token loss | 1.1978 | 0.8404 | -29.84% |
| Held-out perplexity | 3.3128 | 2.3172 | -30.05% |

학습 시간은 159.15초였고, peak allocated GPU memory는 14.50GiB였습니다. 저장한 PEFT adapter는 새 Python process에서 원본 Qwen model과 함께 다시 읽어 inference할 수 있음을 확인했습니다.

이 결과는 짧은 SFT workflow의 동작 기록입니다. held-out conversation 수가 적으므로 일반적인 품질 향상이나 benchmark 성능으로 해석하지 않습니다.
