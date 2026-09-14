# Scaling Estimates: 100B to 1T

아래는 **용량 계산**입니다. 실제 GPU 검증은 30B급까지이며, 100B~1T 실행 결과나 GPU 구매·배치 수량을 뜻하지 않습니다.

## Assumptions

`P`는 full FT의 전체 parameter 수, LoRA에서는 frozen base parameter 수입니다. `f=0.001`은 base 대비 추가 adapter parameter 비율(0.1%)입니다. 앞선 LoRA sweep의 **rank-local shard 기준 비율과 분모가 다릅니다.**

| 저장 항목 | 가정한 bytes/parameter |
| --- | ---: |
| BF16 weight | 2 |
| BF16 gradient | 2 (학습 대상만) |
| FP32 master weight + Adam 1차·2차 moment | 12 (학습 대상만) |

Full FT 학습 state는 `16P`, weights-only checkpoint는 `2P`, optimizer를 포함한 checkpoint는 `14P` bytes로 계산합니다. Checkpoint에는 gradient를 포함하지 않는 가정입니다.
LoRA 학습 state는 `2P + 16fP`, 배포용 adapter weight만 저장하면 `2fP`, **adapter와 Adam state를 함께 저장하면 `14fP`**입니다. LoRA checkpoint는 frozen base를 포함하지 않으므로 base model의 revision과 가중치를 별도로 보존해야 합니다.

12 bytes는 Adam moment만의 크기가 아니라 **master weight를 포함한 이 저장 방식의 가정**입니다. Framework의 gradient dtype·master copy·optimizer state 저장 정책에 따라 달라집니다.
Qwen 약 30.5B를 2 rank로 균등 분할한 optimizer 저장량은 `30.5e9 / 2 × 12` = 약 170.4 GiB/rank이고, 보고된 `offloaded_tensors` 171 GiB/rank와 가깝습니다. 이 일치가 다른 모델·framework의 모든 상태 배치를 검증하지는 않습니다.

## State and Checkpoint Capacity

전체 모델에 대한 합계이며 rank당 크기가 아닙니다. Metadata·padding·중복 저장은 제외합니다.

| 모델 규모 | Full FT 학습 state | LoRA 학습 state | Base weights만 | Full-state checkpoint | LoRA weights만 | LoRA + Adam checkpoint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 30.5B (anchor) | 0.44 TiB | 57.3 GiB | 56.8 GiB | 0.39 TiB | 0.06 GiB | 0.40 GiB |
| 100B | 1.46 TiB | 187.8 GiB | 186.3 GiB | 1.27 TiB | 0.19 GiB | 1.30 GiB |
| 500B | 7.28 TiB | 938.8 GiB | 931.3 GiB | 6.37 TiB | 0.93 GiB | 6.52 GiB |
| 1T | 14.55 TiB | 1877.5 GiB | 1862.6 GiB | 12.73 TiB | 1.86 GiB | 13.04 GiB |

1T의 배포용 LoRA weight는 약 **1.86 GiB**, optimizer를 포함한 LoRA checkpoint는 **13.04 GiB**입니다. 이전 표의 13.04 GiB는 adapter weight만의 크기가 아니었습니다. Full-state 12.73 TiB와 비교할 때도 같은 저장 범위를 사용합니다.

## Ideal Sharding Lower Bound

Full FT의 `16P` bytes를 모든 GPU에 균등하게 나누고, offload 없이 전체 예산을 학습 state에 사용할 수 있다고 가정합니다. 표는 `ceil(16P / device budget)`인 **정수 용량 하한**입니다. DDP는 상태를 복제하므로 이 나눗셈을 적용할 수 없고, 실제 sharding에도 비분할·중복 항목과 topology 제약이 있습니다.

| 모델 규모 | 119 GiB/device 예산 | 80 GB/device 예산 | 141 GB/device 예산 |
| --- | ---: | ---: | ---: |
| 30.5B | 4 | 7 | 4 |
| 100B | 13 | 20 | 12 |
| 500B | 63 | 100 | 57 |
| 1T | 126 | 200 | 114 |

## Limits of the Estimate

Activation, MoE routing·communication buffer, workspace, CUDA context와 allocator 여유분은 제외했습니다. BF16 gradient 대신 FP32 gradient를 유지하면 full FT에서 parameter당 2 bytes가 추가됩니다. Spark의 119 GiB도 CPU와 공유하는 예산이므로 실제 GPU 전용 여유 공간과 같지 않습니다.

Sequence length 효과는 별도입니다. Qwen의 4096→8192 CUDA peak는 `local`에서 +45.8%, `transformer_engine`에서 +12.6%였습니다. 이 관측값을 모든 규모에 일정 배수로 적용하지 않습니다. 새 모델에서는 실제 target module·optimizer·attention 구현과 loading peak를 확인한 뒤 [메모리 추정기](running.md#estimate-memory-before-running)와 pilot으로 실행 가능성을 판단합니다.

2026-09-15 Qwen 30B Megatron full-SFT SGD는 추정상 96.6 GiB/rank로 119 GiB 예산 안이었지만, 실제 2노드 pilot은 FP32 main gradient 구성 중 global OOM으로 종료됐습니다.
따라서 budget 이내라는 판정은 실행 허가가 아니라 하한 점검이며, allocator·unsharded 임시 상태까지 포함한 bounded pilot을 통과해야 합니다.
