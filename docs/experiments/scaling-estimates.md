# Scaling Estimates: 100B to 1T

이 문서는 동일한 dtype과 optimizer 가정을 100B~1T 모델에 적용해 학습 state와 checkpoint에 필요한 **용량의 규모**를 계산합니다.
실제 GPU 검증 범위는 30B급까지이므로, 100B~1T 수치는 실행 결과나 GPU 구매·배치 수량이 아니라 초기 계획을 위한 추정값입니다.

> **핵심**: full FT는 parameter당 16 bytes, LoRA는 base 2 bytes + adapter 몫만 듭니다. 1T에서 full-state checkpoint 12.73 TiB 대 LoRA+Adam checkpoint 13.04 GiB — 약 1,000배 차이가 나며, 대신 LoRA checkpoint는 base를 포함하지 않으므로 base revision을 따로 보존해야 합니다.

## Assumptions

두 변수만 사용합니다.

- `P`: full FT에서는 전체 parameter 수, LoRA에서는 frozen base parameter 수
- `f=0.001`: base 대비 추가 adapter parameter 비율(0.1%)

여기서 `f`는 전체 모델을 기준으로 한 계산용 가정입니다.
앞선 LoRA sweep의 **rank-local shard 기준 비율과는 분모가 다릅니다.**

Parameter 하나가 차지하는 용량은 다음과 같이 가정합니다.

| 저장 항목 | 가정한 bytes/parameter |
| --- | ---: |
| BF16 weight | 2 |
| BF16 gradient | 2 (학습 대상만) |
| FP32 master weight + Adam 1차·2차 moment | 12 (학습 대상만) |

이 값을 Full FT와 LoRA에 적용하면 다음 공식이 됩니다.

| 방식 | 학습 중 model state | 배포용 weight | 재시작용 checkpoint |
| --- | ---: | ---: | ---: |
| Full FT | `16P` | `2P` | `14P` |
| LoRA | `2P + 16fP` | `2fP` | `14fP` |

학습 중 model state에는 weight, 학습 대상의 gradient와 optimizer state가 포함됩니다.
Checkpoint에는 gradient를 저장하지 않으며, 재시작용 checkpoint에는 weight와 optimizer state를 포함합니다.
LoRA checkpoint는 frozen base를 포함하지 않으므로 base model의 revision과 가중치를 별도로 보존해야 합니다.

12 bytes는 Adam moment만의 크기가 아니라 **master weight를 포함한 이 저장 방식의 가정**입니다. Framework의 gradient dtype·master copy·optimizer state 저장 정책에 따라 달라집니다.
Qwen 약 30.5B를 2 rank로 균등 분할하면 이 가정의 optimizer 저장량은 약 170.4 GiB/rank이며, 보고된 `offloaded_tensors` 171 GiB/rank와 가깝습니다.
이는 30B 측정의 크기를 교차 확인할 뿐, 다른 모델·framework의 상태 배치까지 검증하지는 않습니다.

## State and Checkpoint Capacity

> **표에서 바로 읽을 결론**: Full FT는 100B부터 학습 state가 1.46 TiB, 재시작용 checkpoint가 1.27 TiB입니다.
> 1T에서는 각각 14.55 TiB와 12.73 TiB까지 늘어납니다.
> LoRA는 checkpoint를 크게 줄이지만, 학습할 때는 frozen base weight를 여전히 올려야 합니다.

아래 값은 모든 rank에 분산하기 전 **전체 모델의 합계**이며, rank당 크기나 실제 peak memory가 아닙니다.
Metadata·padding·중복 저장은 제외합니다.

| 모델 규모 | Full FT 학습 state | LoRA 학습 state | Base weights만 | Full-state checkpoint | LoRA weights만 | LoRA + Adam checkpoint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 30.5B (anchor) | 0.44 TiB | 57.3 GiB | 56.8 GiB | 0.39 TiB | 0.06 GiB | 0.40 GiB |
| 100B | 1.46 TiB | 187.8 GiB | 186.3 GiB | 1.27 TiB | 0.19 GiB | 1.30 GiB |
| 500B | 7.28 TiB | 938.8 GiB | 931.3 GiB | 6.37 TiB | 0.93 GiB | 6.52 GiB |
| 1T | 14.55 TiB | 1877.5 GiB | 1862.6 GiB | 12.73 TiB | 1.86 GiB | 13.04 GiB |

1T의 배포용 LoRA weight는 약 **1.86 GiB**, 재시작용 LoRA+Adam checkpoint는 **13.04 GiB**입니다.
Full FT의 재시작용 checkpoint 12.73 TiB와 비교하면 약 1,000배 작지만, 별도로 보존해야 하는 base weight 1.82 TiB는 이 값에 포함되지 않습니다.

## Ideal Sharding Lower Bound

> **표를 읽는 법**: 100B Full FT의 `119 GiB/device = 13`은 완벽하게 분할해도 최소 13개 device가 필요하다는 뜻입니다.
> 13개로 실제 실행할 수 있다거나 13개를 구매해야 한다는 뜻은 아닙니다.

표는 Full FT 학습 state `16P`를 모든 device에 똑같이 나누고, 각 device의 전체 예산을 model state에만 쓸 수 있다고 가정한 **이론적 최소 device 수**입니다.
계산식은 `ceil(16P / device budget)`입니다.

| 모델 규모 | 119 GiB/device 예산 | 80 GB/device 예산 | 141 GB/device 예산 |
| --- | ---: | ---: | ---: |
| 30.5B | 4 | 7 | 4 |
| 100B | 13 | 20 | 12 |
| 500B | 63 | 100 | 57 |
| 1T | 126 | 200 | 114 |

실제 실행에는 activation·workspace·통신 buffer와 분할되지 않는 상태가 더 필요하므로 device 수는 이 하한보다 많아집니다.
DDP는 각 device에 상태를 복제하므로 이 계산을 적용할 수 없고, sharding 방식도 topology와 분할 단위에 따라 효율이 달라집니다.

## Limits of the Estimate

이 계산은 model state의 크기를 비교하기 위한 하한이며, 실제 실행 가능성에는 다음 항목이 더 필요합니다.

| 계산에서 빠지거나 달라지는 항목 | 실제 영향 |
| --- | --- |
| Activation, MoE routing·communication buffer, framework workspace | Sequence length, batch와 구현에 따라 추가 memory가 필요함 |
| CUDA context, allocator 단편화와 임시 buffer | 초기화나 첫 step의 peak가 표보다 커질 수 있음 |
| Gradient dtype, master copy와 optimizer 저장 정책 | Framework 설정에 따라 bytes/parameter가 달라짐 |
| Device에 표시된 전체 memory | Spark의 119 GiB처럼 CPU와 공유하면 GPU 전용 예산으로 모두 쓸 수 없음 |

Sequence length의 영향도 일정하지 않습니다.
Qwen의 4096→8192 CUDA peak 증가는 `local` attention에서 +45.8%, `transformer_engine`에서 +12.6%였으므로 한 측정값을 모든 모델 규모에 같은 배수로 적용할 수 없습니다.

실제로 30B Qwen Megatron Full FT는 SGD 기준 추정 90.0 GiB/rank로 119 GiB 예산 안이었지만, FP32 main gradient 구성 중 첫 optimizer step 전에 OOM이 발생했습니다.
따라서 표의 `FITS`나 최소 device 수는 실행 여부가 아니라 후보 구성을 거르는 기준으로만 사용합니다.
실행 가능성은 실제 target module·optimizer·attention·sharding 설정으로 [메모리를 다시 추정](running.md#estimate-memory-before-running)하고, 초기화부터 checkpoint까지 포함한 작은 pilot으로 확인해야 합니다.
