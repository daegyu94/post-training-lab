# Large-Scale LLM Post-Training Lab

## Goal

이 저장소는 LLM post-training을 실제로 실행하고, 단일 GPU 기준 검증에서 multi-GPU·multi-node 확장에 필요한 구성 요소까지 순서대로 확인하기 위한 실습 공간입니다. 최종 목표는 대규모 모델과 다중 노드 GPU cluster에서 post-training workflow의 정확성, 확장성, resource bottleneck을 검증하는 것입니다.

## Framework Strategy

TRL과 Megatron-LM은 서로 다른 범위를 담당합니다.

| 프레임워크 | 주 역할 | large-scale 관점에서의 의미 |
| --- | --- | --- |
| TRL | dataset 형식, assistant-only loss, PEFT/QLoRA, held-out evaluation을 빠르게 검증 | workflow의 기준 동작과 비교 기준 확보 |
| Megatron-LM 계열 | distributed training, parallelism, distributed checkpoint, cluster 자원 사용을 검증 | 대형 model과 multi-GPU·multi-node 실행으로 확장 |

TRL에서 확인한 데이터·평가 기준을 Megatron-LM에서도 유지하면, framework 차이와 scale 차이를 분리해서 관찰할 수 있습니다.

## Scale-up Path

1. Step 1에서는 공개 dataset으로 SFT의 dataset → train → checkpoint → validation 경로를 실제 GPU에서 확인합니다.
2. 다음 단계에서는 multi-GPU와 multi-node에서 model parallelism, data parallelism, checkpoint 저장·복구, communication을 검증합니다.
3. 이후에는 대규모 model과 장시간 실행에서 storage, network, profiling, fault recovery를 함께 측정합니다.

현재 repository의 실행 가능한 범위는 Step 1이며, 이후 단계의 상세 설계나 결과를 Step 1 문서에 섞지 않습니다.

## Shared Success Criteria

- training 전후 같은 held-out dataset에서 loss와 perplexity를 비교할 수 있어야 합니다.
- 저장한 adapter 또는 checkpoint를 새 프로세스에서 다시 읽어 평가할 수 있어야 합니다.
- 실행 configuration, log, resource 사용량을 남겨 원인을 추적할 수 있어야 합니다.
- 대규모 실행으로 확장할 때 framework, model, parallelism, storage 경로가 명시적으로 분리되어야 합니다.
