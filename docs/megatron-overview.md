# Megatron-LM, Megatron Core, and Megatron Bridge Overview

이 저장소의 학습 코드는 Megatron Bridge를 통해 Megatron Core 기반 모델을 사용한다. 아래 구분을 먼저 잡으면 스크립트의 목적이 분명해진다.

- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)은 대규모 Transformer 학습을 위한 참조 애플리케이션이다. Megatron Core와 실행 스크립트를 함께 제공한다.
- [Megatron Core](https://docs.nvidia.com/megatron-core/developer-guide/latest/)는 Transformer 블록과 병렬화 전략을 조합하는 라이브러리다.
- [Megatron Bridge](https://docs.nvidia.com/nemo/megatron-bridge/latest/)는 Hugging Face 체크포인트와 Megatron 형식 사이의 변환, 모델별 recipe, 학습 진입점을 제공하는 연결 계층이다. 이 실습의 Qwen2.5-7B LoRA 설정도 Bridge recipe를 사용한다.

## How This Experiment Uses the Stack

~~~text
Hugging Face Qwen2.5-7B checkpoint
        ↓ AutoBridge
Megatron Bridge recipe (LoRA)
        ↓
Megatron Core model and parallelism settings
~~~

실제 SFT 실행은 기본값으로 TP=1, PP=1, CP=1, DP=1인 단일 GPU 구성이다. 즉, 아래 실습은 분산 학습 성능이나 통신 동작을 검증하지 않는다.

## Parallelism Terms

| 방식 | 나누는 대상 | 주로 해결하는 문제 |
| --- | --- | --- |
| TP (Tensor Parallelism) | 한 레이어의 텐서 연산 | 한 GPU에 레이어 연산/가중치를 모두 두기 어려운 경우 |
| PP (Pipeline Parallelism) | 모델의 레이어 구간 | 모델 깊이를 여러 stage에 분할 |
| DP (Data Parallelism) | 데이터 배치 | 같은 모델 복제본으로 처리량 확장 |
| CP (Context Parallelism) | 한 샘플의 시퀀스 길이 | 긴 context의 activation 및 attention 메모리 부담 완화 |

CP는 한 시퀀스의 token 구간을 CP rank에 나누어 둔다. attention을 계산하려면 다른 구간의 key/value 정보도 필요하므로, 실제 구현에서는 CP group 내부의 통신이 필요하다. 따라서 CP는 단순히 배치를 나누는 DP와 다르며, 긴 context에 특히 의미가 있다. 실습의 Qwen2.5-7B SFT는 sequence length 512를 기본으로 하므로 CP=1을 사용한다.

일반적인 논리적 관계는 다음과 같다.

~~~text
world_size = TP × PP × CP × DP
~~~

실제 가능한 조합과 통신 방식은 모델, sequence length, GPU 메모리, 네트워크에 따라 달라진다. [Megatron Core 병렬화 문서](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/context_parallel.html)를 실제 설정의 기준으로 삼는다.

## CPU-only Concept Exercise

다음 명령은 GPU, CUDA, NCCL, torch.distributed를 초기화하지 않는다.

~~~bash
./scripts/run_megatron_practice.sh
~~~

이 스크립트는 두 작업을 한다.

1. Qwen2.5-7B LoRA Bridge recipe의 기본 병렬화 값을 출력한다. 체크포인트도 내려받지 않는다.
2. TP=2, PP=2, CP=2, DP=2, 총 16개 논리 rank의 group 배치를 JSON으로 출력한다.

직접 조합을 바꿔 보려면 다음과 같이 실행한다.

~~~bash
WORLD_SIZE=16 TP_SIZE=2 PP_SIZE=2 CP_SIZE=2 ./scripts/simulate_parallelism.sh
~~~

출력되는 group은 개념 설명용이다. 실제 멀티 GPU 또는 멀티 노드 실행에는 launcher, process group 초기화, GPU 자원, 네트워크 설정이 별도로 필요하다.
