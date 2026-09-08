# PoC Setup Guide

이 문서는 Post-Training Lab에서 사용할 hardware setup과 현재 검증 범위를 요약합니다.
빠르게 시작하려면 아래 표에서 환경을 고른 뒤 해당 backend 브랜치의 README를 따르세요.

## Choose a Setup

| Setup | Hardware | Purpose | Backend and model |
| --- | --- | --- | --- |
| Setup 1 | RTX PRO 4000 Blackwell 24 GiB, single GPU | 한 장의 GPU에서 SFT 전체 흐름 재현 | [`trl`](https://github.com/daegyu94/post-training-lab/tree/trl): Qwen2.5-14B NF4 QLoRA; [`megatron`](https://github.com/daegyu94/post-training-lab/tree/megatron): Qwen2.5-7B LoRA |
| Setup 2 | `spark1`, `spark2`; 각 DGX Spark GB10, memory 약 119 GiB; 100Gbps RoCE | multi-node 통신, distributed SFT, checkpoint와 parallelism 검증 | `trl` 또는 `megatron`: Qwen3-30B-A3B, GLM-4.7-Flash LoRA |

`main`은 공통 환경과 검증 경계를 관리합니다.
설치, dataset 준비와 실행 command는 선택한 backend 브랜치에서 관리합니다.

## Verified Scope

`verified`는 2026-09-08에 명시된 환경과 짧은 integration run에서 직접 확인했다는 뜻입니다.
장기 학습의 수렴, 모델 품질 또는 일반적인 cluster 성능을 보장하지 않습니다.

| Area | Evidence | Boundary |
| --- | --- | --- |
| Spark nodes | `spark1`, `spark2` SSH 접속, GB10 확인, 노드별 총 memory 약 119 GiB | 두 노드의 memory는 하나의 pool이 아님 |
| RoCE | 100Gbps link와 data-IP ping 확인; `ib_write_bw` 65,536-byte, 1 QP, 5초에서 host-memory 92.57 Gbps | GPU 또는 NCCL throughput 결과가 아님 |
| NCCL | 두 GPU의 all-reduce와 all-to-all 성공, `NET/IB` RoCE 경로 확인 | correctness만 검증; bandwidth와 overlap 효과는 미측정 |
| Megatron, small model | Qwen2.5-0.5B full-parameter DP=2 학습, sync/async DCP 저장, DP=2→TP=2 optimizer/scheduler 재개 | topology 변경 시 RNG/rerun state는 보존하지 않음 |
| Megatron, 30B | Qwen3 + No Robots와 GLM + Self-OSS의 EP=2 BF16 LoRA 1-step, validation과 sync DCP 저장 | 30B checkpoint reload와 장기 학습은 미검증 |
| TRL, small model | Qwen2.5-0.5B LoRA DDP 학습과 adapter reload; FSDP2 full-parameter 1-step과 distcp shard 저장 | 소형 integration evidence이며 30B 결과가 아님 |
| TRL, 30B | Qwen3와 GLM의 BF16 LoRA two-node DDP 1-step, finite train/eval loss, adapter 저장 | 30B FSDP2, DeepSpeed와 장기 학습은 미검증 |

상세 configuration과 결과는 [`megatron` feature labs](https://github.com/daegyu94/post-training-lab/blob/megatron/docs/megatron-feature-labs.md)와 [`trl` Spark guide](https://github.com/daegyu94/post-training-lab/blob/trl/docs/spark-cluster.md)를 참조하세요.

## Setup 2 Constraints

DGX Spark는 GPUDirect RDMA를 지원하지 않으므로 현재 multi-node 경로는 host-buffer 기반 RoCE를 사용합니다.
NCCL 로그에서 GDR이 비활성화돼도 곧바로 Socket transport 또는 설정 오류를 의미하지는 않습니다.
관련 hardware 제약은 [NVIDIA DGX Spark CUDA 포팅 가이드](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html)를 참조하세요.

30B full-parameter training은 기본 경로가 아닙니다.
표준 Megatron BF16 parameter와 FP32 master weight, gradient, Adam state는 약 18 bytes/parameter로 계산되며 30B에서 약 540 GB가 필요합니다.
이는 두 Spark 노드의 총 memory보다 크므로 optimizer와 sharding을 별도로 설계하고 검증하기 전에는 실행하지 않습니다.

## Data and Storage Rules

- Model snapshot은 `spark1`과 `spark2` 각각의 `/home/spark/.cache/huggingface`에 같은 revision으로 저장합니다.
- NFS의 `/home/spark/shared`에는 dataset, checkpoint, log와 result처럼 노드 간 공유가 필요한 artifact만 저장합니다.
- Dataset은 source train split에서 deterministic validation holdout을 만들고 benchmark/test split을 학습에 사용하지 않습니다.
- 모든 run에 model·dataset revision, seed, hardware, launcher, git commit과 cache policy를 기록합니다.
- Model weight, 원본 dataset, checkpoint와 profiler trace는 Git에 저장하지 않습니다.

지원하는 공개 SFT source는 UltraChat, No Robots, BigCode Self-OSS와 xLAM입니다.
No Robots와 Self-OSS는 실제 preparation을 확인했고, gated dataset인 xLAM은 fixture 검증만 완료했습니다.
Schema와 변환 command는 각 backend의 `docs/public-datasets.md`를 따릅니다.

## Recommended Run Order

1. OS, architecture, Python과 CUDA dependency를 확인합니다.
2. Setup 2에서는 NCCL over RoCE correctness를 먼저 검증합니다.
3. Model과 dataset revision을 고정하고 local cache를 준비합니다.
4. Base evaluation, 짧은 training, checkpoint 또는 adapter reload, 동일한 held-out evaluation 순서로 실행합니다.
5. Command, host·GPU, configuration, 시작·종료 시각과 실패 원인을 기록합니다.

짧은 loss 변화나 reload 성공은 workflow가 동작한다는 evidence입니다.
이를 모델 품질, benchmark 또는 cluster throughput 향상으로 확대 해석하지 않습니다.

## Document Ownership

| Concern | Authoritative location |
| --- | --- |
| 공통 hardware, model·dataset 범위와 검증 상태 | `main`, this document |
| Megatron SFT, checkpoint와 parallelism 실습 | [`megatron`](https://github.com/daegyu94/post-training-lab/tree/megatron) |
| TRL QLoRA, DDP와 sharded backend 실습 | [`trl`](https://github.com/daegyu94/post-training-lab/tree/trl) |
| Resource metric과 benchmark 기준 | [`profiling`](https://github.com/daegyu94/post-training-lab/tree/profiling) |

새 evidence가 생기면 먼저 해당 backend 문서와 `profiling` metric contract를 갱신하고, 그다음 이 문서의 검증 표를 갱신합니다.
