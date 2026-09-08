# PoC Setup Guide

이 문서는 Post-Training Lab의 두 PoC hardware setup, 실험 범위와 현재 검증 상태를 정의하는 main branch의 기준 문서입니다. Main은 공통 setup 설명과 검증 경계를 관리하고, backend별 실행 코드와 command는 [`megatron`](https://github.com/daegyu94/post-training-lab/tree/megatron) 및 [`trl`](https://github.com/daegyu94/post-training-lab/tree/trl) 브랜치가 관리합니다.

## Status boundary

문서에서 `verified`는 명시한 환경에서 실제 확인한 사실, `target`은 구성하려는 목표, `planned`는 아직 실행 증거가 없는 작업을 뜻합니다. 아래 계획은 실행 로그나 NCCL·training 결과가 추가되기 전에는 `runnable`로 표시하지 않습니다.

현재 확인 시점은 2026-09-08입니다. SSH로 `spark1`과 `spark2`에 접속해 각 노드가 GB10임을 확인했고, 각 노드에서 총 메모리 약 119 GiB를 확인했습니다. 두 노드 NCCL 통신 correctness, 소형 model feature 경로와 두 30B model의 Megatron·TRL one-step distributed LoRA integration을 확인했지만, 장기 학습과 NCCL throughput benchmark는 아직 검증하지 않았습니다.

| Item | Status | Evidence or boundary |
| --- | --- | --- |
| `spark1`, `spark2` SSH reachability | `verified` | 각 노드 SSH 접속 및 GB10 확인 |
| 각 Spark 노드 memory | `verified` | 총 약 119 GiB, 가용 약 115 GiB 확인 |
| 노드 간 100Gbps RoCE link configuration | `verified` | `/sys/class/infiniband/rocep1s0f0/ports/1/rate`에서 `100 Gb/sec (4X EDR)`와 data-IP ping을 확인했지만 application throughput 측정값은 아님 |
| Host-memory RoCE throughput | `verified` | `ib_write_bw`, 65,536-byte message, 1 QP, 5초에서 92.57 Gbps; GPU throughput 아님 |
| NCCL over RoCE correctness | `verified` | 양 노드 GPU all-reduce와 all-to-all 결과 확인, NCCL 로그 `NET/IB` RoCE 경로; throughput 미측정 |
| Multi-node Megatron 30B SFT | `verified` | Qwen3-30B-A3B + No Robots와 GLM-4.7-Flash + Self-OSS의 EP=2 BF16 LoRA 1-step, finite forward/backward, optimizer, validation, sync DCP 저장과 양 rank exit 0 확인; 장기 수렴·품질·성능 evidence 아님 |
| TRL local integration smoke | `verified` | Spark2, Qwen2.5-0.5B-Instruct, No Robots 2 train/1 eval, LoRA AdamW 2 steps, nonzero parameter update 및 adapter reload 평가 일치 |
| TRL two-node integration smoke | `verified` | DDP: Qwen2.5-0.5B-Instruct, No Robots 4 train/2 eval, LoRA AdamW 2 steps와 adapter reload 확인. FSDP2: 같은 소형 model의 full-parameter SGD 1 step, finite train/eval, two-rank distcp shard 저장 확인. 두 결과 모두 30B evidence 아님 |
| Megatron two-node integration smoke | `verified` | Qwen2.5-0.5B-Instruct full-parameter DP=2 학습, sync/async DCP 저장, fully-reshardable DCP DP=2→TP=2 optimizer/scheduler load와 iteration 5→8 후속 학습 확인; TP/PP mismatch로 RNG/rerun state는 ignored, 30B evidence 아님; 상세 범위는 Megatron branch의 `docs/megatron-feature-labs.md` 참조 |
| TRL 30B distributed training | `verified` | 두 30B model의 BF16 LoRA two-node DDP 1-step, finite train/eval loss, sampled nonzero update, adapter 저장과 양 rank exit 0 확인; FSDP2/DeepSpeed 30B와 장기 학습은 미검증 |

## Setup 1: single RTX PRO 4000 Blackwell

Setup 1은 RTX PRO 4000 Blackwell 24 GiB 한 장을 사용하는 single-GPU 경로입니다. Megatron에서는 `Qwen/Qwen2.5-7B-Instruct` LoRA SFT를, TRL에서는 `Qwen/Qwen2.5-14B-Instruct` NF4 QLoRA를 대상으로 합니다.

현재 `trl` 브랜치는 이 GPU에서 Qwen2.5-14B QLoRA의 base evaluation → 짧은 training → adapter reload → held-out evaluation 흐름을 기록한 경로입니다. `megatron` 브랜치는 Qwen2.5-7B LoRA의 single-GPU workflow를 제공하며, 현재 문서와 구현은 multi-GPU·multi-node 성능을 검증한 것으로 해석하지 않습니다.

자세한 설치, local model·dataset 준비, 실행 command, 결과 형식은 [TRL branch README](https://github.com/daegyu94/post-training-lab/tree/trl)와 [Megatron branch README](https://github.com/daegyu94/post-training-lab/tree/megatron)를 따릅니다. 이 main 문서는 해당 command를 복제하지 않고 setup의 공통 경계만 설명합니다.

## Setup 2: two-node Spark cluster

Setup 2는 `spark1`과 `spark2` 두 노드로 구성된 cluster입니다. 두 노드는 각각 NVIDIA DGX Spark GB10이며, 노드 간 연결은 100Gbps RoCE입니다. Link rate, data-IP ping, host-memory RDMA throughput과 NCCL collective correctness를 확인했습니다. NCCL bandwidth와 학습 통신 overlap의 효과는 별도 측정 대상입니다.

[NVIDIA Spark CUDA 포팅 가이드](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html)에 따르면 Spark는 GPUDirect RDMA를 지원하지 않습니다. 현재 검증된 경로는 host-buffer 기반 RoCE이며, NCCL의 GDR 비활성화 로그를 설정 오류나 Socket transport 사용의 증거로 해석하지 않습니다.

GB10의 CPU·GPU memory가 unified memory 방식으로 보이더라도 `spark1`과 `spark2`의 memory가 자동으로 하나의 memory pool이 되지는 않습니다. 두 노드를 함께 사용하려면 distributed launcher, process group, checkpoint 전략과 model parallelism 또는 data parallelism 설정을 명시해야 합니다.

Cluster에서의 target model은 [`zai-org/GLM-4.7-Flash`](https://huggingface.co/zai-org/GLM-4.7-Flash) (약 30B MoE)와 [`Qwen/Qwen3-30B-A3B`](https://huggingface.co/Qwen/Qwen3-30B-A3B)입니다. Megatron target은 distributed SFT이고, TRL target은 local 또는 distributed SFT입니다. LoRA는 메모리 예산에 맞춰 선택할 수 있는 방식이지 feature 실습의 필수 조건은 아닙니다. 두 target의 one-step distributed LoRA integration은 실행했지만 full-parameter 30B, 장기 수렴이나 일반적인 성능을 검증한 것은 아닙니다.

두 target model의 전체 snapshot은 `spark1`과 `spark2` 각각의 `/home/spark/.cache/huggingface`에 materialize합니다. 분산 실행 시 각 노드는 자신의 local cache에서 model weight를 읽으며, NFS를 model-weight runtime source 또는 authoritative model copy로 사용하지 않습니다. Dataset, checkpoint, result와 같은 공유가 필요한 artifact만 `/home/spark/shared` 아래 NFS 경로에 둘 수 있습니다. 실행 기록에는 Hugging Face model identifier와 revision, 각 host, 사용한 cache policy를 남겨 local snapshot의 provenance를 확인할 수 있게 합니다.

Full-parameter training은 optimizer state, gradient/master-weight dtype과 sharding을 포함해 메모리 적합성을 먼저 계산합니다. 표준 Megatron BF16/FP32 Adam 구성의 30B 상태는 두 노드 용량을 초과하므로 그대로 실행하지 않습니다. Feature A/B는 메모리에 맞는 full-training 모델로도 수행할 수 있지만, 두 30B 모델의 실행 목표를 대체하지 않습니다. 초기 실행은 micro batch size 1과 소수 optimizer step으로 제한하고, sequence length는 실제 sample token 길이와 supervised-token 보존 여부를 확인해 정합니다.

## Common dataset and reproducibility

공개 SFT 데이터는 UltraChat 외에 No Robots(자연어 instruction), BigCode Self-OSS(코딩 instruction), xLAM(JSON 함수 호출 응답) 준비 경로를 제공합니다. Source schema, license/access, canonical `messages` 변환과 지원 범위는 각 backend의 `docs/public-datasets.md`를 따릅니다. 새로운 준비 경로는 source train split에서 중복 없는 deterministic validation holdout을 만들며, benchmark/test split을 읽지 않습니다. 실제 No Robots와 Self-OSS 각각 8 train/2 validation 준비를 확인했고, xLAM은 gated access로 fixture 검증에 한정됩니다. Coding instruction이나 JSON 함수 호출은 실제 agent rollout과 구분합니다.

각 run은 dataset revision을 명시적으로 pin하고 seed `42`를 사용합니다. Model identifier와 revision, dataset identifier와 revision, tokenizer·recipe, split·subset, seed, hardware, launcher, git commit과 model snapshot이 materialize된 host-local cache policy를 함께 기록해야 서로 다른 실행을 비교할 수 있습니다. 원본 dataset, model weight, checkpoint와 profiler trace는 Git에 저장하지 않습니다.

## Execution order and evidence

Setup 1은 x86 workstation에 맞는 software stack과 kernel smoke check에서 시작하고, Setup 2는 두 ARM64 node에 호환되는 software stack과 kernel smoke check에서 시작합니다. Cluster setup은 여기에 NCCL over RoCE communication test를 먼저 통과해야 하며, 그 뒤 model·dataset prepare, base held-out evaluation, 선택한 full/LoRA training, 저장 checkpoint 또는 adapter reload, 동일한 held-out evaluation 순서로 진행합니다.

각 단계의 log에는 command, environment, host·GPU 목록, model·dataset revision, git commit, seed, configuration, 시작·종료 시각과 실패 원인을 남깁니다. Base와 tuned 결과는 같은 held-out split과 generation/evaluation 조건으로 비교하고, 짧은 five-step loss 변화나 reload 성공을 일반적인 model quality·benchmark·cluster throughput 향상으로 확대 해석하지 않습니다.

실행 증거가 없는 항목은 이 문서에서 목표로만 남깁니다. 실제 training과 NCCL 결과가 확보되면 backend branch의 실행 문서와 `profiling` branch의 metric contract를 갱신한 뒤, 이 표의 status와 evidence를 함께 갱신합니다.

## Backend ownership

| Concern | Authoritative location | Current boundary |
| --- | --- | --- |
| 공통 PoC hardware·model·dataset 범위 | `main`, this document | setup 설명과 검증 상태 |
| Megatron model conversion·SFT and feature workflow | [`megatron`](https://github.com/daegyu94/post-training-lab/tree/megatron) | Qwen2.5 setup1, small-model full-parameter feature A/B와 두 30B target의 two-node LoRA one-step path verified; 세부 경계는 branch docs 참조 |
| TRL QLoRA/SFT workflow | [`trl`](https://github.com/daegyu94/post-training-lab/tree/trl) | Qwen2.5 setup1, small-model DDP/FSDP2와 두 30B target의 two-node DDP LoRA one-step path verified; 30B FSDP2/DeepSpeed는 planned |
| Resource metrics와 benchmark 기록 | [`profiling`](https://github.com/daegyu94/post-training-lab/tree/profiling) | throughput·memory·network 측정 기준과 결과 |
