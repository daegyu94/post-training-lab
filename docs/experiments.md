# Experiments

Experiment 파일은 모델·데이터 revision과 학습 조건을, setup은 실행할 노드와 경로를 선언합니다.
공통 runner가 둘을 결합하며 `--execute`가 있을 때만 SSH 학습을 시작합니다.
첫 실행은 [Getting Started](getting-started.md), 설정 책임은 [Architecture](architecture.md)를 따릅니다.

## What These Experiments Ask

| 항목 | 내용 |
| --- | --- |
| 대상 모델 | `Qwen/Qwen3-30B-A3B`, `zai-org/GLM-4.7-Flash` |
| 대상 경로 | TRL LoRA(DDP·FSDP2·DeepSpeed ZeRO-3)와 Megatron MoE |
| 하드웨어 | `spark1`·`spark2` 2노드, 노드당 1 rank, unified memory 119 GiB/노드 |
| 핵심 질문 | ① 30B SFT가 이 장치에서 실행되는가 ② checkpoint I/O와 memory가 얼마나 드는가 ③ 그 값이 100B\~1T에서 어떻게 커지는가 |
| 범위 밖 | 모델 품질, 장기 수렴, framework 간 절대 우열 |

모든 결과는 짧은 step 수의 **실행 가능성과 비용 측정**입니다. 학습 품질 지표가 아닙니다.
0.5B smoke는 runner·전처리·저장 경로를 저비용으로 점검하는 별도 tier이며 30B 결론에 섞지 않습니다.
정적 검사·CPU 테스트·dry-run·GPU 실행은 별도 증거로 구분합니다([AGENTS.md](../AGENTS.md), [판정 기준](getting-started.md#6-verify-the-result)).

## Read the Results First

질문을 먼저 고르고, 해당 결과의 조건과 한계를 함께 읽습니다.

### Feasibility and Scale

| 알고 싶은 것 | 먼저 볼 결과 | 현재까지 말할 수 있는 것 |
| --- | --- | --- |
| 30B SFT와 checkpoint 재로딩이 되는가? | [30B GPU Results](experiments/30b-results.md#30b-gpu-results) | 1-step 실행과 재로딩까지 확인. 장기 수렴·품질 검증은 아님 |
| DeepSpeed runtime offload로 30B full SFT가 가능한가? | [ZeRO-3 NVMe topology comparison](experiments/30b-results.md#ultrachat-full-sft-topology-comparison-2026-09-14) | 1노드는 OOM, 2노드는 Qwen·GLM 모두 학습·평가·native checkpoint 복구 통과 |
| 30B full SFT는 이 장치에 들어가는가? | [Full-SFT capacity](experiments/30b-results.md#full-sft-capacity) | 2노드 Qwen Megatron pilot은 추정 90.0 GiB/rank였지만 첫 step 전 OOM. 별도 single-node TRL 실험은 두 모델 모두 OOM |
| 100B\~1T에 필요한 용량은? | [Scaling Estimates](experiments/scaling-estimates.md#scaling-estimates-100b-to-1t) | 가정한 dtype·optimizer·sharding에 따른 계산. 해당 규모 GPU 실행 결과가 아님 |

### Checkpoint I/O

| 알고 싶은 것 | 먼저 볼 결과 | 현재까지 말할 수 있는 것 |
| --- | --- | --- |
| Async checkpoint가 학습을 덜 막는가? | [Distributed checkpoint write](experiments/30b-results.md#distributed-checkpoint-write) | save API blocking은 크게 줄지만, finalization 포함 완료 시간 이득은 Qwen 7.0%·GLM 19.0%. 장기 overlap은 미검증 |
| Async가 step time을 늘리지는 않는가? | [Multi-step checkpoint impact](experiments/30b-results.md#multi-step-checkpoint-impact) | checkpoint 직접 대기 23.0%·15.3% 감소, steady-step median 2.0%·0.9% 증가. End-to-end throughput은 미검증 |
| 실제 checkpoint restore가 얼마나 걸리는가? | [Single-node TRL I/O Experiment](experiments/checkpoint-io.md#single-node-trl-io-experiment) | 두 30B 모델의 LoRA r=8/16/32를 새 process에서 cold/warm 각 3회 측정 |
| LoRA를 더 많이 학습하면 checkpoint도 커지는가? | [LoRA Ratio and Checkpoint I/O](experiments/checkpoint-io.md#lora-ratio-and-checkpoint-io) | 고정된 Qwen target module에서 trainable 비율 약 10배 → checkpoint 크기 9.99배, save 시간은 1.81배 |

### Memory and Comparison Scope

| 알고 싶은 것 | 먼저 볼 결과 | 현재까지 말할 수 있는 것 |
| --- | --- | --- |
| Sequence length를 늘리면 메모리가 얼마나 늘어나는가? | [Transformer Engine memory](experiments/30b-results.md#transformer-engine-memory) | 같은 backend에서 4096→8192 allocated 증가는 Qwen 12.65%, GLM 12.75% |
| Recompute 범위를 줄이면 무엇을 내주는가? | [Qwen recompute](experiments/30b-results.md#qwen-recompute) | Megatron selective는 step time 약 27% 단축, peak allocated 35.0\~62.9% 증가. Qwen 한정 |
| Qwen 결과를 GLM에도 적용할 수 있는가? | [Qwen and GLM Comparison](experiments/30b-results.md#qwen-and-glm-comparison) | Attention backend를 맞추면 길이 증가율이 유사. GLM host memory가 더 크다는 기존 결론은 철회 |

### Metric Definitions

`rank`는 분산 학습 process 번호입니다. 이 실험은 노드당 1 rank이며, **rank별 합·최대값·반복 median은 서로 다른 집계**입니다.

#### Checkpoint write

| 지표 | 집계 | 무엇을 의미하는가 |
| --- | --- | --- |
| Save 호출 누적 시간 | rank별 성공한 `save()` 시간 합 → rank 최대값 → run median | 학습이 `save()` 안에서 직접 기다린 시간. 단일 checkpoint latency가 아니며 async background 완료 대기는 제외 |
| Blocking finalization | rank별 blocking `finalize_async_saves` 시간 합 → rank 최대값 → run median | run 끝에서 남은 async write를 기다린 시간. Save 시간과 단순 합해도 전체 wall time은 아님 |
| Checkpoint 크기 | 최신 iteration의 `.distcp` shard bytes를 rank 간 합산 | 한 checkpoint의 논리 크기. 전체 iteration·실제 device write traffic·metadata 합계는 아님 |

#### Memory

| 지표 | 집계 | 무엇을 의미하는가 |
| --- | --- | --- |
| CUDA peak allocated | PyTorch allocator가 기록한 측정 구간의 최대 allocation | 전체 device 사용량이 아니므로 측정 구간과 rank 집계가 같은 값끼리 비교 |
| Host memory pressure | 측정 시작 시 `MemAvailable` − 측정 중 최소 `MemAvailable` | 노드 전체 가용 memory 감소량. Process RSS가 아니며 다른 process·page cache의 영향 포함 |

#### Restore and Repeatability

| 지표 | 집계 | 무엇을 의미하는가 |
| --- | --- | --- |
| TRL model restore 시간 | 새 `tuned` process의 model load 시작 → checkpoint 적용 완료 | LoRA는 base+adapter 생성, ZeRO-3 full은 skeleton·engine 준비와 native checkpoint load를 포함 |
| TRL restore 유효 처리율 | (base snapshot + checkpoint logical bytes) ÷ restore 시간 | 파일 read 속도가 아니라 model reconstruction 전체의 유효 처리율 |
| rMAD | `median(abs(x - median(x))) / median(x)` | 반복 간 상대 산포. 낮을수록 결과가 모여 있으며, 10%는 추가 반복 여부를 정하는 기준일 뿐 정확성 보장은 아님 |

수치를 읽을 때 다음을 구분합니다.

- **단위**: MB/GB는 10진, MiB/GiB/TiB는 2진 단위입니다. 예를 들어 69.4 MiB는 약 72.8 MB입니다.
- **메모리**: Spark는 unified memory를 사용하므로 CUDA peak와 host pressure를 합산하지 않습니다.
- **집계**: 원본 manifest는 커밋하지 않고 표에 요약값만 남깁니다. 재집계나 오차 검증에는 해당 run의 원본이 필요합니다.

## Detailed Documents

- [Run Experiments](experiments/running.md): preset 선택, 메모리 추정, 실행 구성과 output 확인
- [30B Results](experiments/30b-results.md): 통제 실험 결과, historical GPU·메모리 기록과 Qwen/GLM 비교
- [Checkpoint and Memory I/O](experiments/checkpoint-io.md): single-node·분산 I/O 설계, 측정값과 한계
- [Scaling Estimates](experiments/scaling-estimates.md): 100B–1T state·checkpoint 용량 추정
