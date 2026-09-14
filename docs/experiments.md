# Experiments

Experiment 파일은 모델·데이터 revision과 학습 조건을, setup은 실행할 노드와 경로를 선언합니다.
공통 runner가 둘을 결합하며 `--execute`가 있을 때만 SSH 학습을 시작합니다.
첫 실행은 [Getting Started](getting-started.md), 설정 책임은 [Architecture](architecture.md)를 따릅니다.

두 노드에서 `Qwen/Qwen3-30B-A3B`(TRL LoRA·Megatron MoE)와 `zai-org/GLM-4.7-Flash`(TRL DDP LoRA·Megatron MoE)의 SFT를 확인합니다.
30B는 메모리·통신·checkpoint·재로딩을, 0.5B smoke는 runner·전처리·저장 경로를 저비용으로 검사합니다.
정적 검사·CPU 테스트·dry-run·GPU 실행은 별도 증거로 구분합니다([AGENTS.md](../AGENTS.md), [판정 기준](getting-started.md#6-verify-the-result)).

## Read the Results First

이 문서는 **실행 가능성**, **반복 측정**, **규모 추정**을 다룹니다. 아래 표에서 질문을 고른 뒤 해당 결과의 조건과 지표를 함께 읽습니다.

| 알고 싶은 것 | 먼저 볼 결과 | 현재 확인된 범위 |
| --- | --- | --- |
| 30B SFT와 checkpoint 재로딩이 되는가? | [30B GPU Results](experiments/30b-results.md#30b-gpu-results) | 짧은 실행·재로딩 확인. 장기 수렴·모델 품질 검증은 아님 |
| 실제 checkpoint restore가 얼마나 걸리는가? | [Single-node TRL I/O Experiment](experiments/checkpoint-io.md#single-node-trl-io-experiment) | 두 30B 모델의 LoRA r=8/16/32를 새 process에서 cold/warm 각 3회 측정 |
| Async가 학습을 덜 막는가? | [Revised Distributed Write Results](experiments/checkpoint-io.md#revised-distributed-write-results-2026-09-14) | n=1에서 완료시간은 Qwen 1.301 → 1.188 s, GLM 1.223 → 0.960 s. 반복·장기 overlap 검증 전에는 speedup으로 단정하지 않음 |
| 분산 방식에 따라 CUDA 메모리가 얼마나 필요한가? | [Memory Footprint](experiments/checkpoint-io.md#memory-footprint) | Qwen TRL LoRA에서 DDP 59.8 → FSDP2 34.1 GB. Full FT + NVMe 6.4 GB는 다른 workload |
| LoRA를 더 많이 학습하면 checkpoint도 커지는가? | [LoRA Ratio and Checkpoint I/O](experiments/checkpoint-io.md#lora-ratio-and-checkpoint-io) | 고정된 Qwen target module에서 trainable 비율 약 10배 → 최신 checkpoint shard 크기 약 10배 |
| Qwen 결과를 GLM에도 적용할 수 있는가? | [Qwen and GLM Comparison](experiments/30b-results.md#qwen-and-glm-comparison) | Attention backend를 맞추면 관측 증가율이 유사함. GLM host memory가 더 크다는 기존 결론은 철회 |
| 100B~1T에 필요한 용량은? | [Scaling Estimates](experiments/scaling-estimates.md#scaling-estimates-100b-to-1t) | 가정한 dtype·optimizer·sharding에 따른 계산. 해당 규모 GPU 실행 결과가 아님 |

### Metric Definitions

`rank`는 분산 학습 process 번호입니다. 이 실험은 노드당 1 rank이며, **rank별 합·최대값·반복 median은 서로 다른 집계**입니다.

| 지표 | 계산·범위 | 해석 |
| --- | --- | --- |
| Save 호출 누적 시간 | 한 run에서 rank별 성공한 `save()` 호출 시간을 합한 뒤 rank 최대값, 이후 run 간 median | 단일 checkpoint latency가 아님. Async에서는 background 완료를 기다리지 않은 반환 시간 포함 |
| Blocking finalization | rank별 blocking `finalize_async_saves` 호출 시간의 합과 rank 최대값 | 아직 끝나지 않은 async 작업을 기다리는 시간. Save 합에 단순히 더해도 첫 enqueue부터 완료까지의 wall time은 아님 |
| Checkpoint 크기 | probe 시점의 **최신 iteration**에 있는 `.distcp` shard bytes를 rank 간 합산 | 한 checkpoint의 논리 크기. 모든 저장 iteration의 합·device write traffic·metadata 전체 크기가 아님 |
| CUDA peak allocated | PyTorch allocator가 기록한 peak allocation | 전체 device 사용량이 아님. 측정 구간·rank 집계가 같은 값끼리 비교 |
| Host memory pressure | 노드의 첫 `MemAvailable` − 측정 중 최소 `MemAvailable` | process RSS가 아닌 노드 전체 변화량. 다른 process·page cache·초기 노드 상태의 영향 포함 |
| Buffered read 처리율 | rank별 logical read bytes 합 ÷ rank별 read 시간 최대값 | probe는 rank 순서로 실행되므로 동시 실행한 cluster throughput이 아닌 집계 지표. Cache 분류도 함께 확인 |
| TRL model restore 시간 | 별도 `tuned` process의 model load 시작부터 checkpoint 적용 완료까지 | LoRA는 base+adapter model 생성, ZeRO-3 full은 skeleton·engine 준비+native checkpoint load를 포함 |
| TRL restore 유효 처리율 | base snapshot과 checkpoint logical bytes ÷ model restore 시간 | 파일 순차 read가 아니라 실제 model reconstruction의 end-to-end 지표 |
| rMAD | `median(abs(x - median(x))) / median(x)` | 반복 간 산포. 10% 이하는 추가 반복 판단 규칙이며 통계적 유의성·정확성 보장은 아님 |

MB/GB는 10진 bytes, MiB/GiB/TiB는 2진 bytes입니다. 예를 들어 69.4 MiB는 약 72.8 MB입니다.
Spark는 unified memory를 사용하므로 CUDA peak와 host pressure를 더해서 총 메모리로 해석하지 않습니다.
원본 manifest는 커밋되지 않으므로 기존 표의 수치는 보고된 요약값이며, 새 집계·오차 검증에는 해당 run의 원본이 필요합니다.

## Detailed Documents

- [Run Experiments](experiments/running.md): preset 선택, 메모리 추정, 실행 구성과 output 확인
- [30B Results](experiments/30b-results.md): 검증 실행, GPU·메모리 결과와 Qwen/GLM 비교
- [Checkpoint and Memory I/O](experiments/checkpoint-io.md): single-node·분산 I/O 설계, 측정값과 한계
- [Scaling Estimates](experiments/scaling-estimates.md): 100B–1T state·checkpoint 용량 추정
