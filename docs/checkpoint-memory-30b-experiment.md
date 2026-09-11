# 30B Local NVMe Checkpoint와 Memory 실험

## 목적

이 실험은 `spark1`과 `spark2`의 local NVMe만 사용해 30B 모델의 checkpoint I/O와 memory footprint를 측정합니다.
NFS는 실험 조건에 포함하지 않으며 Megatron checkpoint는 rank별 local shard 저장 성능만 평가합니다.

다음 질문에 답하는 것이 목표입니다.

1. Megatron distributed checkpoint의 논리 크기와 실제 local NVMe write traffic은 얼마인가?
2. Sync와 async checkpoint가 save latency, finalization과 학습 step time에 어떤 차이를 만드는가?
3. Megatron의 buffered read에서 page cache가 cold·warm 성능에 미치는 영향은 얼마인가?
4. DeepSpeed NVMe offload의 Direct I/O와 framework checkpoint의 buffered I/O는 어떻게 다른가?
5. 같은 30B 모델에서 framework와 분산 전략에 따라 CUDA, unified host memory와 NVMe footprint가 어떻게 달라지는가?
6. Fixed sequence length 1024, 2048, 4096에서 Megatron LoRA의 memory footprint가 어떻게 달라지는가?

이 결과는 local checkpoint의 장애 복구, topology 변경 restore, power-loss durability 또는 framework 간 절대적인 우열을 증명하지 않습니다.

## 고정 조건

| 항목 | 값 |
| --- | --- |
| Model | `Qwen/Qwen3-30B-A3B` |
| Model revision | `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` |
| Dataset | `HuggingFaceH4/ultrachat_200k` |
| Dataset revision | `8049631c405ae6576f93f445c6b8166f76f5505a` |
| Nodes | `spark1`, `spark2` |
| Processes | 노드당 process 하나, world size 2 |
| Precision | BF16 |
| Sequence length | 2048 |
| Micro batch size | 1 |
| Global batch size | 2 |
| Seed | 42 |
| Storage | 각 노드의 `/mnt/post-training/<backend>` local NVMe |

Precision은 그래프 축이 아니라 통제 변수로 둡니다.
연산 precision과 checkpoint의 실제 저장 dtype은 다른 속성이며 현재 저장소의 실행 경로는 BF16으로 고정되어 있습니다.

## I/O 동작

설치된 구현의 I/O 동작이 서로 다르므로 하나의 `NVMe I/O` 항목으로 묶지 않습니다.

| 경로 | File I/O | Page cache | 완료 기준 |
| --- | --- | --- | --- |
| Megatron sync checkpoint write | buffered | 사용 | data-file `fsync()` 포함 |
| Megatron async checkpoint write | buffered worker thread | 사용 | enqueue와 blocking finalization 분리, data-file `fsync()` 포함 |
| Megatron checkpoint read | buffered | 사용 | 일반 file read |
| DeepSpeed parameter/optimizer offload | Linux AIO with `O_DIRECT` | 우회 | AIO 완료, 별도 `fsync()`는 관찰되지 않음 |
| DeepSpeed ZeRO checkpoint artifact | `torch.save()` / `torch.load()` | 사용 | offload AIO 경로와 별개 |

Megatron은 data file과 metadata에 `fsync()`를 호출하지만 metadata rename 이후 부모 directory의 `fsync()`는 관찰되지 않았습니다.
따라서 측정에는 data flush 비용이 포함되지만 power loss에 대한 durability를 입증하지는 않습니다.

## Dataset cohort와 sequence length

UltraChat은 저장소가 지원하는 `HuggingFaceH4/ultrachat_200k`의 고정 revision을 사용합니다.
기존 30B UltraChat 실측은 GLM-4.7-Flash와 `MAX_LENGTH=4096` 조합이며 Qwen3-30B와 UltraChat 조합은 아직 검증되지 않았으므로 먼저 Qwen tokenizer로 cohort를 준비하고 pilot을 실행합니다.

Checkpoint I/O의 primary 조건에서는 `MAX_LENGTH=2048`과 fixed padding을 사용합니다.
Dataset의 최대 길이를 모두 수용하기 위해 상한을 올리면 checkpoint artifact 크기는 거의 그대로인 반면 연산량, memory와 async I/O가 겹칠 수 있는 시간이 달라지기 때문입니다.

준비 단계에서 canonical UltraChat split을 고정된 Qwen3-30B tokenizer와 저장소의 native prompt/completion renderer로 한 번 scan하고 다음을 manifest에 기록합니다.

- source 행 수와 tokenization 성공·실패 수
- token 길이의 minimum, p50, p90, p95, p99, p99.5와 maximum
- 1024, 2048, 4096 token 초과 행 수
- 각 상한에서 선택한 prompt ID와 제외 이유

Megatron 전처리는 truncate하지 않고 제한을 넘는 행에서 중단하므로 성능 run에서는 canonical split 전체 대신 결정적으로 선택한 benchmark cohort를 사용합니다.
Primary cohort는 tokenization에 성공하고 2048 token을 넘지 않는 UltraChat 32행으로 구성하며 prompt ID와 순서를 manifest에 기록합니다.
모든 조건에서 같은 cohort, 순서와 seed를 사용합니다.

8-step run 하나는 global sample 16개를 사용합니다.
여분의 cohort는 입력 계약을 바꾸지 않고 step을 소폭 늘릴 수 있게 합니다.
이 sample 수는 checkpoint와 peak memory 측정에는 충분하지만 학습 수렴이나 dataset 품질의 근거는 아닙니다.

Framework별 memory 비교에서는 모든 backend가 2048 token까지 padding해야 합니다.
고정 padding을 동등하게 적용할 수 없다면 UltraChat scan 결과에서 2048 이하인 같은 16개 long sample을 선택하고 실제 token 길이를 결과에 기록합니다.

Sequence length 자체의 memory 효과는 framework 비교와 섞지 않고 Megatron LoRA의 별도 1024/2048/4096 fixed-padding sweep으로 측정합니다.

## Part A: Megatron local checkpoint

### 조건

검증된 Qwen3-30B LoRA topology를 사용합니다.

```text
FINETUNING_MODE=lora
TP=1
PP=1
EP=2
DP=2
STAGE=train
MAX_STEPS=8
SAVE_INTERVAL=2
SAVE_OPTIMIZER=true
MAX_LENGTH=2048
PAD_TO_MAX_LENGTH=true
MEASURE_TIMING=true
```

| ID | Mode | Checkpoint 경로 |
| --- | --- | --- |
| `M-SYNC` | sync | node-local `<OUTPUT_DIR>/checkpoints` |
| `M-ASYNC` | async | node-local `<OUTPUT_DIR>/checkpoints` |

`spark1`과 `spark2`에서 같은 경로 문자열은 서로 다른 물리 disk를 가리킵니다.
Metadata와 shard가 독립적인 local filesystem에 나뉘면 새 process가 완전한 checkpoint를 찾을 수 없으므로 `tuned`와 `resume`은 실행하지 않습니다.

현재 공통 runner는 Megatron checkpoint directory를 공유 checkout 경로로 덮어씁니다.
실행 전에 `CHECKPOINT_PLACEMENT=local` 선택 하나를 추가해 이미 node-local인 `OUTPUT_DIR`에서 `CHECKPOINT_DIR`을 계산하고 기존 동작은 기본값으로 유지합니다.

### Checkpoint event

| Step | 용도 |
| ---: | --- |
| 2 | 초기화 영향을 받는 warm-up checkpoint |
| 4 | steady checkpoint 1 |
| 6 | steady checkpoint 2 |
| 8 | 종료 직전 checkpoint와 finalization |

각 run에서 step 4와 6의 median을 steady-state 값으로 사용합니다.
마지막 step의 async save는 다음 학습 step과 겹칠 수 없으므로 step 8은 별도로 분석합니다.

### 반복과 중단 기준

30B GPU 시간을 사용하기 전에 0.5B instrumentation smoke를 한 번 실행합니다.
그다음 통계에서 제외할 30B pilot을 mode별로 한 번 실행하고 mode별 측정 run 5개를 교차 순서로 실행합니다.

```text
SYNC pilot
ASYNC pilot
SYNC 1
ASYNC 1
ASYNC 2
SYNC 2
SYNC 3
ASYNC 3
ASYNC 4
SYNC 4
SYNC 5
ASYNC 5
```

한 run 안의 여러 checkpoint는 서로 상관된 관측값입니다.
따라서 독립 sample 수는 steady checkpoint event 10개가 아니라 mode별 run 5개입니다.

5개 run 이후 run별 median의 relative median absolute deviation을 계산합니다.
이 값이 10% 이하면 종료하고 그렇지 않으면 두 mode 모두 측정 run 8개까지 늘립니다.
개별 점, median, minimum과 maximum을 보고하되 이 sample 수로 p95나 p99 latency를 주장하지 않습니다.

### 측정값

모든 checkpoint와 rank에서 다음을 수집합니다.

- save 또는 enqueue host-call 시간
- blocking async finalization 시간
- end-to-end 완료 시간
- checkpoint file 수, logical bytes와 allocated blocks
- NVMe physical read/write bytes, operation 수와 busy time
- process read/write bytes
- checkpoint와 non-checkpoint step time
- CUDA peak allocated/reserved와 node `MemAvailable` 최솟값
- 성공 여부, iteration과 rank identity

집계값은 다음과 같이 계산합니다.

```text
logical checkpoint size
  = spark1 shard bytes + spark2 shard bytes

sync logical write throughput
  = logical checkpoint size / max(rank save completion time)

async effective write throughput
  = logical checkpoint size
    / (first enqueue start to blocking finalization completion)

physical write throughput
  = sum(node physical write bytes) / the same measurement window

write amplification
  = sum(node physical write bytes) / logical checkpoint size

checkpoint step overhead ratio
  = affected step time / non-checkpoint median step time
```

Async throughput의 분모에는 대부분 enqueue 작업만 나타내는 `save()` 반환 시간을 사용하지 않습니다.

## Part B: Megatron buffered read와 page cache

Local layout은 완전한 Megatron restore source가 아닙니다.
이 부분에서는 각 노드에서 완료된 rank-local shard를 순차적으로 읽고 결과에 `raw local-shard read; not Megatron restore`라고 표시합니다.

### Read 순서

모든 pending async save가 finalize된 뒤 step 8 checkpoint를 사용합니다.

1. Checkpoint 완료 직후 shard를 읽어 `warm-after-write`로 기록합니다.
2. 전역 `drop_caches` 대신 파일별 `POSIX_FADV_DONTNEED`로 eviction을 요청합니다.
3. 일반 buffered file I/O로 읽어 `cold-buffered`로 기록합니다.
4. 같은 파일을 즉시 다시 읽어 `warm-buffered`로 기록합니다.
5. 별도의 `O_DIRECT` sequential-read microbenchmark를 device baseline으로 실행합니다.

`fsync()`는 dirty data를 flush하지만 clean page를 제거하지 않으므로 save 직후 첫 read는 page cache의 영향을 받을 것으로 예상합니다.
`POSIX_FADV_DONTNEED`는 advisory이므로 eviction 성공을 가정하지 않고 관찰한 device-read delta로 cache 상태를 분류합니다.

다음 유효성 기준을 사용합니다.

```text
cold-buffered valid
  when physical read bytes >= 90% of logical shard bytes

warm-buffered valid
  when physical read bytes <= 10% of logical shard bytes
```

기준을 통과하지 못하면 `cache-contaminated` 또는 `unexpected-cache-miss`로 표시하고 해당 분포에서 제외합니다.
Read-ahead 때문에 physical bytes가 요청한 logical bytes를 넘을 수 있으므로 이 threshold는 정확한 cache-hit ratio가 아니라 분류 기준입니다.

Read metric은 다음과 같이 계산합니다.

```text
aggregate logical read throughput
  = sum(node shard bytes) / max(rank read wall time)

aggregate physical read throughput
  = sum(node physical read bytes) / max(rank read wall time)

physical-to-logical read ratio
  = sum(node physical read bytes) / sum(node shard bytes)
```

공유 cluster에서 전역 `drop_caches`를 사용하지 않습니다.
Direct I/O baseline은 storage microbenchmark이므로 Megatron framework throughput으로 표시하지 않습니다.

## Part C: DeepSpeed I/O

Runtime offload와 checkpoint artifact를 분리합니다.

### Direct NVMe offload

DeepSpeed parameter와 optimizer swap 구현은 `O_DIRECT`로 파일을 열고 Linux AIO operation을 submit합니다.
Cold/warm page-cache 축 없이 direct read/write throughput, IOPS, latency, queue depth와 optimizer step당 bytes를 측정합니다.
Direct I/O open 성공과 buffer가 구현의 alignment 요구사항을 만족하는지 확인합니다.

### ZeRO checkpoint artifact

설치된 checkpoint engine은 `torch.save()`와 `torch.load()`를 사용하므로 parameter와 optimizer offload가 Direct I/O를 사용해도 artifact 경로는 buffered I/O입니다.
ZeRO checkpoint read를 포함하면 Megatron과 같은 `warm-after-write`, `cold-buffered`, `warm-buffered` 분류를 적용합니다.

두 traffic을 별도로 보고합니다.

```text
whole-training Direct I/O traffic
checkpoint-window buffered I/O traffic
```

TRL DeepSpeed ZeRO-3는 finetuning mode, optimizer, checkpoint format과 runtime offload가 모두 다르므로 Megatron의 순수 baseline이 아니라 framework-native system 비교입니다.

## Part D: Memory footprint

### 조건

| ID | 구성 | 검증 범위 |
| --- | --- | --- |
| `MEM-MEG-EP2` | Megatron LoRA, EP2 | checkpoint run에서 측정 |
| `MEM-TRL-DDP` | TRL DDP LoRA | 측정 run 3개 |
| `MEM-TRL-FSDP2` | TRL FSDP2 LoRA | 측정 run 3개 |
| `MEM-TRL-Z3-NVME` | TRL DeepSpeed ZeRO-3 full, SGD, NVMe offload | 측정 run 3개 |
| `EST-MEG-SGD` | Megatron full, distributed optimizer, SGD | 추정 후 선택적 pilot |
| `EST-MEG-ADAM` | Megatron full, distributed optimizer, Adam | 추정만 수행 |

GPU 실행 전에 `experiments/estimate_memory.py`를 실행하고 parameters, gradients, optimizer, activations와 offload tier 추정값을 구분해 보존합니다.
현재 Megatron full Adam 추정값은 119 GiB node memory에 대해 rank당 약 149.7 GiB이므로 이 조건은 실행하지 않습니다.

Memory 실측 조건은 독립 run 3개로 시작합니다.
관찰된 peak 범위가 median의 5%를 넘으면 해당 조건을 5개 run으로 늘립니다.

### Sequence length sweep

Checkpoint sync/async 비교에서는 2048 하나만 사용하고 sequence length 효과는 Megatron LoRA에서 따로 측정합니다.

| ID | `MAX_LENGTH` | Padding | 반복 |
| --- | ---: | --- | --- |
| `LEN-1024` | 1024 | fixed | pilot 1회, 필요하면 측정 3회 |
| `LEN-2048` | 2048 | fixed | checkpoint run 결과 재사용 |
| `LEN-4096` | 4096 | fixed | pilot 1회, 안정적이면 측정 3회 |

각 길이에는 해당 상한 이하로 tokenization되는 같은 선택 규칙의 UltraChat cohort를 사용합니다.
`LEN-4096` pilot은 OOM, swap 급증, non-finite 값이나 비정상적인 step time이 없을 때만 반복 측정으로 확장합니다.

Sequence length가 커지면 activation과 workspace memory가 증가하고 attention compute는 길이에 더 민감하게 증가할 수 있습니다.
30B parameter와 optimizer state가 차지하는 고정 비용은 거의 변하지 않으므로 전체 memory가 sequence length에 정비례한다고 가정하지 않습니다.
긴 step은 async checkpoint background I/O를 숨길 시간을 늘릴 수도 있으므로 length sweep 결과로 sync/async 우열을 판단하지 않습니다.

### 측정값

Unified-memory hardware에서 CUDA와 host 측정값을 더하지 않고 다음 값을 수집합니다.

- rank-local CUDA peak allocated와 reserved bytes
- rank imbalance
- node baseline과 `MemAvailable` 최솟값
- process RSS와 swap 사용량
- 지원될 때 pinned bytes
- NVMe parameter와 optimizer directory의 logical·allocated size
- whole-run과 checkpoint-window NVMe traffic

```text
host memory pressure
  = baseline MemAvailable - minimum MemAvailable
```

Estimator breakdown과 실측 allocator 또는 host peak는 scope가 다르므로 별도 panel에 표시합니다.

## 결과 record

각 run manifest에는 Git commit과 dirty 상태, model과 dataset revision, package version, CUDA와 Torch version, topology, precision, batch와 sequence 설정, checkpoint policy, optimizer, node, NVMe device와 filesystem, 시작 시 free space, timestamp와 모든 rank exit code를 기록합니다.

각 checkpoint record에는 최소한 다음 field를 기록합니다.

```json
{
  "run_id": "...",
  "node": "spark1",
  "rank": 0,
  "iteration": 4,
  "mode": "async",
  "storage": "local_nvme",
  "io_path": "buffered_fsync",
  "save_call_seconds": 0.0,
  "blocking_finalize_seconds": 0.0,
  "completion_seconds": 0.0,
  "logical_bytes": 0,
  "physical_write_bytes": 0,
  "physical_read_bytes": 0,
  "peak_cuda_allocated_bytes": 0,
  "peak_cuda_reserved_bytes": 0,
  "host_mem_available_min_bytes": 0,
  "cache_condition": null,
  "success": true
}
```

측정할 수 없는 값에는 zero가 아니라 `null`을 사용합니다.

## 유효성 기준

다음을 모두 만족한 run만 결과에 포함합니다.

- 두 rank 모두 exit code가 zero입니다.
- 8개 optimizer step 모두 finite loss와 gradient로 완료됩니다.
- skipped 또는 NaN iteration이 없습니다.
- 두 노드에 step 2, 4, 6, 8 checkpoint가 존재합니다.
- 모든 pending async save가 정상적으로 finalize됩니다.
- Measurement record가 완전합니다.
- Local NVMe physical write가 관찰됩니다.
- 동등한 반복에서 checkpoint 크기가 안정적입니다.
- 관련 없는 대규모 NVMe workload가 측정 구간과 겹치지 않습니다.

누락된 측정값을 zero로 바꾸지 않고 invalid로 분류합니다.

## 그래프

X축에는 실험 조건을 사용하고 BF16은 제목이나 manifest에 기록합니다.

1. `Checkpoint artifact size`: model, optimizer, scheduler와 metadata bytes의 stacked bar
2. `Save latency`: save/enqueue, blocking finalization과 end-to-end 완료 시간의 개별 점과 median
3. `Write throughput`: logical, physical과 write-amplification panel
4. `Training impact`: checkpoint와 async background 구간을 표시한 step별 시간
5. `Megatron read`: `warm-after-write`, `cold-buffered`, `warm-buffered`와 Direct I/O device baseline
6. `DeepSpeed I/O`: 분리한 Direct offload와 buffered ZeRO-checkpoint panel
7. `Memory`: 분리한 CUDA peak, host pressure, NVMe footprint와 estimator panel
8. `Sequence length`: 1024, 2048, 4096별 CUDA peak, host pressure와 step time

Legend에는 동작이 드러나는 label을 사용합니다.

```text
Megatron shard | buffered+fsync | sync
Megatron shard | buffered+fsync | async
Megatron shard | buffered | warm-after-write
Megatron shard | buffered | cold
Megatron shard | buffered | warm
NVMe device baseline | O_DIRECT
DeepSpeed offload | O_DIRECT+AIO
DeepSpeed ZeRO checkpoint | buffered
```

## 실행 순서와 run 예산

1. UltraChat 고정 revision을 각 노드에 준비하고 Qwen tokenizer length manifest와 benchmark cohort를 생성합니다.
2. 0.5B run 하나로 instrumentation을 검증합니다.
3. 통계에서 제외할 30B Megatron sync와 async pilot을 실행합니다.
4. Megatron sync와 async를 각각 5회 측정하고 연결된 read 순서를 실행합니다.
5. 별도 실험 없이 같은 run에서 Megatron 2048 memory 값을 수집합니다.
6. Megatron LoRA의 1024와 4096 length pilot을 실행하고 안정성 기준을 통과한 조건만 각각 3회 측정합니다.
7. TRL DDP와 FSDP2 LoRA memory 조건을 각각 3회 실행합니다.
8. TRL ZeRO-3 NVMe full-SGD memory와 I/O 조건을 3회 실행합니다.
9. 추정값과 관찰된 headroom으로 Megatron full-SGD pilot 실행 여부를 결정합니다.
10. 유효한 run-level summary만 집계해 그래프를 생성합니다.

초기 30B 예산은 Megatron checkpoint run 10개와 TRL memory/I/O run 9개를 합한 측정 run 19개입니다.
Pilot은 이 수에 포함하지 않습니다.
Length sweep을 반복 측정으로 확장하면 1024와 4096 조건의 run 6개가 추가되어 최대 25개가 됩니다.

모든 iteration과 반복의 checkpoint를 보존하면 local storage를 소진할 수 있습니다.
Metric과 raw-read 검사를 마친 뒤 조건별 대표 checkpoint 하나와 manifest·measurement record를 보존하고 나머지는 명시적인 cleanup policy 아래에서만 제거합니다.
