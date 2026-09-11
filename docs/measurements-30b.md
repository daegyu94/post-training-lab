# 30B Local NVMe Checkpoint와 Memory 실험

`spark1`·`spark2`의 local NVMe만 사용해 30B 모델의 checkpoint I/O와 memory footprint를 측정한 기록입니다.
NFS는 조건에 포함하지 않고 Megatron checkpoint는 rank별 local shard 저장 성능만 평가합니다.

실행 절차·측정 field·집계식·유효성 임계값은 모두 코드가 source of truth입니다 — 진입점 `experiments/checkpoint_memory_30b.py`, read/cache 분류 `experiments/checkpoint_io_probe.py`, cohort 생성 `experiments/prepare_checkpoint_cohort.py`.
이 문서는 코드에서 읽어낼 수 없는 것만 남깁니다: 왜 이렇게 측정했는지와 실제로 무엇이 나왔는지.

## 답하려는 질문

1. Megatron distributed checkpoint의 논리 크기와 실제 local NVMe write traffic은 얼마인가?
2. Sync와 async checkpoint가 save latency, finalization과 학습 step time에 어떤 차이를 만드는가?
3. Megatron의 buffered read에서 page cache가 cold·warm 성능에 미치는 영향은 얼마인가?
4. DeepSpeed NVMe offload의 Direct I/O와 framework checkpoint의 buffered I/O는 어떻게 다른가?
5. 같은 30B 모델에서 framework와 분산 전략에 따라 CUDA·unified host memory·NVMe footprint가 어떻게 달라지는가?
6. Sequence length 2048/4096/8192에서 Megatron LoRA의 memory footprint가 어떻게 달라지는가?
7. LoRA trainable parameter 비율이 커지면 checkpoint 크기와 I/O가 어떻게 달라지는가?

이 결과는 local checkpoint의 장애 복구, topology 변경 restore, power-loss durability 또는 framework 간 절대적 우열을 증명하지 않습니다.

## 고정 조건

| 항목 | 값 |
| --- | --- |
| Model | `Qwen/Qwen3-30B-A3B` revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` |
| Dataset | `HuggingFaceH4/ultrachat_200k` revision `8049631c405ae6576f93f445c6b8166f76f5505a` |
| Topology | `spark1`·`spark2`, 노드당 process 하나, world size 2, TP=1·PP=1·EP=2·DP=2 |
| Precision | BF16 (그래프 축이 아니라 통제 변수) |
| Sequence / batch | length 2048, micro 1, global 2, seed 42 |
| Storage | 각 노드 `/mnt/post-training/<backend>` local NVMe |

`spark1`과 `spark2`에서 같은 경로 문자열은 서로 다른 물리 disk를 가리킵니다.
Metadata와 shard가 독립 filesystem에 나뉘면 새 process가 완전한 checkpoint를 찾을 수 없으므로 이 실험에서는 `tuned`·`resume`을 실행하지 않고 save 경로만 측정합니다(`CHECKPOINT_PLACEMENT=local`, `STAGE=train`).

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
따라서 측정에는 data flush 비용이 포함되지만 power loss durability를 입증하지는 않습니다.

TRL DeepSpeed ZeRO-3는 finetuning mode·optimizer·checkpoint format·runtime offload가 모두 다르므로 Megatron의 순수 baseline이 아니라 framework-native system 비교입니다.

## 측정 설계에서 의도적으로 선택한 것

- **Async throughput의 분모로 `save()` 반환 시간을 쓰지 않습니다.** 그 값은 대부분 enqueue 작업만 나타냅니다. 첫 enqueue 시작부터 blocking finalization 완료까지를 씁니다.
- **전역 `drop_caches`를 쓰지 않습니다.** 공유 cluster이기 때문이며, 대신 파일별 `POSIX_FADV_DONTNEED`로 eviction을 요청하고 관찰된 device-read delta로 cache 상태를 분류합니다. `POSIX_FADV_DONTNEED`는 advisory이므로 eviction 성공을 가정하지 않고, read-ahead 때문에 physical bytes가 logical을 넘을 수 있어 분류 임계값은 정확한 cache-hit ratio가 아닙니다.
- **Direct I/O baseline은 storage microbenchmark입니다.** Megatron framework throughput으로 표시하지 않습니다.
- **Cohort는 결정적으로 선택한 소수 행입니다.** Megatron 전처리는 truncate하지 않고 제한을 넘는 행에서 중단하므로 canonical split 전체를 쓸 수 없습니다. 2048 token 이하 UltraChat 32행(`ultrachat-qwen3-30b-2048-v1`)을 모든 조건에서 같은 순서·seed로 재사용합니다. 이 sample 수는 checkpoint·peak memory 측정에는 충분하지만 학습 수렴이나 dataset 품질의 근거가 아닙니다.
- **Sequence length 효과는 framework 비교와 섞지 않습니다.** Megatron LoRA의 별도 fixed-padding sweep으로만 측정합니다. 긴 step은 async background I/O를 숨길 시간도 늘리므로 length sweep 결과로 sync/async 우열을 판단하지 않습니다.
- **측정할 수 없는 값은 zero가 아니라 `null`로 기록하고 invalid로 분류합니다.**
- **각 run의 checkpoint는 probe와 metric 수집 직후 삭제합니다**(`cleanup_checkpoints()`). 모든 iteration·반복의 checkpoint를 보존하면 local storage를 소진하기 때문이며, manifest·measurement record는 그대로 남습니다.

## 실행

기본값은 실제 GPU 작업을 시작하지 않는 dry-run입니다.

```bash
python experiments/checkpoint_memory_30b.py \
  --setup setups/spark/local.json \
  --phase checkpoint \
  --output results/checkpoint-30b \
  --execute

python experiments/checkpoint_memory_30b.py \
  --setup setups/spark/local.json \
  --phase memory \
  --output results/memory-30b \
  --execute
```

Checkpoint phase는 sync/async warmup과 측정, rank별 resource sampling, warm-after-write/cold/warm buffered read와 Direct I/O baseline을 한 번에 수행합니다.
Memory phase는 4096/8192 Megatron pilot과 반복, TRL DDP/FSDP2 LoRA, TRL ZeRO-3 NVMe full-SGD를 실행합니다.
`LEN-2048` Megatron memory 값은 checkpoint phase 결과를 재사용합니다.

## 실측 결과

Raw manifest·measurement record는 커밋하지 않으므로(`results/`는 gitignore 대상) 아래는 요약값입니다.

### Megatron local checkpoint (sync vs async)

5회 반복 모두 종료 기준(rMAD ≤ 10%)을 만족해 8회로 확장하지 않았습니다.

| Variant | Run 수 | Checkpoint 크기(median) | Save 호출 시간(median) | Cold-buffered read(median) | rMAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| sync | 5 | 69.4 MiB | 2.356s | 3.12 GB/s | 0.57% |
| async | 5 | 69.4 MiB | 1.301s | 3.58 GB/s | 1.62% |

LoRA rank 8 고정이라 sync/async가 같은 크기를 저장하므로 크기 차이는 없고, async가 blocking host-call 기준 약 1초 빠릅니다 — enqueue만 하고 반환하는 async 설계와 일치합니다.

### Memory footprint

| 조건 | Run 수 | CUDA peak allocated(median) | Host memory pressure(median) |
| --- | ---: | ---: | ---: |
| `LEN-4096` (Megatron LoRA) | 3 | 39.5 GB | 53.7 GB |
| `LEN-8192` (Megatron LoRA) | 3 | 57.6 GB | 72.6 GB |
| `MEM-TRL-DDP` | 3 | 59.8 GB | 69.7 GB |
| `MEM-TRL-FSDP2` | 3 | 34.1 GB | 104.0 GB |
| `MEM-TRL-Z3-NVME` | 3 | 6.4 GB | 91.8 GB |

Unified-memory hardware이므로 CUDA와 host 측정값을 더하지 않고 별도 panel로 봅니다.
`EST-MEG-ADAM`(Megatron full + Adam) 추정값은 rank당 160.8 GiB로 119 GiB 예산을 초과해 실행하지 않았습니다.
`EST-MEG-SGD`(full + SGD)는 96.6 GiB로 예산 안에 들어오지만 pilot 실행은 별도로 결정하지 않았습니다(추정만 수행).

`LEN-1024`는 최초 설계에서 checkpoint phase의 2048-cohort를 그대로 재사용하다 구조적으로 항상 실패했습니다(cohort 안에 1024 초과 샘플이 있는데 native 전처리는 truncate하지 않음).
이를 계기로 sweep 하한을 2048로 올리고 상한을 8192로 확장했습니다 — activation/attention memory가 non-trivial하게 늘어나는 구간은 위쪽에서 더 잘 드러납니다.

DeepSpeed의 Direct offload traffic과 buffered ZeRO-checkpoint traffic을 분리해 보고하는 항목은 아직 측정하지 않았습니다.

## LoRA trainable-ratio가 checkpoint I/O에 미치는 영향

위 checkpoint 측정은 LoRA rank 8(전체 model-parallel shard의 약 0.03%)에서만 수행했습니다.
같은 파이프라인에서 `LORA_DIM`만 바꿔 비율 축을 확인했습니다(sync만 사용 — sync/async 비교는 위에서 이미 끝났습니다).

Megatron은 학습 시작 시 rank-local model-parallel shard 기준 trainable parameter 수와 비율을 로그로 남깁니다.
`lora_dim=8`에서 trainable 5,111,808 / shard 16,041,719,808 = 0.0319%이고, LoRA는 `target_modules`에 rank당 `r × (in_dim + out_dim)`을 더하므로 trainable 수는 `lora_dim`에 정확히 비례합니다.
따라서 rank 1당 638,976이고 목표 비율 `R`은 `lora_dim = round(R × 16,041,719,808 / 638,976)`으로 역산합니다.
비율은 global 30.52B가 아니라 Megatron이 실제로 로그에 남기는 rank-local shard 기준입니다.

| Variant | `LORA_DIM` | 역산 비율 | Checkpoint 크기(median) | Save 호출 시간(median) | rMAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ratio-0.1pct` | 25 | 0.0996% | 224.9 MB | 1.44s | 2.10% |
| `ratio-0.5pct` | 126 | 0.5019% | 1,128.4 MB | 2.03s | 0.13% |
| `ratio-1pct` | 251 | 0.9998% | 2,246.6 MB | 2.61s | 0.60% |

12/12 run 통과(pilot 3 + 측정 9), 세 조건 모두 rMAD가 10% 기준을 크게 밑돌아 8회로 확장하지 않았습니다.

**선형성**: 0.5pct/0.1pct = 5.02배(`LORA_DIM` 비 126/25 = 5.04), 1pct/0.1pct = 9.99배(비 251/25 = 10.04) — checkpoint 크기가 `LORA_DIM`에 선형 비례함을 실측으로 확인했습니다.
벗어났다면 optimizer state 계산이나 target module 구성이 예상과 다르다는 신호입니다.
Rank 역산 자체도 `ratio-0.1pct` pilot의 Megatron 로그로 확인했습니다: `Trainable parameters: 15,974,400`(계산값과 동일), `Trainable percentage: 0.10%`.

이 조건들은 plan만 바꿔 실행합니다.

```bash
python experiments/checkpoint_memory_30b.py \
  --setup setups/spark/local-checkpoint-memory-30b.json \
  --phase checkpoint \
  --plan experiments/megatron/lora-ratio-checkpoint-io.json \
  --output results/lora-ratio-checkpoint-io \
  --repeats 3 --execute
```
