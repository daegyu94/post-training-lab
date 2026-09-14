# Checkpoint and Memory I/O

<a id="checkpoint-and-memory-experiment"></a>

`spark1`·`spark2`의 local NVMe만 사용해 30B 모델의 checkpoint I/O와 memory footprint를 반복 측정한 전용 실험입니다.
NFS는 repository checkout에만 사용하고 Megatron checkpoint는 rank별 local shard 저장 성능만 평가합니다.

실행·집계는 `experiments/checkpoint_memory_30b.py`, read/cache 분류는 `experiments/checkpoint_io_probe.py`, cohort 생성은 `experiments/prepare_checkpoint_cohort.py`를 기준으로 합니다.

**이 문서에서 찾을 것과 찾지 못할 것**

| 찾는 것 | 위치 |
| --- | --- |
| 분산 checkpoint write의 최신 authoritative 수치 | [30B Controlled Results](30b-results.md#controlled-results-2026-09-15) — 여기에만 한 번 기록 |
| 실험 설계, I/O 경로 정의, 측정 규칙 | 아래 [I/O Paths](#io-paths), [Measurement Design](#measurement-design) |
| 실행 방법 | 아래 [Run the Measurements](#run-the-measurements) |
| 실제 model restore 시간 | 아래 [Single-node TRL I/O Experiment](#single-node-trl-io-experiment) |
| LoRA 비율과 checkpoint 크기의 관계 | 아래 [LoRA Ratio and Checkpoint I/O](#lora-ratio-and-checkpoint-io) |
| 2026-09-14 이전 탐색 측정 | 아래 [Historical Measured Results](#historical-measured-results-2026-09-14) — 최신 표와 합치지 않음 |

## Current Matrix

2026-09-15 matrix는 두 30B 모델에 같은 UltraChat revision, TP=1·PP=1·EP=2, BF16, micro/global batch 1/2, seed 42와 `TRANSFORMER_IMPL=transformer_engine`을 고정했습니다.
각 cell은 별도 warmup 또는 pilot 뒤 3회 측정했고 save-call rMAD가 10%를 넘으면 8회로 연장하도록 했지만 연장이 필요한 cell은 없었습니다.

| 구분 | Qwen | GLM | 결과 |
| --- | --- | --- | --- |
| Distributed write | sync/async, EP2 `torch_dist`, DP2 `fsdp_dtensor` | 동일 | 각각 12/12 통과 |
| TE memory | length 4096/8192 | 동일 | 각각 8/8 통과 |
| Multi-step checkpoint | 8 steps, interval 2, sync/async | 동일 | 각각 8/8 통과 |
| Recompute | length 2048/4096, full/selective | 모델 간 비교 대상 아님 | Qwen 16/16 통과 |
| Full-SFT capacity | SGD 1-step pilot | 미계획 | Qwen은 첫 step 전 global OOM |

Distributed restore는 shared checkpoint store가 없으므로 계획하지 않습니다.
Recompute를 모델 간 결론으로 쓸 때만 GLM matrix를 추가하고, async 장기 throughput이 필요할 때만 100-step 이상 실험을 추가합니다.

## Research Questions

| # | 질문 | 현재 상태 |
| ---: | --- | --- |
| 1 | Megatron distributed checkpoint의 논리 크기와 실제 local NVMe write traffic은 얼마인가? | 논리 크기는 측정 완료. device write traffic은 별도 계측 필요 |
| 2 | Sync와 async가 save latency, finalization, 학습 step time에 어떤 차이를 만드는가? | 8-step 범위까지 측정 완료. 장기 overlap 미검증 |
| 3 | buffered read에서 page cache가 cold·warm 성능에 미치는 영향은? | cold/warm 분류로 측정. eviction은 advisory라 cache-hit ratio는 아님 |
| 4 | DeepSpeed NVMe offload의 Direct I/O와 framework checkpoint의 buffered I/O는 어떻게 다른가? | 경로 구분만 문서화. Direct·buffered traffic 분리 측정은 미실행 |
| 5 | framework·분산 전략별 CUDA·host·NVMe footprint 차이는? | 측정 완료. 조건이 서로 달라 backend 우열로 읽지 않음 |
| 6 | Sequence length 2048/4096/8192에서 Megatron LoRA footprint는? | 측정 완료. backend 통제 후 결론은 [30B Results](30b-results.md#sequence-length-기울기) |
| 7 | LoRA trainable 비율이 커지면 checkpoint 크기와 I/O는? | Qwen에서 측정 완료. 다른 target module은 미확인 |
| 8 | 실제 Megatron restore의 cold/warm latency와 유효 처리율은? | shared checkpoint store가 없어 distributed restore는 미측정 |

이 결과는 local checkpoint의 장애 복구, topology 변경 restore, power-loss durability 또는 framework 간 절대적 우열을 증명하지 않습니다.

## Historical I/O Conditions (2026-09-14)

| 항목 | 값 |
| --- | --- |
| Model | `Qwen/Qwen3-30B-A3B` revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`, `zai-org/GLM-4.7-Flash` revision `7dd20894a642a0aa287e9827cb1a1f7f91386b67` |
| Dataset | `HuggingFaceH4/ultrachat_200k` revision `8049631c405ae6576f93f445c6b8166f76f5505a` |
| Topology | 분산: `spark1`·`spark2`, 노드당 process 하나, world size 2, Megatron EP=2 또는 FSDP DP=2; single-node: `spark1`, world size 1 |
| Precision | BF16 (그래프 축이 아니라 통제 변수) |
| Sequence / batch | length 512, micro 1, global 2, seed 42, 1 optimizer step |
| Storage | 각 노드 `/mnt/post-training/<backend>` local NVMe |

`spark1`과 `spark2`에서 같은 경로 문자열은 서로 다른 물리 disk를 가리킵니다.
Metadata와 shard가 독립 filesystem에 나뉘면 새 process가 완전한 checkpoint를 찾을 수 없으므로 이 실험에서는 `tuned`·`resume`을 실행하지 않고 save 경로만 측정합니다(`CHECKPOINT_PLACEMENT=local`, `STAGE=train`).

## I/O Paths

설치된 구현의 I/O 동작이 서로 다르므로 하나의 `NVMe I/O` 항목으로 묶지 않습니다.

**한 줄 요약**: Megatron sync·async, TRL FSDP2 DCP, DeepSpeed ZeRO checkpoint artifact는 모두 page cache를 쓰는 **buffered I/O**입니다. **Direct I/O(`O_DIRECT`, page cache 우회)는 DeepSpeed의 parameter/optimizer offload 한 경로뿐**입니다 — 같은 DeepSpeed 안에서도 offload와 checkpoint 저장은 다른 I/O 경로를 씁니다.

Buffered인 넷은 모두 PyTorch 직렬화(`torch.distributed.checkpoint`, `torch.save`)를 그대로 쓰고, DeepSpeed offload만 이를 거치지 않고 자체 AIO extension으로 `O_DIRECT`를 명시적으로 엽니다.

| 경로 | File I/O | Page cache | 완료 기준 |
| --- | --- | --- | --- |
| Megatron sync checkpoint write | buffered | 사용 | data-file `fsync()` 포함 |
| Megatron async checkpoint write | buffered worker thread | 사용 | enqueue와 blocking finalization 분리, data-file `fsync()` 포함 |
| Megatron checkpoint read | buffered | 사용 | 일반 file read |
| TRL FSDP2 PyTorch DCP | buffered | 사용 | collective `save()` 반환, `FileSystemWriter(sync_files=True)` |
| DeepSpeed parameter/optimizer offload | Linux AIO with `O_DIRECT` | 우회 | AIO 완료, 별도 `fsync()`는 관찰되지 않음 |
| DeepSpeed ZeRO checkpoint artifact | `torch.save()` / `torch.load()` | 사용 | offload AIO 경로와 별개 |

Megatron은 data file과 metadata에 `fsync()`를 호출하지만 metadata rename 이후 부모 directory의 `fsync()`는 관찰되지 않았습니다.
따라서 측정에는 data flush 비용이 포함되지만 power loss durability를 입증하지는 않습니다.

TRL DeepSpeed ZeRO-3는 finetuning mode·optimizer·checkpoint format·runtime offload가 모두 다르므로 Megatron의 순수 baseline이 아니라 framework-native system 비교입니다.

## Measurement Design

- **Async 완료 시간**: 첫 enqueue 반환 시간과 마지막 blocking finalization을 분리하고, 둘의 합을 checkpoint 완료 시간으로 사용합니다.
- **Cache 분류**: 공유 cluster의 전역 `drop_caches` 대신 파일별 `POSIX_FADV_DONTNEED`를 요청하고 device-read delta를 관찰합니다. Eviction은 advisory이고 read-ahead도 있으므로 임계값은 정확한 cache-hit ratio가 아닙니다.
- **Direct I/O baseline**: storage microbenchmark이며 Megatron throughput이 아닙니다.
- **Cohort**: 모델별 tokenizer로 고른 512-token 이하 UltraChat을 사용합니다. 분산은 32 train/8 eval에서 실제 4/1행만 읽고 single-node는 4/1행을 사용하며, 수렴·데이터 품질의 근거가 아닙니다.
- **Sequence length**: Megatron LoRA fixed-padding sweep으로 분리합니다. 긴 step은 async I/O를 숨길 시간도 늘리므로 sync/async 우열로 해석하지 않습니다.
- **누락값**: `null`로 기록하고 invalid로 분류합니다.
- **용량 관리**: probe·metric 수집 직후 `cleanup_checkpoints()`로 checkpoint를 삭제하고 manifest·measurement record는 보존합니다.

<a id="single-node-trl-io-experiment"></a>

## Single-node TRL I/O Experiment

기존 `checkpoint_io_probe.py`의 순차 shard read는 저장장치와 page cache 상태를 설명하는 보조 microbenchmark로 유지하되, model loading 성능의 대표값으로 사용하지 않습니다.
실제 restore는 `experiments/model_restore_30b.py`가 `spark1`에서 별도 TRL `tuned` process를 실행하고 model과 checkpoint가 적용된 시점까지 직접 측정합니다.

| 구분 | LoRA r=8/16/32 | DeepSpeed ZeRO-3 full SFT |
| --- | --- | --- |
| 학습 목적 | Adapter 크기 변화 생성 | 실제 full-state checkpoint 생성 |
| 학습량 | 1 optimizer step | 1 optimizer step |
| 저장물 | TRL/PEFT adapter | Native ZeRO-3 model checkpoint |
| restore | Base snapshot+adapter | Model skeleton+ZeRO checkpoint load |
| cache 조건 | Base와 adapter를 함께 cold/warm 처리 | Base와 ZeRO checkpoint를 함께 cold/warm 처리 |

실험은 다음 순서로 진행합니다.

1. 두 30B 모델에 모델별 tokenizer로 앞에서부터 선택한 동일 UltraChat revision의 512-token 이하 4 train/1 eval cohort를 사용합니다. 필요한 행을 찾으면 scan을 끝내므로 전체 데이터 길이 분포를 다시 계산하지 않습니다.
2. LoRA r=8/16/32 또는 ZeRO-3 full SFT를 1 optimizer step 실행하고 실제 checkpoint save 완료 시간과 크기를 기록합니다.
3. Base snapshot과 생성 checkpoint에 `POSIX_FADV_DONTNEED`를 요청하고 새 process에서 cold restore를 실행합니다.
4. Eviction 없이 새 process를 다시 실행해 warm restore를 측정합니다.
5. 위 lifecycle을 독립적으로 3회 반복하고 arithmetic mean과 표준편차를 기록합니다.

다음 명령은 먼저 dry-run plan만 만들며, 확인한 뒤 `--execute`를 추가해 실제 GPU restore를 수행합니다.

```bash
python experiments/model_restore_30b.py \
  --setup setups/spark/local.json \
  --dataset-source '<spark1-ultrachat-dir>' \
  --model qwen --variant lora-r8 --repeats 3 \
  --output results/single-node-io-qwen-lora-r8
```

`--model`은 `qwen|glm`, `--variant`는 `lora-r8|lora-r16|lora-r32|zero3-full`입니다.
Fine-tuning 품질은 목적이 아니므로 장기 학습과 restore 후 evaluation은 실행하지 않습니다.
`POSIX_FADV_DONTNEED`는 advisory이므로 cold라는 이름만으로 cache miss를 단정하지 않으며 `/proc/self/io`의 storage read bytes를 함께 기록합니다.
TRL launcher의 기존 resource sampler가 실행 중 `MemAvailable`과 swap 사용량을 0.2초 간격 raw JSONL로 기록합니다.
Memory pressure는 중심 결과가 아니라 모델별 최대 rank인 r=32의 validity guard로만 확인하며, 가용 메모리가 총 RAM의 10% 아래로 내려가면서 swap 사용량도 의미 있게 증가할 때 해당 모델의 rank sweep을 재검토합니다.

### Single-node Results (2026-09-14)

> **핵심**: cold restore는 LoRA rank와 거의 무관하게 Qwen 약 49~53초, GLM 약 38~40초입니다. 지배 요인은 adapter가 아니라 **base snapshot 58–63 GB의 재로딩**입니다.

LoRA 여섯 조건은 각 3회 모두 학습, checkpoint 생성, 별도 cold restore process와 별도 warm restore process를 완료했습니다.
표의 `±`는 sample standard deviation이며, checkpoint 시간과 크기는 3회 arithmetic mean입니다.

| Model / LoRA | Trainable 비율 | Checkpoint | Save | Cold restore | Warm restore |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen r=8 | 0.0219% | 38.2 MB | 0.936 s | 53.34 ± 2.36 s | 23.89 ± 9.39 s |
| Qwen r=16 | 0.0438% | 65.0 MB | 0.853 s | 50.94 ± 1.86 s | 9.74 ± 6.99 s |
| Qwen r=32 | 0.0875% | 118.4 MB | 2.488 s | 48.76 ± 4.61 s | 12.56 ± 9.57 s |
| GLM r=8 | 0.0351% | 62.4 MB | 0.551 s | 39.84 ± 1.64 s | 5.76 ± 2.00 s |
| GLM r=16 | 0.0702% | 104.4 MB | 1.727 s | 38.47 ± 0.97 s | 5.27 ± 0.34 s |
| GLM r=32 | 0.1403% | 188.6 MB | 2.622 s | 39.84 ± 4.55 s | 6.48 ± 1.38 s |

![Single-node 30B model restore](../figures/single-node-model-restore.svg)

값을 읽는 방법:

- **Checkpoint 크기는 rank에 비례**합니다(Qwen 38.2 → 65.0 → 118.4 MB, r 2배마다 약 1.7~1.8배). 반면 **cold restore 시간은 거의 평평**합니다(53.34 → 50.94 → 48.76 s). 두 값의 자릿수 차이(MB 대 수십 GB)가 이유입니다.
- Warm은 모든 경우 cold보다 짧지만 Qwen의 표준편차가 최대 ±9.57 s로 커서, rank별 작은 차이를 성능 추세로 해석하지 않습니다.
- GLM이 Qwen보다 cold·warm 모두 짧지만, 두 모델은 base snapshot과 loading 경로가 달라 이 표만으로 모델별 loading 효율을 판정하지 않습니다.
Raw manifest는 `results/single-node-io-{qwen|glm}-lora-r{8|16|32}*-20260914/manifest.json`이며 GLM r=32는 중단 없이 수집한 part 1·2의 세 record를 합쳤습니다.

**ZeRO-3 full SFT는 이 표에 값이 없습니다.** Qwen과 GLM을 각각 1회 실행했으나 optimizer 초기화 중 host memory와 swap을 소진한 뒤 kernel OOM으로 종료됐습니다.
checkpoint 자체가 만들어지지 않았으므로 cold/warm restore 값도 없습니다 — 현재 single-node memory capacity에서는 해당 full-SFT 조건이 성립하지 않습니다(2노드에서는 통과하며, [Topology Comparison](30b-results.md#ultrachat-full-sft-topology-comparison-2026-09-14) 참고).
Qwen raw 결과는 `results/single-node-io-qwen-zero3-full-pilot3-20260914`, GLM raw 결과는 `results/single-node-io-glm-zero3-full-pilot-20260914`에 있습니다.

## Run the Measurements

기본값은 실제 GPU 작업을 시작하지 않는 dry-run입니다.

```bash
python experiments/checkpoint_memory_30b.py \
  --setup setups/spark/local.json \
  --phase checkpoint \
  --output results/checkpoint-30b

python experiments/checkpoint_memory_30b.py \
  --setup setups/spark/local.json \
  --phase memory \
  --output results/memory-30b
```

계획을 확인한 뒤 같은 명령에 `--execute`를 붙입니다. Setup은 [Getting Started](../getting-started.md)에 따라 실제 경로로 준비하고, 각 노드에 선택 모델의 고정 snapshot과 원본 UltraChat을 준비합니다.

GLM checkpoint는 다음처럼 선택합니다. `--model glm`이 GLM cohort와 checkpoint plan을 함께 선택합니다.

```bash
python experiments/checkpoint_memory_30b.py \
  --setup setups/spark/local.json --model glm --phase checkpoint \
  --output results/checkpoint-30b-glm
```

2026-09-14의 local-NVMe write-only matrix는 모델에 맞는 `experiments/megatron/distributed-write-30b{,-glm}.json`을 `--plan`으로 주고 `--repeats 1 --steps 1 --within-run-warmup 0 --checkpoint-interval 1 --write-only --execute`로 실행했습니다.
TRL 비교 셀은 같은 driver에 `--phase memory --condition trl-fsdp2-dcp --repeats 1 --execute`를 사용합니다.

`--plan`을 직접 지정하면 모든 variant의 model·dataset ID/revision이 선택한 cohort와 맞아야 합니다. 예를 들어 Qwen plan에 `--model glm`을 함께 주면 SSH 전에 실패합니다.
Memory phase에서도 `--model glm`을 사용합니다. `--condition trl-ddp`처럼 조건을 선택할 수 있고, 전체 matrix의 GLM FSDP2는 현재 실패가 기록된 조건입니다.
`--resume`은 memory phase의 기존 run 수집·실행 재개 옵션이며 모델 checkpoint에서 학습을 이어가는 `STAGE=resume`과 다릅니다. 다른 모델의 출력 경로는 재사용할 수 없습니다.

Checkpoint phase는 sync/async warmup과 측정, rank별 resource sampling, warm-after-write/cold/warm buffered read와 Direct I/O baseline을 한 번에 수행합니다.
Buffered/Direct read 값은 storage 진단용이며 model restore 성능은 위의 별도 restore 실험으로 다시 측정합니다.
Memory phase는 4096/8192 Megatron pilot과 반복, TRL DDP/FSDP2 LoRA, TRL ZeRO-3 NVMe full fine-tuning을 실행합니다.
`LEN-2048` Megatron memory 값은 checkpoint phase 결과를 재사용합니다.

<a id="실측-결과"></a>

## Historical Measured Results (2026-09-14)

Raw manifest·measurement record는 커밋하지 않으므로(`results/`는 gitignore 대상) 아래는 요약값입니다.
아래의 n=1 후속 권고와 혼합-backend memory 표는 위 current matrix로 대체됐습니다.

### Historical Distributed Write Results (2026-09-14)

> **핵심**: n=1 탐색 측정입니다. 여기서 async 완료 시간이 sync에 가깝다는 신호를 얻었고, 그 신호를 3회 반복으로 확정한 것이 [2026-09-15 통제 결과](30b-results.md#distributed-checkpoint-write)입니다.

분산 결과는 실행 시간이 길어 조건당 1회만 수행한 exploratory 측정입니다.
모든 checkpoint는 각 노드의 local NVMe에 기록하고 inventory 수집 뒤 삭제했으며, 공유 remote filesystem을 사용하지 않으므로 restore는 실행하지 않았습니다.

| Model / backend | Format / layout | Size | Enqueue 또는 save | Finalize | 완료 latency |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen Megatron sync | EP=2 `torch_dist` | 10.6 MB | 1.301 s | 0.000 s | 1.301 s |
| Qwen Megatron async | EP=2 `torch_dist` | 10.6 MB | 0.429 s | 0.760 s | 1.188 s |
| Qwen Megatron Phase 2 | EP=2 `torch_dist` | 10.6 MB | 1.319 s | 0.000 s | 1.319 s |
| Qwen Megatron Phase 2 | DP=2 `fsdp_dtensor` | 20.9 MB | 0.425 s | 0.000 s | 0.425 s |
| Qwen TRL FSDP2 DCP | trainable model state | 28.2 MB | 0.245 s | 포함 | 0.245 s ⚠ |
| GLM Megatron sync | EP=2 `torch_dist` | 22.0 MB | 1.222 s | 0.001 s | 1.223 s |
| GLM Megatron async | EP=2 `torch_dist` | 22.0 MB | 0.170 s | 0.790 s | 0.960 s |
| GLM Megatron Phase 2 | EP=2 `torch_dist` | 22.0 MB | 1.041 s | 0.000 s | 1.042 s |
| GLM Megatron Phase 2 | DP=2 `fsdp_dtensor` | 43.1 MB | 0.401 s | 0.000 s | 0.402 s |

![Distributed checkpoint completion latency](../figures/distributed-checkpoint-write.svg)

처리 크기가 10–43 MB로 작아 고정 latency가 지배할 수 있으므로 completion latency를 주 지표로 사용합니다.
Throughput은 rank별 logical bytes 합을 이 시간으로 나눠 raw manifest에서 계산할 수 있지만 storage device의 절대 bandwidth나 framework 우열을 나타내지 않습니다.
특히 async는 enqueue만 보면 빠르지만 finalization을 포함한 Qwen 완료 시간은 1.188 s로 sync 1.301 s와 가깝습니다.

Qwen TRL DCP는 성공했지만 `MemAvailable`이 최소 18.0%까지 내려가고 swap 사용량이 최대 10.2 GB 증가했습니다.
10% validity guard에는 걸리지 않았으나 다른 성공 셀보다 memory pressure가 크므로 표와 그림에 주의가 필요한 탐색값으로 남깁니다.
GLM TRL FSDP2는 Accelerate 1.14의 CPU-efficient load가 plain `Tensor`에서 `.device_mesh`를 요구해 학습 전에 실패했습니다.
해당 최적화를 끄면 타입 오류는 사라졌지만 full replica를 FSDP sharding 전에 GPU로 올리면서 CUDA OOM이 발생해, 현재 환경에서는 유효한 GLM TRL DCP 값이 없습니다.
`zarr`는 object-store 지향 경로이고 이번 실험은 node-local filesystem write 완료가 질문이므로 Phase 2 비교에서 제외했습니다.

Raw manifest는 Qwen Phase 1 `results/distributed-io-qwen-n1-v4-20260914`, Qwen Phase 2 `results/distributed-io-qwen-phase2-n1-v5-20260914`, Qwen TRL `results/distributed-io-trl-dcp-qwen-n1-v3-20260914`, GLM Megatron `results/distributed-io-glm-n1-v2-20260914`에 있습니다.
GLM TRL 실패 증거는 `results/distributed-io-trl-dcp-glm-n1-v3-20260914`과 호환성 우회 run인 `results/distributed-io-trl-dcp-glm-n1-v4-20260914`에 있습니다.
그림은 `scripts/plot_io_experiment_results.py`가 이 raw record를 직접 읽어 생성하며, 실패한 GLM TRL 값은 그리지 않습니다.

기존 실험과 이번 개정의 차이는 다음과 같습니다.

| 기존 측정 | 2026-09-14 개정 | 당시 후속 판단 |
| --- | --- | --- |
| 순차 shard read probe | 실제 새 process TRL model restore를 cold/warm로 측정 | probe는 cache 진단용으로만 유지 |
| Qwen 중심 checkpoint 결과 | Qwen·GLM 모두 같은 512-token, 1-step 조건 | n=1이므로 결론이 필요하면 성공 셀만 3회 반복 |
| sync/async `save()` 반환시간 | async blocking finalization까지 포함한 완료시간 | 장기 학습 overlap은 별도 실험 필요 |
| 한 가지 Megatron layout | EP=2 `torch_dist`와 DP=2 `fsdp_dtensor` | payload 범위 통제 후에만 형식 우열 비교 |

<a id="megatron-local-checkpoint-sync-vs-async"></a>

### Local Checkpoint Results

5회 반복 모두 종료 기준(rMAD ≤ 10%)을 만족해 8회로 확장하지 않았습니다.
이후 반복 수는 memory phase와 같은 **3회로 통일**했습니다(`--repeats` 기본값).
아래 표의 5회 결과를 앞 3회만으로 다시 median을 내면 sync `2.3697s`(+0.57%), async `1.2824s`(-1.41%)로 rMAD는 1.06%·2.49%에 머물러, 나머지 2회가 결론을 바꾸지 않았기 때문입니다.

| Variant | Run 수 | Checkpoint 크기(median) | Run당 save 호출 누적 시간(median) | Cold-buffered read(median) | rMAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| sync | 5 | 69.4 MiB | 2.356s | 3.12 GB/s | 0.57% |
| async | 5 | 69.4 MiB | 1.301s | 3.58 GB/s | 1.62% |

LoRA rank 8에서 최신 checkpoint 크기는 같고 async의 run당 save 호출 누적 시간은 약 1초 짧습니다.
이는 rank별 save 호출 시간을 run 안에서 합산한 값의 차이이며, 단일 save latency나 전체 저장 완료 시간의 비교는 아닙니다. Async의 background I/O와 finalization, 학습 step time을 함께 봐야 학습 중단 비용을 평가할 수 있습니다.

### Memory Footprint

> **핵심**: 같은 TRL LoRA에서 FSDP2는 DDP 대비 CUDA peak를 43% 낮춥니다(59.8 → 34.1 GB). ZeRO-3 NVMe의 6.4 GB는 full fine-tuning + runtime offload라 같은 줄에서 비교할 조건이 아닙니다.

| 조건 | Run 수 | CUDA peak allocated(median) | Host memory pressure(median) |
| --- | ---: | ---: | ---: |
| `LEN-4096` (Megatron LoRA) | 3 | 39.5 GB | 53.7 GB |
| `LEN-8192` (Megatron LoRA) | 3 | 57.6 GB | 72.6 GB |
| `MEM-TRL-DDP` | 3 | 59.8 GB | 69.7 GB |
| `MEM-TRL-FSDP2` | 3 | 34.1 GB | 104.0 GB |
| `MEM-TRL-Z3-NVME` | 3 | 6.4 GB | 91.8 GB |

![30B CUDA memory footprint](../figures/memory-footprint.svg)

Megatron LoRA에서 sequence length를 4096에서 8192로 늘리면 CUDA peak allocated가 39.5 GB에서 57.6 GB로 약 46% 증가했습니다.
같은 TRL LoRA 조건에서 FSDP2는 DDP보다 CUDA peak가 59.8 GB에서 34.1 GB로 약 43% 낮았습니다.
ZeRO-3 NVMe의 6.4 GB는 full fine-tuning에 runtime offload를 건 결과이므로 LoRA 조건과 직접적인 backend 우열로 비교하지 않습니다.

Unified-memory hardware이므로 CUDA와 host 측정값을 더하지 않고 별도 panel로 봅니다.

Full-parameter 조건의 사전 추정과 실제 결과:

| 조건 | 추정 rank당 | 119 GiB 예산 판정 | 실제 |
| --- | ---: | --- | --- |
| `EST-MEG-ADAM`(Megatron full + Adam) | 160.8 GiB | 초과 | 실행하지 않음 |
| `EST-MEG-SGD`(full + SGD) | 96.6 GiB | 예산 내 | 2노드 pilot이 global OOM으로 종료 — **fit 판정 기각** |

추정은 allocator 여유분과 sharding되지 않은 임시 상태를 포함하지 않으므로 **하한 점검**입니다. 자세한 실패 지점은 [Full-SFT capacity](30b-results.md#full-sft-capacity)를 따릅니다.

> **`MEM-TRL-Z3-NVME`의 optimizer 표기 정정(2026-09-12).**
> 이 조건은 `OPTIMIZER=sgd`로 실행됐지만, DeepSpeed는 optimizer state를 offload하면 client optimizer를 `DeepSpeedCPUAdam`으로 교체합니다(`deepspeed/runtime/engine.py`는 다른 client optimizer를 `zero_force_ds_cpu_optimizer` 기본값에서 거부).
> 실제 저장된 optimizer state를 열어 확인한 결과 `exp_avg`·`exp_avg_sq`(Adam 모멘트)와 `betas`·`eps`가 들어 있었습니다 — SGD(momentum=0)라면 state가 없어야 합니다.
> 따라서 위 6.4 GB / 91.8 GB는 **SGD가 아니라 Adam 상태를 NVMe로 offload한 실행의 값**이며, 설정 파일은 실제 동작에 맞춰 `adamw`로 고쳤고 이 조합은 이제 [검증 단계에서 거부](../backends/trl.md#configure-training)됩니다.

`LEN-1024`는 2048-cohort의 길이 초과 행에서 실패해 sweep을 2048~8192로 변경했습니다.

DeepSpeed의 Direct offload traffic과 buffered ZeRO-checkpoint traffic을 분리해 보고하는 항목은 아직 측정하지 않았습니다.

<a id="lora-trainable-ratio가-checkpoint-io에-미치는-영향"></a>

## LoRA Ratio and Checkpoint I/O

> **핵심**: trainable 비율을 10배 늘리면 checkpoint 크기는 9.99배로 **거의 정비례**하지만, save 시간은 1.81배만 늘어납니다 — 크기에 비례하지 않는 고정 비용이 존재합니다.

위 LoRA rank 8(약 0.03%) 측정에 이어, sync 파이프라인에서 `LORA_DIM`만 바꿔 비율의 영향을 확인했습니다.

Megatron은 학습 시작 시 rank-local model-parallel shard 기준 trainable parameter 수와 비율을 로그로 남깁니다.
`lora_dim=8`에서 trainable 5,111,808 / shard 16,041,719,808 = 0.0319%이고, LoRA는 `target_modules`에 rank당 `r × (in_dim + out_dim)`을 더하므로 trainable 수는 `lora_dim`에 정확히 비례합니다.
따라서 rank 1당 638,976이고 목표 비율 `R`은 `lora_dim = round(R × 16,041,719,808 / 638,976)`으로 역산합니다.
비율은 global 30.52B가 아니라 Megatron이 실제로 로그에 남기는 rank-local shard 기준입니다.

| Variant | `LORA_DIM` | 역산 비율 | Checkpoint 크기(median) | Run당 save 호출 누적 시간(median) | rMAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ratio-0.1pct` | 25 | 0.0996% | 224.9 MB | 1.44s | 2.10% |
| `ratio-0.5pct` | 126 | 0.5019% | 1,128.4 MB | 2.03s | 0.13% |
| `ratio-1pct` | 251 | 0.9998% | 2,246.6 MB | 2.61s | 0.60% |

![LoRA parameter ratio and checkpoint I/O](../figures/lora-ratio-checkpoint.svg)

두 값이 다르게 움직입니다.

| 구간 | `LORA_DIM` 비 | Checkpoint 크기 비 | Save 시간 비 |
| --- | ---: | ---: | ---: |
| 0.1% → 0.5% | 5.04배 | 5.02배 | 1.41배 |
| 0.1% → 1.0% | 10.04배 | 9.99배 | 1.81배 |

- **크기는 `LORA_DIM`에 정비례**합니다. 역산이 맞았다는 것은 pilot 로그로도 확인됩니다: `Trainable parameters: 15,974,400`, `Trainable percentage: 0.10%`.
- **시간은 정비례하지 않습니다.** 동일한 저장 횟수에서 크기와 무관한 비용이 존재한다는 뜻입니다. 메타데이터·동기화·직렬화 중 무엇이 지배적인지는 별도 계측이 필요합니다.

시간·크기 값의 범위:

- 시간은 `CHECKPOINT_MODE=sync`에서 데이터 파일 `fsync()`까지 포함한 blocking `save()` 호출들을 rank별로 합한 뒤 최대값을 취한 것입니다(async의 enqueue 반환 시간과 달리 실제 쓰기를 포함하지만, 부모 디렉터리 `fsync()`가 빠져 있어 장애 durability를 뜻하지는 않습니다).
- 크기는 최신 iteration 하나의 shard 합입니다. 이 두 값으로 단일 checkpoint의 write bandwidth를 계산하지 않습니다.

12/12 run 통과(pilot 3 + 측정 9), 세 조건 모두 rMAD가 10% 기준을 크게 밑돌아 8회로 확장하지 않았습니다.

이 조건들은 plan만 바꿔 실행합니다.

```bash
python experiments/checkpoint_memory_30b.py \
  --setup setups/spark/local.json \
  --phase checkpoint \
  --plan experiments/megatron/lora-ratio-checkpoint-io.json \
  --output results/lora-ratio-checkpoint-io \
  --repeats 3 --execute
```
