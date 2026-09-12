# Experiments

Experiment 파일은 모델·데이터 revision과 학습 조건을, setup은 실행할 노드와 경로를 선언합니다.
공통 runner가 둘을 결합하며 `--execute`가 있을 때만 SSH 학습을 시작합니다.
첫 실행은 [Getting Started](getting-started.md), 설정 책임은 [Architecture](architecture.md)를 따릅니다.

두 노드에서 `Qwen/Qwen3-30B-A3B`(TRL LoRA·Megatron MoE)와 `zai-org/GLM-4.7-Flash`(Megatron MoE)의 SFT를 확인합니다.
30B는 메모리·통신·checkpoint·재로딩을, 0.5B smoke는 runner·전처리·저장 경로를 저비용으로 검사합니다.
정적 검사·CPU 테스트·dry-run·GPU 실행은 별도 증거로 구분합니다([AGENTS.md](../AGENTS.md), [판정 기준](getting-started.md#6-verify-the-result)).

## Presets

| 파일 | 노드·학습 | 단계 |
| --- | --- | --- |
| `experiments/trl/single-node-smoke.json` | 1노드 DDP LoRA, 1 step | base/train/tuned |
| `experiments/trl/smoke.json` | 2노드 DDP LoRA, 1 step | base/train/tuned |
| `experiments/megatron/smoke.json` | 2노드 full SFT, 2 step | base/train/tuned |
| `experiments/megatron/resume-smoke.json` | 2노드 full SFT, 2→3 step | base/train/resume/tuned |
| `experiments/trl/nvme-offload-30b.json` | 2노드 Qwen3 30B full SFT, ZeRO-3 NVMe offload | base/train/tuned |
| `experiments/megatron/qwen3-30b-lora.json` | 2노드 Qwen3 30B MoE LoRA, 평가와 node-local async checkpoint | train |
| `experiments/megatron/glm-4.7-flash-30b-lora.json` | 2노드 GLM-4.7-Flash 30B MoE LoRA, 평가와 node-local async checkpoint | train |
| `experiments/trl/glm-4.7-flash-30b-lora.json` | 2노드 GLM-4.7-Flash 30B DDP LoRA, no_robots | all |
| `experiments/megatron/qwen3-30b-full.json` | 2노드 Qwen3 30B MoE full-parameter — 아래 참고, 메모리 적합성 미검증 | all |

앞의 smoke 네 개는 Qwen2.5-0.5B와 고정 No Robots 입력을 쓰며 모델 품질·장기 수렴·성능 비교용이 아닙니다.
Megatron의 두 30B preset은 검증 당시와 같은 TP=1·PP=1·EP=2로 1 step, 평가 1회와 async checkpoint를 실행합니다.
TRL NVMe preset은 `train` 후 별도 `tuned` process에서 native ZeRO checkpoint를 복원해 평가합니다([30B NVMe 실습](../labs/nvme-30b/README.md)).

`qwen3-30b-full.json`은 **실측 미검증**입니다.
로컬 `no_robots` 5000행 중 길이 초과 행에서 중단됐으며, 메모리 비교 기준인 `MAX_LENGTH=2048`은 유지했습니다.
[메모리 추정기](#estimate-memory-before-running)의 per-rank 예측은 Adam 149.7 GiB(파라미터 29.9 + gradient 29.9 + Adam 89.6), `--optimizer sgd` 90.0 GiB입니다.
119 GiB 예산 대비 Adam은 약 31 GiB 초과, SGD는 이내지만 둘 다 추정값입니다.

`experiments/run.py`는 `--backend`, `--setup`, `--experiment`, `--output`을 요구합니다.
기본은 dry-run이고 `--timeout`은 기본 900초의 양의 정수이며 기존 출력 디렉터리는 재사용할 수 없습니다.
모든 preset은 `MODEL_ID`, `MODEL_REVISION`, `DATASET_ID`, `DATASET_REVISION`을 명시해야 하고, 백엔드가 지원하는 모든 CLI 옵션이 runner에 노출된 것은 아닙니다.
Runner는 output 디렉터리 이름을 `OBSERVATORY_RUN_ID`로 예약해 framework metric과 host telemetry를 같은 실행에 연결합니다.

## Estimate Memory Before Running

실행 전 snapshot의 `*.safetensors` 헤더에서 텐서 바이트를 읽어 메모리를 추정합니다.
아키텍처별 수식 대신 실제 크기를 쓰며, expert 가중치는 이름으로 식별해 EP sharding을 적용합니다.

```bash
python experiments/estimate_memory.py \
  --model-dir '<snapshot-dir>' \
  --backend megatron --finetuning-mode full --ep 2 --world-size 2 \
  --budget-gib 119
```

`--budget-gib`는 per-rank 예산과 비교해 `FITS`/`DOES NOT FIT`을 판정하고, `--offload cpu|nvme`는 optimizer·gradient를 on-device에서 빼되 해당 계층이 감당할 용량을 따로 알려줍니다.

| 검증 대상 | 실측 ([30B GPU Results](#30b-gpu-results)) | 예측 |
| --- | --- | --- |
| TRL DDP LoRA | 58.825 GiB | 59.2 GiB |
| TRL FSDP2 LoRA | 32.147 GiB | 29.3 GiB |
| Megatron LoRA EP=2 | 34.759 GiB | 30.8 GiB |

CUDA context·allocator 단편화·workspace를 제외한 **추정값**이므로 실측보다 낮을 수 있습니다(표에서 최대 -11%).
예산에 근접한 `FITS`는 실제 적합성을 보장하지 않습니다.

## Build an Experiment from Knobs

`experiments/build.py`는 backend·dataset·model·offload·epoch 옵션을 `experiments/generated/<output 이름>.json`으로 저장하고 `run.py`에 넘깁니다.

```bash
python experiments/build.py \
  --backend trl --dataset ultrachat \
  --model-id Qwen/Qwen2.5-0.5B-Instruct --model-revision <40-hex> \
  --setup setups/spark/local.json --output results/ultrachat-epoch1 \
  --epochs 1 --train-samples 256 --eval-samples 32
```

- 기본값은 실제 2노드 클러스터에 맞춰져 있습니다(`--nnodes 2`).
- `--dataset` 선택지는 고정 목록이 아니라 `datasets_lab.public_data.PRESETS`에서 읽습니다(`reference_only` 제외).
- `--epochs`와 `--max-steps`는 배타적입니다. TRL은 `SFTConfig`로 처리합니다. Megatron은 controller에서 `manifest.json`을 읽어 `steps = ceil(epochs * train_count / global_batch_size)`로 변환하므로, **현재 node-local 데이터 구성에서는 `--max-steps`를 씁니다.**
- `--offload {none,cpu,nvme}`는 TRL 전용입니다. `cpu`/`nvme`는 `--distributed-backend deepspeed`를 강제하고 해당 DeepSpeed 설정을 선택하며, NVMe profile은 `--finetuning-mode full`이 필요합니다. `--backend megatron`과 함께 쓰면 즉시 오류입니다.
- Megatron 전용 `--tp`/`--pp`/`--ep`/batch 옵션은 검증된 30B preset 기본값(TP=1, PP=1, EP=2)을 그대로 씁니다. 각 옵션의 뜻은 `--help`에서 확인합니다.
- Dataset은 [Datasets](datasets.md)에 따라 미리 준비합니다.

### 실제로 확인한 조합

`build.py --execute`로 확인한 조합입니다.

| Case | Backend | Model | Dataset | knob | 결과 |
| --- | --- | --- | --- | --- | --- |
| 1 | TRL DDP | Qwen2.5-0.5B-Instruct | ultrachat (256/32) | `--epochs 1` | base/train/tuned 통과 |
| 2 | TRL DeepSpeed | Qwen3-30B-A3B | no_robots | `--offload nvme --max-steps 1` | train 통과, 별도 `tuned` 평가 통과 |
| 3 | Megatron | Qwen3-30B-A3B | self_oss (40,000/8,000) | `--max-steps 1 --stage all` | base `1.402232` → tuned `1.342737` |
| 4 | Megatron | GLM-4.7-Flash | ultrachat (160,000/30,000) | `--max-steps 1 --max-length 4096 --stage all` | base `2.269922` → tuned `2.012836` |

**Case 1**은 256샘플·16 step에서 `base` 1.4340 → `train` 1.4308 → `tuned` 1.4308을 확인했습니다.
저장 전후 eval loss 일치는 이 실행의 adapter 재로딩 증거이며 모델 품질 지표는 아닙니다.

**Case 3·4**는 node-local에 흩어진 `torch_dist` metadata·shard를 NFS checkpoint 경로로 모은 뒤 base→train→tuned와 iteration 1 재로딩을 통과했습니다.
NaN·skipped iteration은 0, `checkpoint_reload_verified`는 `true`였습니다.
Case 4는 UltraChat 길이 초과로 `--max-length 2048`에서 실패한 뒤 4096으로 통과했습니다.

<a id="30b-gpu-results"></a>

## 30B GPU Results

2026-09-09에 commit `54a5fe0b69d168a86746d7de576d5dcae61c36b9`의 깨끗한 checkout으로 `spark1`·`spark2`에서 GPU당 process 하나를 실행했습니다.
**이 표는 1-step 실행 가능성만 보여주며 장기 안정성이나 학습 품질을 뜻하지 않습니다.**

Megatron은 TP=1·PP=1·EP=2·DP=2, BF16, sequence length 2048, global batch 2, attention LoRA를 사용했고 두 모델 모두 1 optimizer step, 평가, async `torch_dist` checkpoint와 양 rank exit 0을 확인했습니다.

| Model | Train result | Peak allocated |
| --- | --- | --- |
| Qwen3-30B-A3B | loss `4.110986`, grad norm `15.480`, step `7.28 s` | `34.759 GiB` |
| GLM-4.7-Flash | loss `2.497179`, grad norm `7.883`, step `6.37 s` | `34.671 GiB` |

TRL은 Qwen3-30B-A3B, 같은 precision·길이·batch와 attention LoRA를 사용했습니다.

| Backend | Train / eval loss | Peak allocated / reserved | Update evidence |
| --- | --- | --- | --- |
| DDP | `3.156563` / `2.587339` | `58.825 / 58.971 GiB` | sampled parameter delta가 0이 아님 |
| FSDP2 | `3.156250` / `2.566406` | `32.147 / 34.188 GiB` | optimizer step과 sharded checkpoint |

같은 조건은 `experiments/megatron/{qwen3-30b-lora,glm-4.7-flash-30b-lora}.json`으로 재실행할 수 있습니다.
[Checkpoint and Memory Experiment](#checkpoint-and-memory-experiment)는 같은 모델을 반복 측정한 별도 실험이므로, 이 표와 그쪽의 median은 서로 다른 run의 값입니다.

### NVMe and Full SFT

두 노드의 `/mnt/post-training`은 로컬 NVMe root filesystem에 있고 backend별 디렉터리에 `spark` 쓰기 권한이 있습니다.
Megatron Qwen3-30B-A3B와 GLM-4.7-Flash LoRA는 로컬 NVMe에 데이터 cache와 로그를 쓰고, NFS의 공통 `torch_dist` checkpoint에서 iteration 1을 새 process로 재로딩해 `STAGE=all`을 완료했습니다 — 정확한 수치는 위 [실제로 확인한 조합](#실제로-확인한-조합)의 Case 3·4와 같습니다.
Megatron은 NVMe를 native training state offload 대상으로 지원하지 않으므로 이 결과는 dataset cache와 checkpoint I/O 검증입니다.

30B full SFT의 DDP·FSDP2 실패는 dataset 전체 적재가 원인이 아닙니다.
DDP는 parameter와 gradient가 unified memory 한도에 근접하고, 설치된 Accelerate의 FSDP2 준비 과정은 sharding 전에 trainable BF16 parameter를 FP32로 올립니다.
남은 후보였던 TRL DeepSpeed ZeRO-3 NVMe offload는 2노드에서 1 optimizer step(train loss 약 13.21), 별도 `tuned` process 평가(eval loss 약 11.96)와 복구 가능한 native ZeRO checkpoint까지 확인했습니다.
같은 실행에서 두 노드 모두 `/mnt/post-training/trl/zero_stage_3` swap footprint가 약 256GiB로 각 노드 물리 RAM 119GiB보다 컸습니다 — NVMe offload 없이는 이 구성이 노드 RAM만으로 성립하지 않는다는 근거이며, 자세한 수치와 한계는 [30B NVMe 실습](../labs/nvme-30b/README.md#why-nvme-offload-is-necessary)을 따릅니다.

<a id="checkpoint-and-memory-experiment"></a>

## Checkpoint and Memory Experiment

`spark1`·`spark2`의 local NVMe만 사용해 30B 모델의 checkpoint I/O와 memory footprint를 반복 측정한 전용 실험입니다.
NFS는 조건에 포함하지 않고 Megatron checkpoint는 rank별 local shard 저장 성능만 평가합니다.

실행·집계는 `experiments/checkpoint_memory_30b.py`, read/cache 분류는 `experiments/checkpoint_io_probe.py`, cohort 생성은 `experiments/prepare_checkpoint_cohort.py`를 기준으로 합니다.
아래는 측정 이유와 결과입니다.

### 답하려는 질문

1. Megatron distributed checkpoint의 논리 크기와 실제 local NVMe write traffic은 얼마인가?
2. Sync와 async checkpoint가 save latency, finalization과 학습 step time에 어떤 차이를 만드는가?
3. Megatron의 buffered read에서 page cache가 cold·warm 성능에 미치는 영향은 얼마인가?
4. DeepSpeed NVMe offload의 Direct I/O와 framework checkpoint의 buffered I/O는 어떻게 다른가?
5. 같은 30B 모델에서 framework와 분산 전략에 따라 CUDA·unified host memory·NVMe footprint가 어떻게 달라지는가?
6. Sequence length 2048/4096/8192에서 Megatron LoRA의 memory footprint가 어떻게 달라지는가?
7. LoRA trainable parameter 비율이 커지면 checkpoint 크기와 I/O가 어떻게 달라지는가?

이 결과는 local checkpoint의 장애 복구, topology 변경 restore, power-loss durability 또는 framework 간 절대적 우열을 증명하지 않습니다.

### 고정 조건

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

### I/O 동작

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

### 측정 설계에서 의도적으로 선택한 것

- **Async throughput**: enqueue 반환 시간이 아니라 첫 enqueue부터 blocking finalization 완료까지를 분모로 씁니다.
- **Cache 분류**: 공유 cluster의 전역 `drop_caches` 대신 파일별 `POSIX_FADV_DONTNEED`를 요청하고 device-read delta를 관찰합니다. Eviction은 advisory이고 read-ahead도 있으므로 임계값은 정확한 cache-hit ratio가 아닙니다.
- **Direct I/O baseline**: storage microbenchmark이며 Megatron throughput이 아닙니다.
- **Cohort**: 길이 초과 시 중단하는 전처리에 맞춰 2048 token 이하 UltraChat 32행(`ultrachat-qwen3-30b-2048-v1`)을 같은 순서·seed로 재사용합니다. Checkpoint·peak memory 측정용이며 수렴·데이터 품질의 근거가 아닙니다.
- **Sequence length**: Megatron LoRA fixed-padding sweep으로 분리합니다. 긴 step은 async I/O를 숨길 시간도 늘리므로 sync/async 우열로 해석하지 않습니다.
- **누락값**: `null`로 기록하고 invalid로 분류합니다.
- **용량 관리**: probe·metric 수집 직후 `cleanup_checkpoints()`로 checkpoint를 삭제하고 manifest·measurement record는 보존합니다.

### 실행

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
Memory phase는 4096/8192 Megatron pilot과 반복, TRL DDP/FSDP2 LoRA, TRL ZeRO-3 NVMe full fine-tuning을 실행합니다.
`LEN-2048` Megatron memory 값은 checkpoint phase 결과를 재사용합니다.

### 실측 결과

Raw manifest·measurement record는 커밋하지 않으므로(`results/`는 gitignore 대상) 아래는 요약값입니다.

#### Megatron local checkpoint (sync vs async)

5회 반복 모두 종료 기준(rMAD ≤ 10%)을 만족해 8회로 확장하지 않았습니다.
이후 반복 수는 memory phase와 같은 **3회로 통일**했습니다(`--repeats` 기본값).
아래 표의 5회 결과를 앞 3회만으로 다시 median을 내면 sync `2.3697s`(+0.57%), async `1.2824s`(-1.41%)로 rMAD는 1.06%·2.49%에 머물러, 나머지 2회가 결론을 바꾸지 않았기 때문입니다.

| Variant | Run 수 | Checkpoint 크기(median) | Save 호출 시간(median) | Cold-buffered read(median) | rMAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| sync | 5 | 69.4 MiB | 2.356s | 3.12 GB/s | 0.57% |
| async | 5 | 69.4 MiB | 1.301s | 3.58 GB/s | 1.62% |

LoRA rank 8에서 저장 크기는 같고 async host-call은 약 1초 짧습니다.
이는 enqueue 반환 시간의 차이이며 전체 저장 완료 시간의 비교는 아닙니다.

#### Memory footprint

| 조건 | Run 수 | CUDA peak allocated(median) | Host memory pressure(median) |
| --- | ---: | ---: | ---: |
| `LEN-4096` (Megatron LoRA) | 3 | 39.5 GB | 53.7 GB |
| `LEN-8192` (Megatron LoRA) | 3 | 57.6 GB | 72.6 GB |
| `MEM-TRL-DDP` | 3 | 59.8 GB | 69.7 GB |
| `MEM-TRL-FSDP2` | 3 | 34.1 GB | 104.0 GB |
| `MEM-TRL-Z3-NVME` | 3 | 6.4 GB | 91.8 GB |

![30B CUDA memory footprint](figures/memory-footprint.svg)

Megatron LoRA에서 sequence length를 4096에서 8192로 늘리면 CUDA peak allocated가 39.5 GB에서 57.6 GB로 약 46% 증가했습니다.
같은 TRL LoRA 조건에서 FSDP2는 DDP보다 CUDA peak가 59.8 GB에서 34.1 GB로 약 43% 낮았습니다.
ZeRO-3 NVMe의 6.4 GB는 full fine-tuning에 runtime offload를 건 결과이므로 LoRA 조건과 직접적인 backend 우열로 비교하지 않습니다.

Unified-memory hardware이므로 CUDA와 host 측정값을 더하지 않고 별도 panel로 봅니다.
`EST-MEG-ADAM`(Megatron full + Adam) 추정값은 rank당 160.8 GiB로 119 GiB 예산을 초과해 실행하지 않았습니다.
`EST-MEG-SGD`(full + SGD)는 96.6 GiB로 예산 안에 들어오지만 pilot 실행은 별도로 결정하지 않았습니다(추정만 수행).

> **`MEM-TRL-Z3-NVME`의 optimizer 표기 정정(2026-09-12).**
> 이 조건은 `OPTIMIZER=sgd`로 실행됐지만, DeepSpeed는 optimizer state를 offload하면 client optimizer를 `DeepSpeedCPUAdam`으로 교체합니다(`deepspeed/runtime/engine.py`는 다른 client optimizer를 `zero_force_ds_cpu_optimizer` 기본값에서 거부).
> 실제 저장된 optimizer state를 열어 확인한 결과 `exp_avg`·`exp_avg_sq`(Adam 모멘트)와 `betas`·`eps`가 들어 있었습니다 — SGD(momentum=0)라면 state가 없어야 합니다.
> 따라서 위 6.4 GB / 91.8 GB는 **SGD가 아니라 Adam 상태를 NVMe로 offload한 실행의 값**이며, 설정 파일은 실제 동작에 맞춰 `adamw`로 고쳤고 이 조합은 이제 [검증 단계에서 거부](backends/trl.md#configure-training)됩니다.

`LEN-1024`는 2048-cohort의 길이 초과 행에서 실패해 sweep을 2048~8192로 변경했습니다.

DeepSpeed의 Direct offload traffic과 buffered ZeRO-checkpoint traffic을 분리해 보고하는 항목은 아직 측정하지 않았습니다.

### LoRA trainable-ratio가 checkpoint I/O에 미치는 영향

위 LoRA rank 8(약 0.03%) 측정에 이어, sync 파이프라인에서 `LORA_DIM`만 바꿔 비율의 영향을 확인했습니다.

Megatron은 학습 시작 시 rank-local model-parallel shard 기준 trainable parameter 수와 비율을 로그로 남깁니다.
`lora_dim=8`에서 trainable 5,111,808 / shard 16,041,719,808 = 0.0319%이고, LoRA는 `target_modules`에 rank당 `r × (in_dim + out_dim)`을 더하므로 trainable 수는 `lora_dim`에 정확히 비례합니다.
따라서 rank 1당 638,976이고 목표 비율 `R`은 `lora_dim = round(R × 16,041,719,808 / 638,976)`으로 역산합니다.
비율은 global 30.52B가 아니라 Megatron이 실제로 로그에 남기는 rank-local shard 기준입니다.

| Variant | `LORA_DIM` | 역산 비율 | Checkpoint 크기(median) | Save 호출 시간(median) | rMAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ratio-0.1pct` | 25 | 0.0996% | 224.9 MB | 1.44s | 2.10% |
| `ratio-0.5pct` | 126 | 0.5019% | 1,128.4 MB | 2.03s | 0.13% |
| `ratio-1pct` | 251 | 0.9998% | 2,246.6 MB | 2.61s | 0.60% |

![LoRA parameter ratio and checkpoint I/O](figures/lora-ratio-checkpoint.svg)

목표 비율 0.1%에서 1.0%로 약 10배 늘리면 checkpoint 크기도 224.9 MB에서 2,246.6 MB로 9.99배 증가했습니다.
반면 같은 구간에서 save 시간은 1.44초에서 2.61초로 1.81배만 늘었습니다 — 크기는 10배인데 시간은 1.8배이므로, 작은 checkpoint일수록 고정 비용(메타데이터 기록·rank 동기화)이 시간을 지배하고 커질수록 byte당 처리율이 좋아진다는 뜻입니다.
이 값은 `CHECKPOINT_MODE=sync`에서 데이터 파일 `fsync()`까지 포함해 반환되는 blocking `save()` 호출 시간을 rank별로 재고 가장 느린 rank를 취한 것입니다(async의 enqueue 반환 시간과 달리 실제 쓰기를 포함하지만, 부모 디렉터리 `fsync()`는 빠져 있어 장애 durability를 뜻하지는 않습니다).

12/12 run 통과(pilot 3 + 측정 9), 세 조건 모두 rMAD가 10% 기준을 크게 밑돌아 8회로 확장하지 않았습니다.

Checkpoint 크기 비는 5.02배·9.99배로 `LORA_DIM` 비 5.04배·10.04배에 근접했습니다.
`ratio-0.1pct` pilot 로그도 역산과 일치했습니다: `Trainable parameters: 15,974,400`, `Trainable percentage: 0.10%`.

이 조건들은 plan만 바꿔 실행합니다.

```bash
python experiments/checkpoint_memory_30b.py \
  --setup setups/spark/local-checkpoint-memory-30b.json \
  --phase checkpoint \
  --plan experiments/megatron/lora-ratio-checkpoint-io.json \
  --output results/lora-ratio-checkpoint-io \
  --repeats 3 --execute
```

## Qwen vs GLM: 방법론이 일반화되는가

여기까지의 모든 반복 측정(checkpoint I/O, memory footprint, LoRA ratio)은 Qwen3-30B-A3B 하나에서만 실행됐습니다.
GLM-4.7-Flash는 [30B GPU Results](#30b-gpu-results)에서 Megatron 1-step 실행 가능성만 확인됐고, TRL은 `spark_config.py`·`spark_train.py`에 `glm4_moe_lite` family와 전용 LoRA target module이 이미 있었지만 **한 번도 실행된 적이 없었습니다.**
2026-09-12에 `--model glm`으로 같은 세 실험을 그대로 재실행해 방법론이 일반화되는지 확인했습니다(모두 3회 반복으로 통일).

### 실험 1: TRL DDP LoRA — 처음 실행되는 경로

`experiments/trl/glm-4.7-flash-30b-lora.json`(no_robots, MAX_STEPS=1)으로 처음 실행했습니다.

| Stage | eval_loss | Peak allocated |
| --- | --- | --- |
| base | 1.926144 | 57.699 GiB |
| train | 1.928212 | 57.843 GiB |
| tuned | 1.928212 | 57.699 GiB |

`train`과 `tuned`의 eval_loss가 소수점까지 일치 — Qwen Case 1과 같은 adapter 재로딩 신뢰성 증거입니다.
Peak allocated는 Qwen TRL DDP LoRA(58.825 GiB)보다 **약 1.7% 낮았습니다.**

### 실험 2: Memory footprint — 같은 방향, 다른 기울기

| 조건 | Qwen CUDA / host | GLM CUDA / host | CUDA 차이 |
| --- | ---: | ---: | ---: |
| `LEN-4096`(Megatron LoRA) | 39.5 / 53.7 GB | 34.54 / 48.65 GB | −12.6% ⚠ |
| `LEN-8192`(Megatron LoRA) | 57.6 / 72.6 GB | 38.94 / 57.30 GB | −32.4% ⚠ |
| `MEM-TRL-DDP` | 59.8 / 69.7 GB | 58.24 / 109.50 GB | −2.6% (host 값은 아래 참고, 모델 신호 아님) |
| `MEM-TRL-Z3-NVME` | 6.4 / 91.8 GB | 7.27 / 85.88 GB | +13.6% |
| `MEM-TRL-FSDP2` | 34.1 / 104.0 GB(통과) | **3/3 실패**(아래) | — |

⚠ 두 `LEN-*` 행은 attention backend가 서로 다릅니다(Qwen `local`, GLM `transformer_engine`). 모델 차이로 읽으면 안 되며, 통제된 비교는 [아래 대조 실험](#sequence-length-기울기)을 따릅니다.

<a id="sequence-length-기울기"></a>

**Sequence length 기울기 차이는 모델이 아니라 attention 구현 때문입니다.**
표면적으로 Qwen은 4096→8192에서 CUDA peak가 +45.8%(39.5→57.6 GB), GLM은 +12.7%(34.54→38.94 GB)만 증가해 아키텍처 차이처럼 보입니다.
하지만 두 측정은 **애초에 같은 attention 경로가 아니었습니다** — `megatron_lab/config.py`의 `select_transformer_impl()`은 `TRANSFORMER_IMPL=auto`를 family별로 다르게 해석합니다.

```python
selected = ("transformer_engine" if requested == "auto" and family == "glm4_moe_lite"
            else "local" if requested == "auto" else requested)
```

즉 Qwen은 `local`, GLM은 `transformer_engine`으로 측정됐습니다(GLM은 MLA provider 제약으로 `local`을 아예 거부합니다).
Qwen만 backend를 바꿔 같은 조건으로 다시 측정한 결과입니다.

| 조건 | 4096 | 8192 | 기울기 |
| --- | ---: | ---: | ---: |
| Qwen + `local`(기존 측정) | 39.5 GB | 57.6 GB | **+45.8%** |
| Qwen + `transformer_engine`(대조 실험) | 36.85 GB | 41.51 GB | **+12.6%** |
| GLM + `transformer_engine` | 34.54 GB | 38.94 GB | **+12.7%** |

**같은 backend에서 두 모델의 기울기는 +12.6% vs +12.7%로 사실상 동일합니다.**
따라서 원래 세웠던 "GLM의 MLA가 KV를 압축해 완만하다"는 가설은 **기각됩니다** — 차이를 만든 것은 Megatron `local` 경로가 attention 행렬을 materialize해 sequence length에 제곱으로 증가하는 항을 남기는 반면, Transformer Engine의 fused attention은 그렇지 않다는 점입니다.

이 비교에서 배운 운영상의 교훈은 모델 비교 자체보다 큽니다: `auto`처럼 **입력에 따라 조용히 다른 구현을 고르는 설정은 A/B 비교의 통제 변수를 깨뜨립니다.**
아래 memory footprint 표의 `LEN-*` 행도 Qwen은 `local`, GLM은 `transformer_engine` 측정이므로 두 값을 모델 차이로 읽으면 안 됩니다.

**TRL DDP의 host memory pressure 차이는 모델 차이가 아니라 측정 방법의 문제였습니다.**
처음에는 CUDA peak가 거의 같은데(59.8 vs 58.24 GB) host pressure만 GLM이 57% 높다고(109.50 vs 69.7 GB) 기록했습니다.
통제된 재측정에서 이 결론은 **철회됩니다.**

먼저 로딩 자체를 격리해 CPU로만 모델을 올리며 RSS를 0.2초 간격으로 샘플링했습니다(`AutoModelForCausalLM.from_pretrained`, GPU 미사용).

| | 로드된 parameter | peak RSS | parameter 대비 |
| --- | ---: | ---: | ---: |
| Qwen | 56.87 GiB | 111.02 GiB | **×1.95** |
| GLM | 55.77 GiB | 105.47 GiB | **×1.89** |

**두 모델의 로딩 비용은 사실상 같습니다**(둘 다 가중치의 약 2배). 따라서 "GLM만 두 벌을 쓴다"는 해석은 성립하지 않습니다.

그다음 Qwen DDP를 오늘 같은 조건으로 다시 돌려 host pressure의 rank별 분포를 봤습니다.

| 조건 | rank별 host pressure | 편차 |
| --- | --- | ---: |
| GLM(3 run × 2 rank) | 99.3 / 103.0 / 104.0 / 117.0 / 117.6 / 118.4 GB | 19.1 GB |
| Qwen(오늘 재측정, 2 rank) | 69.1 / **122.3** GB | **53.2 GB** |

같은 모델·같은 run 안에서도 두 rank가 69 GB와 122 GB로 갈렸고, Qwen의 최대값이 오히려 GLM보다 높습니다.
즉 이 지표는 노드 상태에 크게 흔들려 **모델을 구분하는 근거로 쓸 수 없습니다.**
원래의 57% 격차는 GLM은 rank 최대값, Qwen은 다른 세션에서 측정된 값을 비교한 데서 생긴 것이었습니다.

조사 과정에서 배제한 후보도 함께 남깁니다.

- **모델 크기 아님.** safetensors 헤더 실측으로 GLM 58.2 GiB(~31.2B), Qwen 56.9 GiB(~30.5B)로 오히려 GLM이 2.3% 큽니다. GLM의 `model.safetensors.index.json`은 `total_size`를 실제의 정확히 절반인 29.1 GiB로 기록하므로 **이 필드를 메모리 추정에 쓰면 2배 틀립니다.**
- **CUDA allocator 단편화 아님.** reserved−allocated 격차가 두 모델 모두 정확히 0.54 GiB입니다.
- **shard 분할 아님.** GLM은 레이어별 expert 텐서가 정확히 1개 shard에 모여 있고(median 1, max 1) Qwen은 최대 2개로 오히려 더 흩어져 있습니다.
- **dtype 변환 아님.** 두 모델 모두 파일과 목표 dtype이 BF16으로 같습니다.
- **expert fusion 자체도 아님.** 두 모델 모두 디스크에는 per-expert 텐서로 저장되고 메모리에서는 fused 3D 파라미터(`experts.gate_up_proj`)를 쓰므로, 조립 비용은 양쪽 다 발생합니다.

남은 유력 후보는 **shard 분할 방식**입니다 — GLM은 48 shard(평균 1.2 GiB), Qwen은 16 shard(평균 3.6 GiB)로 저장돼 있어, 하나의 fused expert 파라미터를 완성하는 데 동시에 살아 있어야 하는 source 버퍼 수가 다릅니다.
다만 이는 위 배제 결과에서 좁혀낸 **가설이며 직접 계측하지 않았습니다.**

**FSDP2는 정도가 아니라 종류가 다른 실패입니다.** Qwen은 34.1 GB로 통과하지만 GLM은 3회 모두 같은 지점에서 실패했습니다:

```
accelerate/utils/fsdp_utils.py:543, fsdp2_load_full_state_dict()
AttributeError: 'Tensor' object has no attribute 'device_mesh'
```

**원인은 persistent buffer입니다.** 두 모델을 meta device에 올려 accelerate와 같은 순서로 `fully_shard`를 적용한 뒤 `state_dict()`에서 `DTensor`가 아닌 항목을 센 결과입니다.

| 모델 | 감싼 decoder layer | non-DTensor 항목 |
| --- | ---: | --- |
| Qwen3-30B-A3B | 48 | **0개** |
| GLM-4.7-Flash | 47 | **46개** — 전부 `model.layers.N.mlp.gate.e_score_correction_bias` (shape 64) |

`modeling_glm4_moe_lite.py:376`이 MoE 라우터의 expert-score correction bias를 `register_buffer(...)`로 등록하는데 `persistent=False`가 없어 `state_dict()`에 포함됩니다.
FSDP2의 `fully_shard`는 `nn.Parameter`만 샤딩하고 buffer는 평범한 텐서로 남기는데, accelerate(`fsdp_utils.py:543`)는 `state_dict()`의 모든 항목이 `DTensor`라고 가정하고 `.device_mesh`를 읽습니다.
Qwen3 MoE가 통과하는 이유도 같은 지점에서 설명됩니다 — 이 모델이 등록하는 buffer는 `inv_freq` 계열뿐이고 전부 `persistent=False`라 `state_dict()`에 아예 들어가지 않습니다.

즉 이것은 GLM의 결함이 아니라 **persistent buffer를 가진 모델 전반에 적용되는 accelerate FSDP2 경로의 가정 오류**이며, MoE·MLA 여부와는 무관합니다.
`first_k_dense_replace: 1`이라 MoE 레이어가 47개 중 46개인 것과 실패 항목 46개가 정확히 일치합니다.
**우회를 실제로 시도했고, 두 번째 장벽이 나왔습니다.**
문제의 `fsdp2_load_full_state_dict()`는 accelerate에서 `cpu_ram_efficient_loading`이 켜져 있을 때만 호출되므로, 이 플래그를 꺼서 해당 경로를 건너뛰어 봤습니다.
첫 장벽은 실제로 사라져 처음으로 `accelerator.prepare`를 통과하고 가중치 로딩까지 끝냈지만, 곧바로 다음에서 실패했습니다.

```
fsdp2_prepare_model → fully_shard → _move_states_to_device → tensor.to(device)
torch.OutOfMemoryError: 119.69 GiB 중 659 MiB만 남은 상태에서 768 MiB 할당 실패
(해당 process가 73.56 GiB 사용 중)
```

이 플래그를 끄면 **샤딩하기 전에** 각 모듈의 전체 가중치를 device로 올리는데, 이는 `cpu_ram_efficient_loading`이 애초에 막으려던 동작입니다.
즉 버그 하나를 메모리 폭발과 맞바꾸는 셈이라 119 GiB unified memory에서는 쓸 수 없습니다.

정리하면 이 조합에는 **독립적인 장벽이 두 개**이고, 둘 다 이 하드웨어에서는 넘을 수 없습니다.

| 경로 | 결과 |
| --- | --- |
| 기본값(`cpu_ram_efficient_loading` on) | accelerate의 DTensor 가정이 persistent buffer에서 깨짐 — 3/3 실패 |
| 우회(`cpu_ram_efficient_loading` off) | 샤딩 전 전체 가중치를 device로 이동하다 CUDA OOM |

남은 선택지는 accelerate 상류 수정, 또는 모델의 buffer를 `persistent=False`로 바꾸는 것입니다.
후자는 `cpu_ram_efficient_loading` 경로에서 rank 0만 실제 가중치를 읽으므로 다른 rank가 이 라우터 bias를 못 받아 **조용히 다른 routing 결과를 낼 위험**이 있어 채택하지 않았습니다.
이 저장소에서 GLM을 쓸 때는 DDP 또는 DeepSpeed를 사용합니다.

### 실험 3: Checkpoint I/O — 같은 방향(async 우위), 다른 배수

| Variant | Qwen(3회 재계산) | GLM(3회) | 비율 |
| --- | ---: | ---: | ---: |
| sync save | 2.3697s | 3.1121s | ×1.31 |
| async save | 1.2824s | 1.5805s | ×1.23 |
| Checkpoint 크기 | 72.8 MB | 150.2 MB | ×2.06 |

두 모델 모두 **async가 sync보다 빠르다는 방향은 같습니다**(Qwen 1.85배, GLM 1.97배 — 격차도 비슷한 크기).
rMAD는 GLM sync 0.31%, async 0.90%로 10% 기준을 크게 밑돌아 3회로 충분했습니다.

**Checkpoint 크기 2.06배는 우연이 아니라 LoRA target module 구성 차이입니다.**
`backends/megatron/megatron_lab/config.py`는 family별로 다른 target module을 씁니다.

```python
target_modules = (
    ["linear_q_down_proj", "linear_q_up_proj", "linear_kv_down_proj", "linear_kv_up_proj", "linear_proj"]
    if spec.family == "glm4_moe_lite"
    else ["linear_qkv", "linear_proj"]
)
```

Qwen은 fused QKV 1개 + proj 1개(2종류), GLM은 MLA의 Q/KV down·up projection 4개 + proj 1개(5종류)입니다.
같은 `lora_dim=8`에서 실제 로그도 이를 뒷받침합니다: Qwen `Trainable parameters: 5,111,808`(0.0319%), GLM `Trainable parameters: 10,515,968`(0.07%) — 비율 **2.057배**로 checkpoint 크기 비율(2.063배)과 거의 일치합니다.
[LoRA trainable-ratio 실험](#lora-trainable-ratio가-checkpoint-io에-미치는-영향)에서 확인한 "checkpoint 크기는 trainable parameter 수에 선형 비례한다"는 관계가 **모델이 달라도 유지**되며, 다만 같은 `lora_dim`에서의 절대 배수는 target module 구성에 따라 달라진다는 뜻입니다.

### 종합: 일반화되는 것과 안 되는 것

| 일반화됨(정성적으로 같은 방향) | 일반화 안 됨(모델·아키텍처 의존) |
| --- | --- |
| Async checkpoint가 sync보다 빠름 | ~~Sequence length 증가율~~ → 같은 backend에서는 동일(+12.6% vs +12.7%), 차이는 모델이 아니라 attention 구현이었음 |
| Checkpoint 크기 ∝ trainable parameter 수(선형) | 같은 `lora_dim`에서의 절대 trainable parameter 수(target module 구성 차이) |
| Train/tuned eval_loss 일치 → adapter 재로딩 신뢰성 | FSDP2 사용 가능 여부 — 원인은 모델이 아니라 persistent buffer를 만난 accelerate 경로이며, 우회도 OOM으로 막힘 |
| 가중치 로딩 비용(둘 다 parameter의 약 1.9배) | — |

[Scaling Estimates](#scaling-estimates-100b-to-1t)의 parameter당 고정 비용(weight 2 + gradient 2 + Adam 12 bytes)은 optimizer·저장 방식에서 나오므로 모델 아키텍처와 무관하게 일반화될 가능성이 높습니다.
Activation 비용도 **backend를 고정하면 두 모델이 같은 기울기**를 보였으므로(+12.6% vs +12.7%), 아키텍처보다 attention 구현이 지배적인 변수입니다.
남는 모델 의존 항목은 LoRA target module 구성에 따른 trainable parameter 배수와 가중치 로딩 시 host memory 배수이며, 이 둘은 새 모델마다 실측해야 합니다.

이번 조사에서 가장 재사용성이 높은 교훈은 방법론 쪽입니다 — **모델 A에서 관측한 차이를 모델 B의 속성으로 귀속하기 전에, 두 실행이 정말 같은 구현 경로였는지 먼저 확인해야 합니다.** 여기서는 `TRANSFORMER_IMPL=auto`가 모델별로 다른 backend를 조용히 선택하고 있었고, 그 사실을 확인하기 전까지는 backend 차이가 아키텍처 차이로 보였습니다.

## Scaling Estimates: 100B to 1T

30B 실측을 근거로 중대형·1T 규모에서 GPU memory와 checkpoint 크기가 어떻게 변하는지 **추정**합니다.
전부 계산값이며 실행으로 확인한 값이 아닙니다 — 이 저장소가 실제로 돌린 최대 규모는 30B입니다.

### 계산 근거와 그 검증

BF16 학습의 parameter당 고정 비용은 다음과 같습니다.

| 항목 | parameter당 bytes | 적용 대상 |
| --- | ---: | --- |
| Weights (BF16) | 2 | 전체 parameter |
| Gradients (BF16) | 2 | trainable parameter |
| Adam state (fp32 master + `exp_avg` + `exp_avg_sq`) | 12 | trainable parameter |

이 12 bytes/param이 맞는지는 실측으로 확인됐습니다.
Qwen3-30B-A3B(30.5B)를 2 rank로 나누면 rank당 15.25B parameter이고, 여기에 12 bytes를 곱하면 **170.4 GiB**입니다.
같은 조건에서 실제로 디스크에 쓰인 `global_step1/offloaded_tensors`는 **171 GiB**로 **0.3% 차이**였습니다.
즉 아래 표의 곱셈은 최소한 optimizer state 항목에 대해서는 실측에 맞춰져 있습니다.

### 규모별 추정

Full fine-tuning은 `2+2+12 = 16 bytes/param`, LoRA(trainable 0.1% 가정)는 frozen weight 2 bytes/param에 trainable 몫만 더합니다.

| 모델 규모 | Full FT 학습 state | LoRA(0.1%) 학습 state | Checkpoint: weights만 | Checkpoint: full state | Checkpoint: LoRA adapter |
| --- | ---: | ---: | ---: | ---: | ---: |
| 30B (실측 anchor) | 0.44 TiB | 57.3 GiB | 56.8 GiB | 0.39 TiB | 0.40 GiB |
| 100B | 1.46 TiB | 187.8 GiB | 186.3 GiB | 1.27 TiB | 1.30 GiB |
| 500B | 7.28 TiB | 938.8 GiB | 931.3 GiB | 6.37 TiB | 6.52 GiB |
| 1T | 14.55 TiB | 1.83 TiB | 1.82 TiB | 12.73 TiB | 13.04 GiB |

학습 state를 device memory에만 담는다고 가정했을 때 필요한 GPU 수(활성화·통신 버퍼 제외한 하한):

| 모델 규모 | DGX Spark (119 GiB) | H100 (80 GB) | H200 (141 GB) |
| --- | ---: | ---: | ---: |
| 30B | 3.8 | 6.1 | 3.5 |
| 100B | 12.5 | 20.0 | 11.3 |
| 500B | 62.6 | 100.0 | 56.7 |
| 1T | 125.2 | 200.0 | 113.5 |

### 읽는 방법

- **Tuning 방법이 checkpoint 크기를 3자리수 바꿉니다.** 1T에서 full-state checkpoint는 12.73 TiB, 같은 모델의 LoRA adapter는 13.04 GiB로 약 1,000배 차이입니다. 저장 주기·보존 개수·복구 시간 설계는 모델 크기보다 tuning 방법에 먼저 좌우됩니다.
- **Optimizer가 full FT 비용의 75%입니다.** 16 bytes 중 12가 Adam state이므로, optimizer state를 어디에 두느냐(device / CPU / NVMe)가 "이 모델이 들어가는가"를 사실상 결정합니다. 30B에서 NVMe offload로 CUDA peak가 6.4 GB까지 내려간 것이 이 구조 때문입니다.
- **1T full FT는 이 계산만으로도 100 GPU급입니다.** 위 표는 활성화·통신 버퍼·CUDA context를 뺀 하한이므로 실제로는 더 듭니다. 반면 1T LoRA의 학습 state는 1.83 TiB로, 대부분이 frozen weight라 offload·sharding 전략의 선택지가 훨씬 넓습니다.
- **Sequence length는 위 표에 없습니다.** 30B LoRA 실측에서 length를 4096 → 8192로 늘렸을 때 CUDA peak가 39.5 GB → 57.6 GB(약 +46%)였습니다. 이 증가분은 parameter 수와 무관한 activation 비용이라 위 곱셈에 포함되지 않습니다.

### 이 추정이 담지 않는 것

Activation(길이·batch·recompute 정책에 좌우), MoE routing 내부 버퍼, CUDA context와 allocator 단편화, NCCL 통신 버퍼, framework workspace는 모두 빠져 있습니다.
[메모리 추정기](#estimate-memory-before-running)가 30B 실측 대비 최대 -11%로 낮게 나온 것도 같은 이유이며, 규모가 커질수록 이 누락분의 절대값도 함께 커집니다.
따라서 위 숫자는 **하한이자 규모 감각**이며, 특정 구성이 실제로 들어가는지는 해당 규모에서 실행해 확인해야 합니다.
이 앵커 자체도 Qwen3-30B-A3B 하나에서만 검증됐습니다 — [Qwen vs GLM 비교](#qwen-vs-glm-방법론이-일반화되는가)에서 weight·optimizer 비용과 sequence length 기울기는 모델이 달라도 같았지만(후자는 attention backend를 맞췄을 때), LoRA target module 배수와 로딩 시 host memory 배수는 모델마다 달랐습니다.

## Repeated Megatron Measurements

`experiments/benchmarks.py`는 [benchmark plan](../experiments/megatron/benchmark-plan.json)의 cell·variant를 읽어 Qwen2.5-0.5B와 No Robots를 고정한 A/B 비교를 수행합니다.
여기서 얻은 시간·메모리 차이를 30B 성능으로 일반화하지 않습니다.

기본 계획은 8개 cell, variant별 4회 측정과 별도 warmup입니다.
길이 비교는 `MAX_LENGTH` 상한만 바꾸지 않고 고정 길이 padding을 씁니다.
Selective recompute는 현재 설정 오류로 warmup에서 실패하므로 full/selective 비교가 완결된다고 기대하지 않습니다.

```bash
python experiments/benchmarks.py \
  --setup setups/spark/local.json \
  --output results/benchmarks-16x2 \
  --steps 16 --repeats 2 --within-run-warmup 4 \
  --checkpoint-intervals 4 8 \
  --execute
```

이 축소 계획은 warmup 포함 48개 run을 선택하고 첫 4개 within-run step을 측정에서 제외합니다.
Cell 이름의 checkpoint interval 16/32는 유지되지만 실제 interval은 인자의 4/8이므로 run config를 기준으로 해석합니다.
`--plan`으로 계획을, `--timeout`으로 개별 실행 제한을 바꿀 수 있고 실패한 variant의 후속 반복은 생략될 수 있습니다.
`--execute` 없이 같은 인자로 먼저 계획을 확인합니다.

## Read the Outputs

출력에는 계획·실행 설정, manifest, rank 로그, measurement JSONL과 `records.jsonl`이 남습니다.
기록은 성공·실패·생략을 포함하며 정상 종료·예상 step 수·finite 값 검사를 통과한 measured run만 비교합니다.
Warmup 성공은 측정 반복 수에 포함하지 않습니다.

해석할 때 구분할 것:

- Whole-run 시간은 원격 시작·초기화·평가·정리를 포함합니다.
- CUDA allocator peak는 전체 장치 사용량이 아닙니다.
- Checkpoint save와 blocking finalization은 host 호출 시간이며 native step timer와 별도입니다.
- Async background I/O의 자원 경쟁이 학습 시간에 영향을 줄 수 있습니다.

반복 결과는 각 실행의 manifest와 `records.jsonl`에서 판정하며 과거 run log를 저장소에 복제해 보관하지 않습니다.
