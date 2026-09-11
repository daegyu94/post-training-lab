# Experiments

Experiment 파일은 모델·데이터 revision과 학습 조건을, setup은 실행할 노드와 경로를 선언합니다.
공통 runner가 둘을 결합하며 `--execute`가 있을 때만 SSH 학습을 시작합니다.
첫 실행은 [Getting Started](getting-started.md), 설정 책임은 [Architecture](architecture.md)를 따릅니다.

현재 목표는 두 노드에서 30B 모델의 SFT 실행 경로를 확인하는 것입니다 — `Qwen/Qwen3-30B-A3B`는 TRL LoRA와 Megatron MoE로, `zai-org/GLM-4.7-Flash`는 Megatron MoE로 확인합니다.
30B 실행에서는 모델별 준비, 메모리 배치, 2노드 통신, checkpoint와 재로딩을 봅니다.
0.5B preset은 이 목표를 대신하는 실험이 아니라 runner·전처리·저장·재로딩 같은 기본 동작을 싸게 검사하는 smoke이며, 두 결과는 서로 다른 범위로 해석합니다.

이 문서는 preset·도구 사용법부터 30B와 반복 측정에서 실제로 얻은 수치까지 한곳에서 관리합니다.
정적 검사, CPU 테스트, dry-run, 실제 GPU 실행이 서로 다른 증거라는 원칙은 [AGENTS.md](../AGENTS.md)를, 개별 run 판정 기준은 [Getting Started](getting-started.md#6-verify-the-result)를 따릅니다.

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
| `experiments/megatron/qwen3-30b-full.json` | 2노드 Qwen3 30B MoE full-parameter — 아래 참고, 메모리 적합성 미검증 | all |

앞의 smoke 네 개는 Qwen2.5-0.5B와 고정 No Robots 입력을 쓰며 모델 품질·장기 수렴·성능 비교용이 아닙니다.
Megatron의 두 30B preset은 검증 당시와 같은 TP=1·PP=1·EP=2로 1 step, 평가 1회와 async checkpoint를 실행합니다.
TRL NVMe preset의 `train`은 같은 process에서 평가하지 않고, 평가는 `tuned`를 별도 process로 실행해 native ZeRO checkpoint를 새 엔진에 복원한 뒤 수행합니다 — 이유와 준비 조건은 [30B NVMe 실습](../labs/nvme-30b/README.md)을 따릅니다.

`qwen3-30b-full.json`은 **실측으로 검증되지 않았습니다.** 로컬 `no_robots` 5000행 중 한 행이 검증된 LoRA preset과 같은 `MAX_LENGTH=2048`을 넘어 "never truncates" 정책에 막혔습니다 — 이 경계값은 memory 비교의 기준이라 늘리지 않았습니다. [메모리 추정기](#estimate-memory-before-running)는 per-rank 149.7 GiB(파라미터 29.9 + gradient 29.9 + Adam 89.6)로 예측해 119 GiB 예산을 약 31 GiB 초과한다고 봅니다. `--optimizer sgd`는 90.0 GiB로 예산 안에 들어옵니다. 둘 다 추정이며 실행으로 확인한 값이 아닙니다.

`experiments/run.py`는 `--backend`, `--setup`, `--experiment`, `--output`을 요구합니다.
기본은 dry-run이고 `--timeout`은 기본 900초의 양의 정수이며 기존 출력 디렉터리는 재사용할 수 없습니다.
모든 preset은 `MODEL_ID`, `MODEL_REVISION`, `DATASET_ID`, `DATASET_REVISION`을 명시해야 하고, 백엔드가 지원하는 모든 CLI 옵션이 runner에 노출된 것은 아닙니다.
Runner는 output 디렉터리 이름을 `OBSERVATORY_RUN_ID`로 예약해 framework metric과 host telemetry를 같은 실행에 연결합니다.

## Estimate Memory Before Running

GPU 시간을 쓰기 전에 "이 구성이 메모리에 들어가는가"를 먼저 계산합니다.
파라미터 크기는 아키텍처별 수식이 아니라 snapshot의 `*.safetensors` 헤더에서 **실제 텐서 바이트**를 읽으므로 MoE·dense 구분이나 새 아키텍처 대응에 코드 변경이 필요 없습니다.
Expert 가중치는 텐서 이름으로 식별해 expert parallel sharding만 따로 적용합니다.

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

**이 값은 추정이지 측정이 아닙니다.**
CUDA context, allocator 단편화, framework workspace를 모델링하지 않아 실측보다 **낮게** 나오는 경향이 있습니다(위 표에서 최대 -11%).
여유가 빠듯하게 `FITS`로 나오면 실제로는 안 들어갈 수 있다고 봐야 합니다.

## Build an Experiment from Knobs

매번 새 JSON을 손으로 쓰는 대신 `experiments/build.py`가 backend·dataset·model·offload·epoch 같은 knob을 스키마로 매핑해 `experiments/generated/<output 이름>.json`에 쓰고, 그 파일을 그대로 `run.py`에 넘깁니다.
검증·SSH 실행 로직은 재사용하며 새로 만들지 않습니다.

```bash
python experiments/build.py \
  --backend trl --dataset ultrachat \
  --model-id Qwen/Qwen2.5-0.5B-Instruct --model-revision <40-hex> \
  --setup setups/spark/local.json --output results/ultrachat-epoch1 \
  --epochs 1 --train-samples 256 --eval-samples 32
```

- 기본값은 실제 2노드 클러스터에 맞춰져 있습니다(`--nnodes 2`).
- `--dataset` 선택지는 고정 목록이 아니라 `datasets_lab.public_data.PRESETS`에서 읽습니다(`reference_only` 제외).
- `--epochs`와 `--max-steps`는 배타적입니다. TRL은 `spark_train.py`가 HF `SFTConfig`로 직접 처리합니다. Megatron은 step 기반 scheduler라 `build.py`가 `steps = ceil(epochs * train_count / global_batch_size)`를 미리 계산하는데, 이는 dataset `manifest.json`을 controller에서 읽을 수 있을 때만 가능합니다 — `data_dir`가 node-local인 현재 구성에서는 **Megatron의 `--epochs`는 항상 오류로 멈추므로 `--max-steps`를 직접 씁니다.**
- `--offload {none,cpu,nvme}`는 TRL 전용입니다. `cpu`/`nvme`는 `--distributed-backend deepspeed`를 강제하고 해당 DeepSpeed 설정을 선택하며, NVMe profile은 `--finetuning-mode full`이 필요합니다. `--backend megatron`과 함께 쓰면 즉시 오류입니다.
- Megatron 전용 `--tp`/`--pp`/`--ep`/batch 옵션은 검증된 30B preset 기본값(TP=1, PP=1, EP=2)을 그대로 씁니다. 각 옵션의 뜻은 `--help`에서 확인합니다.
- Dataset 준비는 `build.py`가 대신 실행하지 않습니다 — Hub 다운로드 같은 부수효과를 조립 단계에 숨기지 않기 위해서이며 절차는 [Datasets](datasets.md)를 따릅니다.

### 실제로 확인한 조합

`build.py`가 다양한 조합에서 실제로 다른 결과를 내는지 `--execute`로 확인한 네 가지입니다.

| Case | Backend | Model | Dataset | knob | 결과 |
| --- | --- | --- | --- | --- | --- |
| 1 | TRL DDP | Qwen2.5-0.5B-Instruct | ultrachat (256/32) | `--epochs 1` | base/train/tuned 통과 |
| 2 | TRL DeepSpeed | Qwen3-30B-A3B | no_robots | `--offload nvme --max-steps 1` | train 통과, 별도 `tuned` 평가 통과 |
| 3 | Megatron | Qwen3-30B-A3B | self_oss (40,000/8,000) | `--max-steps 1 --stage all` | base `1.402232` → tuned `1.342737` |
| 4 | Megatron | GLM-4.7-Flash | ultrachat (160,000/30,000) | `--max-steps 1 --max-length 4096 --stage all` | base `2.269922` → tuned `2.012836` |

**Case 1**은 `--epochs`가 실제로 동작하는지와 재로딩 신뢰성을 봅니다: `base` 1.4340 → `train` 1.4308 → `tuned` 1.4308.
`train`과 `tuned`의 eval_loss가 소수점까지 같다는 것은 저장된 adapter를 다시 읽어도 수치가 흔들리지 않는다는 증거입니다.
감소폭이 작은 건 256샘플·16 step만 학습했기 때문이며 이 값을 모델 품질 지표로 확대 해석하지 않습니다.

**Case 3·4**의 이전 실패 원인은 HF dataset이 아니라 공유 파일시스템을 전제한 `torch_dist` checkpoint를 node-local NVMe에 나눠 저장한 것이었습니다(rank 0에 metadata와 `__0_0.distcp`, rank 1에 `__1_0.distcp`만 남음).
Runner가 checkpoint를 NFS checkout 아래 두도록 고친 뒤 두 30B MoE 모델 모두 base→train→tuned 단일 실행과 iteration 1 재로딩을 통과했고, NaN·skipped iteration은 0, `checkpoint_reload_verified`도 `true`였습니다.
Case 4는 `--max-length` 기본값 2048에서 ultrachat 샘플 하나가 길이를 초과해 한 번 실패했고 4096으로 재실행해 통과했습니다 — preset마다 실제 대화 길이가 다르다는 실제 사례입니다.

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

실행 절차·측정 field·집계식·유효성 임계값은 모두 코드가 source of truth입니다 — 진입점 `experiments/checkpoint_memory_30b.py`, read/cache 분류 `experiments/checkpoint_io_probe.py`, cohort 생성 `experiments/prepare_checkpoint_cohort.py`.
아래는 코드에서 읽어낼 수 없는 것만 남깁니다: 왜 이렇게 측정했는지와 실제로 무엇이 나왔는지.

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

- **Async throughput의 분모로 `save()` 반환 시간을 쓰지 않습니다.** 그 값은 대부분 enqueue 작업만 나타냅니다. 첫 enqueue 시작부터 blocking finalization 완료까지를 씁니다.
- **전역 `drop_caches`를 쓰지 않습니다.** 공유 cluster이기 때문이며, 대신 파일별 `POSIX_FADV_DONTNEED`로 eviction을 요청하고 관찰된 device-read delta로 cache 상태를 분류합니다. `POSIX_FADV_DONTNEED`는 advisory이므로 eviction 성공을 가정하지 않고, read-ahead 때문에 physical bytes가 logical을 넘을 수 있어 분류 임계값은 정확한 cache-hit ratio가 아닙니다.
- **Direct I/O baseline은 storage microbenchmark입니다.** Megatron framework throughput으로 표시하지 않습니다.
- **Cohort는 결정적으로 선택한 소수 행입니다.** Megatron 전처리는 truncate하지 않고 제한을 넘는 행에서 중단하므로 canonical split 전체를 쓸 수 없습니다. 2048 token 이하 UltraChat 32행(`ultrachat-qwen3-30b-2048-v1`)을 모든 조건에서 같은 순서·seed로 재사용합니다. 이 sample 수는 checkpoint·peak memory 측정에는 충분하지만 학습 수렴이나 dataset 품질의 근거가 아닙니다.
- **Sequence length 효과는 framework 비교와 섞지 않습니다.** Megatron LoRA의 별도 fixed-padding sweep으로만 측정합니다. 긴 step은 async background I/O를 숨길 시간도 늘리므로 length sweep 결과로 sync/async 우열을 판단하지 않습니다.
- **측정할 수 없는 값은 zero가 아니라 `null`로 기록하고 invalid로 분류합니다.**
- **각 run의 checkpoint는 probe와 metric 수집 직후 삭제합니다**(`cleanup_checkpoints()`). 모든 iteration·반복의 checkpoint를 보존하면 local storage를 소진하기 때문이며, manifest·measurement record는 그대로 남습니다.

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
Memory phase는 4096/8192 Megatron pilot과 반복, TRL DDP/FSDP2 LoRA, TRL ZeRO-3 NVMe full-SGD를 실행합니다.
`LEN-2048` Megatron memory 값은 checkpoint phase 결과를 재사용합니다.

### 실측 결과

Raw manifest·measurement record는 커밋하지 않으므로(`results/`는 gitignore 대상) 아래는 요약값입니다.

#### Megatron local checkpoint (sync vs async)

5회 반복 모두 종료 기준(rMAD ≤ 10%)을 만족해 8회로 확장하지 않았습니다.

| Variant | Run 수 | Checkpoint 크기(median) | Save 호출 시간(median) | Cold-buffered read(median) | rMAD |
| --- | ---: | ---: | ---: | ---: | ---: |
| sync | 5 | 69.4 MiB | 2.356s | 3.12 GB/s | 0.57% |
| async | 5 | 69.4 MiB | 1.301s | 3.58 GB/s | 1.62% |

LoRA rank 8 고정이라 sync/async가 같은 크기를 저장하므로 크기 차이는 없고, async가 blocking host-call 기준 약 1초 빠릅니다 — enqueue만 하고 반환하는 async 설계와 일치합니다.

#### Memory footprint

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

### LoRA trainable-ratio가 checkpoint I/O에 미치는 영향

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

## Repeated Megatron Measurements

`experiments/benchmarks.py`는 [benchmark plan](../experiments/megatron/benchmark-plan.json)의 cell과 두 variant를 읽습니다.
이 반복 측정도 Qwen2.5-0.5B와 No Robots를 고정한 저비용 A/B 비교입니다 — 설정 하나의 영향만 비교하려면 모델·데이터를 고정해야 하고, 여덟 조건을 30B로 반복하는 비용도 피할 수 있습니다.
따라서 여기서 얻은 시간·메모리 차이를 30B 성능으로 일반화하지 않습니다.

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
