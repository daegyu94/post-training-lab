# Experiments

실험 파일은 모델·데이터 revision과 학습 조건을 선언하고 setup은 실행할 노드와 경로를 선언합니다.
공통 runner가 두 파일을 결합하며 `--execute`가 있을 때만 SSH 학습을 시작합니다.
첫 실행 명령은 [Getting Started](getting-started.md), 설정 책임은 [Architecture](architecture.md)를 따릅니다.

## 현재 프로젝트 목표

현재 목표는 Spark 두 노드에서 다음 30B 모델의 SFT 실행 경로를 확인하는 것입니다.

| 모델 | 확인할 경로 |
| --- | --- |
| `Qwen/Qwen3-30B-A3B` | TRL LoRA와 Megatron MoE |
| `zai-org/GLM-4.7-Flash` | Megatron MoE |

30B 실행에서는 모델별 준비, 메모리 배치, 2노드 통신, checkpoint와 재로딩을 확인합니다.
구체적인 지원 범위와 제한은 [TRL](backends/trl.md)과 [Megatron](backends/megatron.md) 문서에서 관리합니다.

아래 0.5B preset은 이 목표를 대신하는 실험이 아닙니다.
다운로드와 반복 실행 비용을 낮춰 runner, 데이터 전처리, 저장·재로딩 같은 기본 동작을 빠르게 검사하는 smoke입니다.
30B 통합 결과와 0.5B smoke 결과는 서로 다른 범위로 기록하고 해석합니다.

## Presets

| 파일 | 노드·학습 | 단계 |
| --- | --- | --- |
| `experiments/trl/single-node-smoke.json` | 1노드 DDP LoRA, 1 step | base/train/tuned |
| `experiments/trl/smoke.json` | 2노드 DDP LoRA, 1 step | base/train/tuned |
| `experiments/megatron/smoke.json` | 2노드 full SFT, 2 step | base/train/tuned |
| `experiments/megatron/resume-smoke.json` | 2노드 full SFT, 2→3 step | base/train/resume/tuned |
| `experiments/trl/nvme-offload-30b.json` | 2노드 Qwen3 30B full SFT, ZeRO-3 NVMe state offload | base/train/tuned |
| `experiments/megatron/qwen3-30b-lora.json` | 2노드 Qwen3 30B MoE LoRA, 평가와 node-local NVMe async checkpoint | train |
| `experiments/megatron/glm-4.7-flash-30b-lora.json` | 2노드 GLM-4.7-Flash 30B MoE LoRA, 평가와 node-local NVMe async checkpoint | train |
| `experiments/megatron/qwen3-30b-full.json` | 2노드 Qwen3 30B MoE full-parameter(non-LoRA) — memory 적합성 미검증, [Verification](verification.md#known-implementation-limits) 참고 | all |

앞의 smoke 네 개는 작은 Qwen2.5-0.5B와 고정 No Robots 입력을 사용합니다.
모델 품질·장기 수렴·일반적인 성능 비교용 preset은 아닙니다.

TRL NVMe preset과 두 Megatron 30B preset은 작은 smoke 네 개에 포함되지 않습니다.
Megatron의 Qwen과 GLM preset은 검증 당시와 같은 TP=1, PP=1, EP=2 구성으로 1 step, 평가 1회와 async checkpoint를 실행합니다.
GLM preset의 `TRANSFORMER_IMPL=auto`는 Transformer Engine 구현을 선택합니다.
TRL NVMe preset의 `train` stage는 학습 직후 같은 프로세스에서 평가를 실행하지 않고, 평가는 `tuned` stage를 별도 프로세스로 실행해 학습이 남긴 native DeepSpeed ZeRO checkpoint를 새 엔진에 복원한 뒤에 수행합니다.
이 preset에서 NVMe offload가 왜 필요한지(측정된 footprint와 노드 RAM 비교)는 [30B NVMe 실습](../labs/nvme-30b/README.md#why-nvme-offload-is-necessary-here)을 따릅니다.
준비 조건과 두 backend에서 `NVMe`가 뜻하는 범위, `tuned` stage 실행 조건은 [30B NVMe 실습](../labs/nvme-30b/README.md)을 따릅니다.

`experiments/run.py`는 `--backend`, `--setup`, `--experiment`, `--output`을 요구합니다.
기본은 dry-run이고 실제 실행의 `--timeout`은 기본 900초이며 양의 정수여야 합니다.
기존 출력 디렉터리는 재사용할 수 없습니다.
허용된 환경변수는 [runner](../experiments/run.py)의 `TRL_ENV`와 `MEGATRON_ENV`에서 확인합니다.
모든 preset은 `MODEL_ID`, `MODEL_REVISION`, `DATASET_ID`, `DATASET_REVISION`을 명시해야 합니다.
백엔드가 지원하는 모든 직접 CLI 옵션이 공통 runner에 노출된 것은 아닙니다.
Runner는 output 디렉터리 이름을 `OBSERVATORY_RUN_ID`로 예약해 framework metric과 host telemetry를 같은 실행에 연결합니다.

## Estimate Memory Before Running (`experiments/estimate_memory.py`)

GPU 시간을 쓰기 전에 "이 구성이 메모리에 들어가는가"를 먼저 계산합니다.
파라미터 크기는 아키텍처별 수식이 아니라 snapshot의 `*.safetensors` 헤더에서 **실제 텐서 바이트**를 읽어 구하므로, MoE·dense 구분이나 새 아키텍처 대응에 코드 변경이 필요 없습니다. Expert 가중치는 텐서 이름으로 식별해 expert parallel sharding만 따로 적용합니다.

```bash
python experiments/estimate_memory.py \
  --model-dir '<snapshot-dir>' \
  --backend megatron --finetuning-mode full --ep 2 --world-size 2 \
  --budget-gib 119
```

`--budget-gib`를 주면 per-rank 예산과 비교해 `FITS` / `DOES NOT FIT`을 판정합니다.
`--offload cpu|nvme`는 optimizer·gradient를 on-device에서 빼고, 대신 해당 계층이 감당해야 할 용량을 따로 알려줍니다.

| 검증 대상 | 실측 ([Verification](verification.md#30b-gpu-results)) | 이 도구의 예측 |
| --- | --- | --- |
| TRL DDP LoRA | 58.825 GiB | 59.2 GiB |
| TRL FSDP2 LoRA | 32.147 GiB | 29.3 GiB |
| Megatron LoRA EP=2 | 34.759 GiB | 30.8 GiB |

**이 값은 추정이지 측정이 아닙니다.** CUDA context, allocator 단편화, framework workspace를 모델링하지 않아 실측보다 **낮게** 나오는 경향이 있습니다(위 표에서 최대 -11%). 따라서 여유가 빠듯하게 `FITS`로 나오면 실제로는 안 들어갈 수 있다고 봐야 하며, 판정 근거로 쓸 때는 [Verification](verification.md)의 증거 구분 원칙을 그대로 따릅니다.

## Build an Experiment from Knobs (`experiments/build.py`)

매번 새 `experiments/*.json`을 손으로 작성하는 대신, `experiments/build.py`가 backend·dataset·model·offload·epoch 같은 간단한 knob을 그 스키마로 매핑해 `experiments/generated/<output 이름>.json`에 쓰고, 그 파일을 그대로 `experiments/run.py`의 `load_setup`/`load_experiment`/`build_plan`/`execute`에 넘깁니다. 검증·SSH 실행 로직은 전혀 새로 만들지 않고 재사용합니다.

```bash
python experiments/build.py \
  --backend trl --dataset ultrachat \
  --model-id Qwen/Qwen2.5-0.5B-Instruct --model-revision <40-hex> \
  --setup setups/spark/local.json --output results/ultrachat-epoch1 \
  --epochs 1 --train-samples 256 --eval-samples 32
```

기본값은 이 프로젝트의 실제 2노드 Spark 클러스터에 맞춰져 있습니다(`--nnodes 2`).
`--dataset`의 선택지는 고정 목록이 아니라 두 백엔드가 공유하는 `datasets_lab.public_data.PRESETS`에서 그대로 읽습니다(`reference_only`인 preset은 제외).

`--epochs`는 `--max-steps`와 배타적입니다. TRL은 `spark_train.py`가 HF `SFTConfig`의 `num_train_epochs`/`max_steps=-1` sentinel로 직접 처리합니다. Megatron은 step 기반 scheduler라 `--epochs`를 `build.py`가 미리 `steps = ceil(epochs * train_count / global_batch_size)`로 계산해 `MAX_STEPS`/`SCHEDULE_STEPS`로 넣습니다 — 이 계산은 이미 준비된 dataset의 `manifest.json`(`train_count`)을 controller에서 직접 읽을 수 있을 때만 가능합니다. `data_dir`가 node-local 경로([30B NVMe 실습](../labs/nvme-30b/README.md#training-data-storage-general-principle-vs-this-poc) 참고)인 지금 이 클러스터 구성에서는 controller가 그 경로를 읽을 수 없으므로 **Megatron에서 `--epochs`는 항상 명확한 오류로 즉시 멈추며, `--max-steps`를 직접 써야 합니다.** TRL은 `spark_train.py`가 실행 시점에 노드에서 직접 epoch를 처리하므로 이 제약이 없습니다.

`--offload {none,cpu,nvme}`는 TRL 전용입니다. `cpu`/`nvme`는 `--distributed-backend deepspeed`를 강제하고 `backends/trl/configs/deepspeed-zero3-{cpu,nvme}.json`을 선택합니다. NVMe profile은 full fine-tuning 전용이므로 `--finetuning-mode full`이 필요합니다. `--backend megatron`과 함께 쓰면 즉시 오류입니다 — Megatron은 오늘 offload 경로가 없습니다.

Megatron 전용 `--tp`/`--pp`/`--ep`/`--global-batch-size`/`--micro-batch-size`는 이 저장소에서 이미 검증된 30B preset의 기본값(TP=1, PP=1, EP=2)을 그대로 씁니다 — Megatron parallelism을 몰라도 일단 돌아가는 값입니다. 각 옵션의 뜻은 `python experiments/build.py --help`에서 확인합니다.

Dataset 준비(`prepare_public_data.py`)는 `build.py`가 대신 실행하지 않습니다 — HF Hub 다운로드 같은 부수효과를 조립 단계에 숨기지 않기 위해서이며, 준비 절차는 [Datasets](datasets.md)를 그대로 따릅니다.

### Representative Runs

`build.py`가 실제로 다양한 조합을 돌려 다른 결과를 내는지 확인한 네 가지 실행과 그 해석입니다. 모두 이 프로젝트의 실제 2노드 Spark 클러스터, node-local model·dataset 구성([30B NVMe 실습](../labs/nvme-30b/README.md#training-data-storage-general-principle-vs-this-poc) 참고)에서 `--execute`로 실행했습니다.

| Case | Backend | Model | Dataset | knob | 결과 |
| --- | --- | --- | --- | --- | --- |
| 1 | TRL DDP | Qwen2.5-0.5B-Instruct | ultrachat (256 train/32 eval) | `--epochs 1` | base/train/tuned 모두 통과 |
| 2 | TRL DeepSpeed | Qwen3-30B-A3B | no_robots | `--offload nvme --max-steps 1` | train 통과, 별도 `tuned` process 평가 통과 |
| 3 | Megatron | Qwen3-30B-A3B | self_oss (40,000 train/8,000 eval) | TP=1·PP=1·EP=2, `--max-steps 1 --stage all` | base `1.402232` → tuned `1.342737`, 통과 |
| 4 | Megatron | GLM-4.7-Flash | ultrachat (160,000 train/30,000 eval) | TP=1·PP=1·EP=2, `--max-steps 1 --max-length 4096 --stage all` | base `2.269922` → tuned `2.012836`, 통과 |

**Case 1 (epoch 검증)**: `--epochs`가 실제로 HF `SFTConfig`의 `num_train_epochs`를 통해 동작하는지, 그리고 학습이 진짜 loss를 낮추는지 확인하는 사례입니다. `summary-{base,train,tuned}.json`의 `evaluation.eval_loss`를 비교합니다.

| Stage | eval_loss | 의미 |
| --- | --- | --- |
| `base` | 1.4340 | 학습 전 원본 모델 |
| `train` | 1.4308 | 1 epoch(=16 optimizer step) 학습 직후, 같은 process에서 평가 |
| `tuned` | 1.4308 | 학습이 끝난 adapter를 별도 process에서 다시 읽어 평가 |

`train`과 `tuned`의 `eval_loss`가 소수점까지 정확히 같다는 것은 저장된 LoRA adapter를 다시 읽어도 수치가 흔들리지 않는다는 재로딩 신뢰성 증거입니다. `base`→`train`의 감소폭이 작은 건 256개 샘플·16 step만 학습했기 때문이며, 이 값 자체를 모델 품질의 일반적 지표로 확대 해석하지 않습니다.

이 사례는 DDP+LoRA의 node-local 저장 문제도 드러냈습니다: 당시에는 global rank 0의 노드에만 adapter가 남아 `tuned`가 실패했습니다. 현재는 `train` 단계가 각 rank의 node-local adapter를 저장하며, 2노드 `base`→`train`→`tuned` smoke로 재로딩까지 검증했습니다. 자세한 동작은 [TRL 백엔드 문서](backends/trl.md#choose-a-distributed-backend)를 따릅니다.

**Case 3·4 (Megatron `--stage all`)**: 이전 실패 원인은 HF dataset이 아니라 공유 파일시스템을 전제로 한 `torch_dist` checkpoint를 node-local NVMe에 나누어 저장한 것이었습니다. rank 0에 metadata와 `__0_0.distcp`, rank 1에 `__1_0.distcp`만 남아 새 process가 완전한 checkpoint를 볼 수 없었습니다. 공통 runner가 checkpoint를 NFS checkout 아래에 두도록 수정한 뒤 두 30B MoE 모델 모두 base→train→tuned 단일 실행과 iteration 1 재로딩을 통과했습니다. 학습 iteration의 NaN과 skipped iteration은 두 사례 모두 0이었고 `summary.json`의 `checkpoint_reload_verified`도 `true`입니다.

Case 4는 `--max-length` 기본값(2048)에서 ultrachat의 한 샘플이 길이를 초과해 한 번 실패했고, `--max-length 4096`으로 재실행해 통과했습니다 — preset마다 실제 대화 길이가 다르므로 `--max-length`를 데이터셋에 맞게 조정해야 할 수 있다는 실제 사례입니다.

## Repeated Megatron Measurements

이 반복 측정도 `Qwen/Qwen2.5-0.5B-Instruct`와 No Robots를 고정한 저비용 A/B 비교입니다.
모델과 데이터를 고정해야 설정 하나의 영향만 비교할 수 있고, 여덟 조건을 30B 모델로 반복하는 비용도 피할 수 있습니다.
따라서 여기서 얻은 시간·메모리 차이를 30B 모델의 성능으로 일반화하지 않습니다.

`experiments/benchmarks.py`는 [benchmark plan](../experiments/megatron/benchmark-plan.json)의 cell과 두 variant를 읽습니다.
저장소 루트의 실제 Git checkout에서 실행하며 기본값은 dry-run입니다.

```bash
python experiments/benchmarks.py \
  --setup setups/spark/local.json \
  --output results/benchmarks-dry-run
```

기본 계획은 8개 cell, variant별 4회 측정과 별도 warmup으로 구성됩니다.
길이 비교는 `MAX_LENGTH` 상한만 바꾸지 않고 고정 길이 padding을 사용합니다.
Selective recompute는 현재 설정 오류로 warmup에서 실패하므로 full/selective 비교가 완결된다고 기대하지 않습니다.

아래는 16 step·2회 측정으로 줄인 실행안입니다.
여러 GPU 작업과 checkpoint 쓰기를 연속 수행하므로 준비된 두 노드·모델·데이터와 충분한 저장 공간이 필요합니다.
먼저 `--execute` 없이 같은 인자를 사용해 계획을 확인합니다.

```bash
python experiments/benchmarks.py \
  --setup setups/spark/local.json \
  --output results/benchmarks-16x2 \
  --steps 16 --repeats 2 --within-run-warmup 4 \
  --checkpoint-intervals 4 8 \
  --execute
```

이 축소 계획은 별도 warmup을 포함해 48개 run을 선택하며 첫 4개 within-run step을 측정에서 제외합니다.
Cell 이름의 checkpoint interval 16/32는 유지되지만 실제 interval은 인자의 4/8이므로 run config를 기준으로 해석합니다.
`--plan`으로 계획을 바꾸고 `--timeout`으로 개별 실행 제한을 지정할 수 있습니다.
실패한 variant의 후속 반복은 생략될 수 있습니다.

## Read the Outputs

출력에는 계획·실행 설정, manifest, rank 로그, measurement JSONL과 `records.jsonl`이 남습니다.
기록은 성공·실패·생략을 포함하며 정상 종료·예상 step 수·finite 값 검사를 통과한 measured run만 비교합니다.
Warmup 성공을 측정 반복 수에 포함하지 않습니다.

Whole-run 시간은 원격 시작·초기화·평가·정리를 포함합니다.
CUDA allocator peak는 전체 장치 사용량이 아닙니다.
Checkpoint save와 blocking finalization은 host 호출 시간이며 native step timer와 별도입니다.
Async background I/O의 자원 경쟁은 학습 시간에 영향을 줄 수 있습니다.
반복 결과는 각 실행의 manifest와 `records.jsonl`에서 판정하고, 저장소에는 과거 run log를 복제해 보관하지 않습니다.
