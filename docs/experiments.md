# Experiments

Experiment 파일은 모델·데이터 revision과 학습 조건을, setup은 실행할 노드와 경로를 선언합니다.
공통 runner가 둘을 결합하며 `--execute`가 있을 때만 SSH 학습을 시작합니다.
첫 실행은 [Getting Started](getting-started.md), 설정 책임은 [Architecture](architecture.md)를 따릅니다.

현재 목표는 두 노드에서 30B 모델의 SFT 실행 경로를 확인하는 것입니다 — `Qwen/Qwen3-30B-A3B`는 TRL LoRA와 Megatron MoE로, `zai-org/GLM-4.7-Flash`는 Megatron MoE로 확인합니다.
30B 실행에서는 모델별 준비, 메모리 배치, 2노드 통신, checkpoint와 재로딩을 봅니다.
0.5B preset은 이 목표를 대신하는 실험이 아니라 runner·전처리·저장·재로딩 같은 기본 동작을 싸게 검사하는 smoke이며, 두 결과는 서로 다른 범위로 해석합니다.

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

| 검증 대상 | 실측 ([30B Measurements](measurements-30b.md#30b-gpu-results)) | 예측 |
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
30B에서 실제로 측정한 checkpoint·memory 수치는 [30B Measurements](measurements-30b.md)에 있습니다.
