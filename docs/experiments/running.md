# Run Experiments

실험 preset을 선택하고 실행 전 메모리를 추정한 뒤 결과를 검증하는 절차입니다.

## Presets

| 파일 | 노드·학습 | 단계 |
| --- | --- | --- |
| `experiments/trl/single-node-smoke.json` | 1노드 DDP LoRA, 1 step | base/train/tuned |
| `experiments/trl/smoke.json` | 2노드 DDP LoRA, 1 step | base/train/tuned |
| `experiments/megatron/smoke.json` | 2노드 full SFT, 2 step | base/train/tuned |
| `experiments/megatron/resume-smoke.json` | 2노드 full SFT, 2→3 step | base/train/resume/tuned |
| `experiments/trl/nvme-offload-30b.json` | 2노드 Qwen3 30B full SFT, ZeRO-3 NVMe offload | base/train/tuned |
| `experiments/trl/nvme-offload-ultrachat-30b.json` | 2노드 Qwen3 30B full SFT, UltraChat·ZeRO-3 NVMe offload | base/train/tuned |
| `experiments/trl/nvme-offload-ultrachat-glm-30b.json` | 2노드 GLM 30B full SFT, UltraChat·ZeRO-3 NVMe offload | base/train/tuned |
| `experiments/megatron/qwen3-30b-lora.json` | 2노드 Qwen3 30B MoE LoRA, 평가와 node-local async checkpoint | train |
| `experiments/megatron/glm-4.7-flash-30b-lora.json` | 2노드 GLM-4.7-Flash 30B MoE LoRA, 평가와 node-local async checkpoint | train |
| `experiments/trl/glm-4.7-flash-30b-lora.json` | 2노드 GLM-4.7-Flash 30B DDP LoRA, no_robots | all |
| `experiments/megatron/qwen3-30b-full.json` | 2노드 Qwen3 30B MoE full-parameter — 아래 참고, 메모리 적합성 미검증 | all |
| `experiments/megatron/qwen3-30b-full-sgd-pilot.json` | 2노드 Qwen3 30B full-SFT SGD 용량 pilot | train |

Preset을 고를 때 알아야 할 것:

- **smoke 네 개**는 Qwen2.5-0.5B와 고정 No Robots 입력을 씁니다. 모델 품질·장기 수렴·성능 비교용이 아닙니다.
- **Megatron 30B LoRA 두 개**는 검증 당시와 같은 TP=1·PP=1·EP=2로 1 step, 평가 1회와 async checkpoint를 실행합니다.
- **TRL NVMe preset**은 `train` 후 별도 `tuned` process에서 native ZeRO checkpoint를 복원해 평가합니다([30B NVMe 실습](../../labs/nvme-30b/README.md)).

실행하면 안 되는 두 preset:

| Preset | 막힌 지점 |
| --- | --- |
| `qwen3-30b-full.json` | Adam 구성이 **per-rank 추정부터** 장치 예산 초과 — 실행 대상 아님 |
| `qwen3-30b-full-sgd-pilot.json` | UltraChat 2048 고정 코호트와 SGD로 범위를 줄였지만, FP32 main gradient 구성 중 global OOM으로 첫 optimizer step 미완료 |

같은 2노드 구성을 반복하지 말고 [현재 결과와 후속 조건](30b-results.md#full-sft-capacity)을 먼저 확인합니다.

`experiments/run.py`는 `--backend`, `--setup`, `--experiment`, `--output`을 요구합니다.
기본은 dry-run이고 `--timeout`은 기본 900초의 양의 정수이며 기존 출력 디렉터리는 재사용할 수 없습니다.
모든 preset은 `MODEL_ID`, `MODEL_REVISION`, `DATASET_ID`, `DATASET_REVISION`을 명시해야 하고, 백엔드가 지원하는 모든 CLI 옵션이 runner에 노출된 것은 아닙니다.
Runner는 output 디렉터리 이름을 `OBSERVATORY_RUN_ID`로 예약해 application metric과 host telemetry를 같은 실행에 연결합니다.

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

| 검증 대상 | 실측 ([30B GPU Results](30b-results.md#30b-gpu-results)) | 예측 | 오차 |
| --- | ---: | ---: | ---: |
| TRL DDP LoRA | 58.825 GiB | 59.2 GiB | +0.6% |
| TRL FSDP2 LoRA | 32.147 GiB | 29.3 GiB | −8.9% |
| Megatron LoRA EP=2 | 34.759 GiB | 30.8 GiB | −11.4% |

추정값은 CUDA context·allocator 단편화·workspace를 제외하므로 **실측보다 낮게 나오는 쪽으로 치우칩니다**(표에서 최대 −11%).
따라서 예산에 근접한 `FITS`는 실제 적합성을 보장하지 않습니다 — 추정 90.0 GiB/rank가 119 GiB 예산 안이었는데도 실제로는 OOM이 난 [Full-SFT capacity](30b-results.md#full-sft-capacity)가 그 사례입니다.

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
- Megatron 전용 `--tp`/`--pp`/`--ep`/batch 옵션은 검증된 30B preset 기본값(TP=1, PP=1, EP=2)을 그대로 씁니다. `--optimizer` 기본값은 Megatron `adam`, TRL `adamw`이며 명시한 값은 launcher까지 전달됩니다.
- Dataset은 [Datasets](../datasets.md)에 따라 미리 준비합니다.

## Read the Outputs

| 확인할 질문 | 기록 위치 | 읽는 방법 |
| --- | --- | --- |
| 개별 실행이 성공했는가? | `runs/<run>/manifest.json`의 `status`, `ranks` | 각 rank 종료 상태부터 확인. 실패·pilot·warmup을 측정 median에 섞지 않음 |
| Checkpoint 반복 결과는? | 최상위 `manifest.json`의 `checkpoint_summary.<variant>` | `run_count`, `save_call_seconds_median`, `logical_checkpoint_bytes_median`, rMAD 확인 |
| Save와 finalization을 따로 보려면? | 각 record의 `metrics`, `measurements/rank-<rank>-train.jsonl` | Save는 rank별 호출 합, finalization은 별도 blocking 호출 합. 누락된 값은 0으로 대체하지 않음 |
| Read가 정말 cold였는가? | record의 `post_run.ranks`, `post_run.aggregate.reads` | `rank_classifications`와 physical/logical bytes를 함께 확인. cold 요청 성공만으로 실제 eviction을 단정하지 않음 |
| Memory condition의 값은? | Memory manifest의 `records[].resources`, `metrics_by_rank` 또는 `trl_summary` | `condition`, `pilot`, `status`로 실행을 고른 뒤 동일 rank 집계 방식으로 비교 |

`run_count`는 post-run probe가 있는 성공한 measured run 수입니다. 개별 metric 누락 여부는 원본 record에서 추가로 확인합니다. `extend_to_eight_runs=true`는 추가 반복 권고이며 자동 실행 기능은 아닙니다.

출력에는 계획·실행 설정, manifest, rank 로그, measurement JSONL과 `records.jsonl`이 남습니다.
기록은 성공·실패·생략을 포함하며 정상 종료·예상 step 수·finite 값 검사를 통과한 measured run만 비교합니다.
Warmup 성공은 측정 반복 수에 포함하지 않습니다.

해석할 때 구분할 것:

- Whole-run 시간은 원격 시작·초기화·평가·정리를 포함합니다.
- CUDA allocator peak는 전체 장치 사용량이 아닙니다.
- Checkpoint save와 blocking finalization은 host 호출 시간이며 native step timer와 별도입니다.
- Async background I/O의 자원 경쟁이 학습 시간에 영향을 줄 수 있습니다.

반복 결과는 각 실행의 manifest와 `records.jsonl`에서 판정하며 과거 run log를 저장소에 복제해 보관하지 않습니다.

현재 30B 통제 matrix와 재실험 우선순위는 [30B Controlled Results](30b-results.md#controlled-results-2026-09-15)를 따릅니다.
NFS는 repository checkout에만 사용하므로 distributed checkpoint restore 예시는 제공하지 않습니다.
