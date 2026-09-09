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
| `experiments/trl/nvme-offload-30b.json` | 2노드 Qwen3 30B full SFT, ZeRO-3 NVMe state offload | train |
| `experiments/megatron/nvme-checkpoint-30b.json` | 2노드 Qwen3 30B LoRA, node-local NVMe async checkpoint | train |

앞의 smoke 네 개는 작은 Qwen2.5-0.5B와 고정 No Robots 입력을 사용합니다.
모델 품질·장기 수렴·일반적인 성능 비교용 preset은 아닙니다.

두 NVMe preset은 30B Qwen3 모델을 사용하며 작은 smoke 네 개에 포함되지 않습니다.
준비 조건과 두 backend에서 `NVMe`가 뜻하는 범위는 [30B NVMe 실습](../labs/nvme-30b/README.md)을 따릅니다.

`experiments/run.py`는 `--backend`, `--setup`, `--experiment`, `--output`을 요구합니다.
기본은 dry-run이고 실제 실행의 `--timeout`은 기본 900초이며 양의 정수여야 합니다.
기존 출력 디렉터리는 재사용할 수 없습니다.
허용된 환경변수는 [runner](../experiments/run.py)의 `TRL_ENV`와 `MEGATRON_ENV`에서 확인합니다.
백엔드가 지원하는 모든 직접 CLI 옵션이 공통 runner에 노출된 것은 아닙니다.

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
