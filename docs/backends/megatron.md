# Megatron Backend

Megatron backend는 Megatron Bridge recipe를 사용한 Spark 분산 SFT와 checkpoint workflow를 제공합니다.
주요 학습 대상은 full parameter SFT이며 LoRA와 parallelism feature도 설정으로 다룹니다.

## GPU 없이 개념 확인

Recipe와 논리적 TP/PP/CP/DP rank layout은 CUDA, NCCL과 distributed process group을 초기화하지 않고 확인할 수 있습니다.

```bash
cd backends/megatron
./scripts/run_megatron_practice.sh
```

이 명령은 inspect_megatron_recipe.sh와 simulate_parallelism.sh를 순서대로 실행합니다.
개념 simulation은 실제 training process group이나 GPU 실행을 검증하지 않습니다.

## Spark SFT

공통 runner로 experiments/megatron/smoke.json을 dry-run한 뒤 실행합니다.
Launcher의 stage는 base, train, tuned이며 RESUME_AFTER_TRAIN=true일 때 내부적으로 resume stage가 추가됩니다.

Checkpoint를 다시 읽는 preset은 experiments/megatron/resume-smoke.json입니다.
Checkpoint 저장·재개는 uninterrupted run과의 수치 동등성이나 장애 후 durability를 자동으로 보장하지 않습니다.

Megatron은 다음 topology 조건을 먼저 확인합니다.

- world_size는 TP*PP와 PP*EP로 나누어져야 합니다.
- GLOBAL_BATCH_SIZE는 MICRO_BATCH_SIZE*DP로 나누어져야 합니다.
- SEQUENCE_PARALLEL=true이면 TP>=2와 TRANSFORMER_IMPL=transformer_engine이 필요합니다.

모델 snapshot은 각 node의 local path에 준비하고 학습 JSONL은 두 node에서 보이는 DATA_DIR에 둡니다.
MODEL_REVISION과 DATASET_REVISION은 experiment에서 pin합니다.

## 반복 feature 측정

반복 측정의 실행법과 output schema는 [Experiments](../experiments.md)에서 관리합니다.
주요 비교 축은 overlap, recompute, sequence parallel과 sync/async checkpoint입니다.
현재 plan의 feature 지원 여부와 실제 성공 여부는 source configuration과 archive된 verification record를 함께 확인해야 합니다.

## 확인 기준

Rank별 exit status, finite loss, expected checkpoint/summary와 manifest를 확인합니다.
Checkpoint 저장 성공은 crash recovery나 성능 우위를 의미하지 않습니다.
짧은 실행의 loss와 timing은 해당 model·topology·revision에 한정된 관측입니다.

## 테스트

```bash
python -m pytest -q
```
