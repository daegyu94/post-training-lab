# Experiments

Experiment 파일은 model·dataset revision과 training option을 선언합니다.
Runner는 이 파일을 setup과 결합해 plan을 만들고 --execute가 있을 때만 원격 학습을 시작합니다.

## 단일 실행

experiments/run.py의 CLI는 다음 네 인자를 요구합니다.

```text
--backend {trl,megatron}
--setup SETUP
--experiment EXPERIMENT
--output OUTPUT
```

Dry-run은 기본 동작입니다.
실행하려면 --execute를 추가하고 timeout은 양의 정수 --timeout으로 조정합니다.
Output directory는 기존 경로를 재사용할 수 없습니다.

## 제공 preset

| Backend | Preset |
| --- | --- |
| TRL | experiments/trl/single-node-smoke.json, experiments/trl/smoke.json |
| Megatron | experiments/megatron/smoke.json, experiments/megatron/resume-smoke.json |

Preset은 bounded smoke입니다.
모델 품질과 성능을 비교하려면 고정된 dataset, model revision, seed, topology와 충분한 repeat를 별도로 설계해야 합니다.

## 반복 Megatron 측정

experiments/benchmarks.py는 experiments/megatron/benchmark-plan.json의 cell과 두 variant를 읽습니다.
각 run을 공통 runner로 실행하고 config, manifest, rank log, measurement JSONL과 records.jsonl을 저장합니다.
기본 동작은 dry-run입니다.

먼저 계획만 확인합니다.

```bash
python experiments/benchmarks.py \
  --setup setups/spark/local.json \
  --output results/benchmarks-dry-run
```

짧은 실제 측정은 다음처럼 실행할 수 있습니다.

```bash
python experiments/benchmarks.py \
  --setup setups/spark/local.json \
  --output results/benchmarks-16x2 \
  --steps 16 \
  --repeats 2 \
  --within-run-warmup 4 \
  --checkpoint-intervals 4 8 \
  --execute
```

--plan으로 다른 benchmark plan을 지정할 수 있고 --timeout 기본값은 900초입니다.
각 variant에는 별도 warmup이 있고 warmup은 timing summary에서 제외됩니다.
실패한 variant의 이후 repeat는 skip될 수 있습니다.

## 결과 해석

records.jsonl은 성공·실패·skip을 포함한 raw execution record입니다.
Summary는 finite loss/gradient, expected step 수와 정상 종료를 통과한 measured run만 사용합니다.
Checkpoint save time은 training-step time과 별도로 읽습니다.
두 번 이상의 짧은 repeat만으로 일반적인 speedup이나 model quality를 결론 내리지 않습니다.
