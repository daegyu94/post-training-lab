# Megatron Bridge Post-Training Lab

Megatron Bridge로 Setup2의 두 Spark 노드에서 분산 학습, checkpoint 저장·재개와 parallelism을 실습하는 `megatron` 브랜치입니다.
단일 GPU post-training 입문은 [`trl` 브랜치](https://github.com/daegyu94/post-training-lab/tree/trl), 공통 하드웨어 구성은 [main의 PoC Setups](https://github.com/daegyu94/post-training-lab/blob/main/README.md#poc-setups)를 참고하세요.

## Start Here

먼저 [Setup2 guide](docs/spark-cluster.md)에서 노드별 환경과 NCCL 통신, pinned model/data 준비 방법을 확인합니다.
이후 작은 dense 모델에서 분산 기능을 익히고 30B MoE integration으로 확장합니다.

| 순서 | 실습 | 확인할 내용 |
| --- | --- | --- |
| 1 | [0.5B dense workload](docs/megatron-feature-labs.md#small-dense-feature-workload) | 두 노드 실행과 rank별 로그 |
| 2 | [Checkpoint save/resume](docs/megatron-feature-labs.md#measured-dcprestart-evidence), [DP→TP reshard](docs/megatron-feature-labs.md#layout-reshard) | 저장·재개 correctness와 topology 변경의 제약 |
| 3 | [Feature A/B](docs/megatron-feature-labs.md#matrix) | overlap, recompute, sequence parallel 비교 |
| 4 | [30B MoE integration](docs/spark-cluster.md#verified-30b-integration-smoke) | Qwen3/GLM EP=2 LoRA; 현재 검증은 1-step smoke 범위 |

GPU 없이 시작하려면 아래 [CPU 개념 실습](#cpu-only-concept-exercise)으로 논리적 rank 배치를 먼저 살펴볼 수 있습니다.

## Workflow and Validation Scope

Setup2 launcher의 기본 흐름은 base held-out evaluation → LoRA 학습과 checkpoint 저장 → 별도 process에서 tuned evaluation입니다.

```text
Model + Prepared Dataset
          |
          v
Base Held-out Evaluation
          |
          v
LoRA Training + Checkpoint Save
          |
          +---------------------------------+
          |                                 |
          | Default                         | Setup2: optional
          |                                 v
          |                       Resume Training
          |                       (optimizer + scheduler)
          |                                 |
          |                                 v
          |                       Save Updated Checkpoint
          |                                 |
          +<--------------------------------+
          |
          v
Tuned Evaluation (new process)
          |
          v
Compare Base / Tuned -> summary.json
```

Setup2는 `RESUME_AFTER_TRAIN=true`와 `RESUME_MAX_STEPS`로 optimizer·scheduler state를 읽는 학습 재개 단계를 추가할 수 있습니다.
Adapter reload 평가와 학습 상태 resume는 별도로 확인합니다.
그림은 launcher의 실행 순서이며, 각 모델에서 전체 경로의 검증이 완료되었다는 뜻은 아닙니다.

기존 [Setup2 검증 기록](docs/spark-cluster.md#verified-30b-integration-smoke)은 두 30B 모델의 EP=2 LoRA **1-step** 학습·validation·sync DCP 저장을 다룹니다.
Launcher 기본값인 **5-step 전체 workflow** 또는 30B checkpoint의 별도-process reload 완료를 의미하지 않습니다.
소형 Qwen2.5-0.5B는 Setup2의 분산·checkpoint·feature A/B를 확인하는 보조 모델이며 상세 결과는 [feature labs](docs/megatron-feature-labs.md)에 정리되어 있습니다.
짧은 smoke와 작은 subset의 loss 변화는 장기 수렴, 일반적인 model quality 또는 안정적인 throughput 결과로 해석하지 않습니다.

## CPU-only Concept Exercise

GPU 초기화, checkpoint download, NCCL 통신 없이 Bridge recipe와 논리적 parallel rank group을 살펴볼 수 있습니다.

```bash
./scripts/run_megatron_practice.sh
```

기본 예제는 TP=2, PP=2, CP=2, DP=2인 16개 논리 rank를 출력합니다.
실제 distributed process group을 만들지는 않습니다.
자세한 내용은 [Megatron stack overview](docs/megatron-overview.md)를 참고하세요.

## Repository Layout

- `megatron_lab/config.py`: 0.5B dense 및 Qwen3/GLM Setup2 provider 설정
- `megatron_lab/sft.py`: base, train, resume, tuned stage 진입점
- `megatron_lab/cluster.py`: explicit torchrun topology와 dense/expert DP 검증
- `megatron_lab/feature_lab.py`: feature variant와 warmup 제외 timing summary
- `megatron_lab/prepare_data.py`: deterministic UltraChat subset 준비
- `megatron_lab/compare.py`: base와 reloaded-checkpoint loss 비교
- `megatron_lab/inspect_recipe.py`: GPU 초기화 없는 recipe 요약
- `megatron_lab/parallelism.py`: TP/PP/CP/DP rank group simulation
- `run_summary.py`: framework 공통 summary schema
- `scripts/run_spark_cluster.sh`: explicit two-node setup2 launcher
- `scripts/run_feature_lab.sh`: small dense model feature A/B harness
- `tests/`: data, log parsing, rank layout의 CPU unit tests

## Run the CPU Tests

model checkpoint나 GPU 없이 data selection, log parsing, parallel rank layout을 검증합니다.

```bash
.venv/bin/python -m pytest -q
```

## Related Guides

- [공통 Dataset Guides](https://github.com/daegyu94/post-training-lab/blob/main/README.md#dataset-guides): 공개·사내 데이터 기준
- [Megatron 데이터 준비](docs/dataset-preparation.md): 변환 명령과 학습 연결 제약
- `system-integration` branch: data·checkpoint·serving lifecycle 설계
- `profiling` branch: cluster resource 분석
