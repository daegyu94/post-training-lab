# Megatron Bridge Post-Training Lab

Megatron Bridge로 LoRA supervised fine-tuning(SFT)과 분산 학습 기능을 실습하는 `megatron` 브랜치입니다.
공통 하드웨어 구성과 Setup 1·2 소개는 [main의 PoC Setups](https://github.com/daegyu94/post-training-lab/blob/main/README.md#poc-setups)에서 확인하세요.

## Start Here

1. 단일 GPU SFT는 [Setup1 guide](docs/single-gpu.md)의 환경 설치 → model/data 준비 → 실행 → 결과 확인 순서를 따릅니다.
2. Spark 두 노드 실행은 [Setup2 guide](docs/spark-cluster.md)의 노드별 환경 확인 → model/data 준비 → 양 노드 launcher 실행 순서를 따릅니다.
3. Checkpoint save/resume, overlap, recompute, sequence parallel 비교는 Setup2 환경에서 [feature labs](docs/megatron-feature-labs.md)를 진행합니다.

실제 LLM 연산은 Spark 노드에서 실행하고 controller는 개발과 실행 조율에 사용합니다.
Controller의 `/home/daegyu/shared/post-training-lab`와 Spark 노드의 `/home/spark/shared/post-training-lab`는 같은 NFS 파일을 가리킵니다.
명령의 경로는 실행 노드의 mount 경로를 사용합니다.

## Workflow and Validation Scope

두 launcher의 기본 흐름은 base held-out evaluation → LoRA 학습과 checkpoint 저장 → 별도 process에서 tuned evaluation입니다.
Setup2는 `RESUME_AFTER_TRAIN=true`와 `RESUME_MAX_STEPS`로 optimizer·scheduler state를 읽는 학습 재개 단계를 추가할 수 있습니다.
Adapter reload 평가와 학습 상태 resume는 별도로 확인합니다.

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

- `megatron_lab/config.py`: Qwen2.5-7B single-GPU 및 Qwen3/GLM setup2 provider 설정
- `megatron_lab/sft.py`: base, train, resume, tuned stage 진입점
- `megatron_lab/cluster.py`: explicit torchrun topology와 dense/expert DP 검증
- `megatron_lab/feature_lab.py`: feature variant와 warmup 제외 timing summary
- `megatron_lab/prepare_data.py`: deterministic UltraChat subset 준비
- `megatron_lab/compare.py`: base와 reloaded-checkpoint loss 비교
- `megatron_lab/preflight.py`: GPU, dependency, input path 검사
- `megatron_lab/inspect_recipe.py`: GPU 초기화 없는 recipe 요약
- `megatron_lab/parallelism.py`: TP/PP/CP/DP rank group simulation
- `run_summary.py`: framework 공통 summary schema
- `scripts/run_experiment.sh`: setup1 end-to-end experiment orchestration
- `scripts/run_spark_cluster.sh`: explicit two-node setup2 launcher
- `scripts/run_feature_lab.sh`: small dense model feature A/B harness
- `tests/`: data, log parsing, rank layout의 CPU unit tests

## Run the CPU Tests

model checkpoint나 GPU 없이 data selection, log parsing, parallel rank layout을 검증합니다.

```bash
.venv/bin/python -m pytest -q
```

## Related Guides

- [Public datasets](docs/public-datasets.md): 공개 dataset의 canonical JSONL 변환
- [Internal data](docs/internal-data-guide.md): 서비스 데이터 준비
- `system-integration` branch: data·checkpoint·serving lifecycle 설계
- `profiling` branch: cluster resource 분석
