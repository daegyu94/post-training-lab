# Post-Training Lab

Post-Training Lab은 이미 학습된 LLM을 특정 작업에 맞게 추가 학습하는 post-training 실습 저장소입니다.
데이터 준비부터 학습·평가, 실행 중 자원 사용량 측정까지 따라갈 수 있습니다.
실행 코드는 backend 디렉터리에, 공통 데이터 기준은 `docs/datasets`에 모읍니다.
기본 분산 실행 환경은 `spark1`, `spark2` 두 노드이며 실제 workload는 Spark 노드에서 수행합니다.

지원 조합과 구조는 [Architecture](docs/architecture.md), 반복 성능 측정은 [실험 가이드](docs/experiments/repeated-measurements.md)를 참고하세요.
실제 Spark 실행과 실패 원인, CPU 검사 결과는 [통합 검증 기록](docs/verification/integration-20260908/README.md)에 있습니다.

## Start Here

| 목적 | 시작점 | 범위 |
| --- | --- | --- |
| SFT 시작하기 | [TRL backend](backends/trl/README.md) | QLoRA와 Spark 분산 SFT |
| 분산 학습 구조 익히기 | [Megatron backend](backends/megatron/README.md) | Spark 분산 학습, parallelism과 checkpoint |
| 실행 자원 측정하기 | [Observability lab](observability/README.md) | GPU·host·network·storage 측정 |
| 데이터 기준 확인하기 | [Dataset guides](docs/datasets/) | 공개 dataset과 승인된 service trace |
| 공통 실습 작성하기 | [Lab catalog](labs/README.md) | 실습 템플릿과 확장 계획 |

## PoC Setups

| Setup | Hardware | 실행 범위 |
| --- | --- | --- |
| Spark | GB10 두 노드, 노드당 memory 약 119 GiB | backend와 parallelism에 따라 한 노드 또는 두 노드 |

현재 설정 진입점은 Spark이며, 다른 환경은 setup을 추가해 연결합니다.
기존 RTX TRL 실습 문서는 기능 보존 자료이며 이번 통합의 재검증 대상이 아닙니다.

## Learning Paths

먼저 [시작 가이드](docs/getting-started.md)에서 Spark 환경 설정과 backend별 실험 설정을 준비합니다.
TRL과 Megatron의 Spark 실행 문서에는 실제로 검증된 범위와 실패 조건을 구분해 기록합니다.
Observability는 GPU나 exporter 없이 실행할 수 있는 CPU 검증과 실제 Spark 수집 경로를 함께 제공합니다.

각 backend 문서의 명령은 해당 backend 디렉터리에서 실행합니다.
예를 들어 TRL 명령은 `cd backends/trl` 뒤 실행하고, Megatron 명령은 `cd backends/megatron` 뒤 실행합니다.

## Dataset Guides

[SFT 데이터 준비 가이드](docs/datasets/README.md)에서 공개·사내 데이터 선택, 변환 명령, TRL·Megatron 학습 연결과 검증 기준을 함께 설명합니다.

## Repository Layout

```text
setups/spark/          Spark nodes, runtime paths and communication settings
experiments/           backend experiment presets and execution tools
backends/trl/          TRL SFT implementation and Spark launcher
backends/megatron/     Megatron Bridge Spark implementation
observability/         profiling helpers, examples and monitoring scripts
docs/backends/         backend setup and verification records
docs/observability/    resource profiling guides and metric contract
docs/datasets/         common data policy and provenance guides
tests/                 combined CPU unit tests grouped by backend
```

[System integration](docs/design/system-integration/README.md)은 데이터·checkpoint·serving lifecycle의 설계 문서이며 실행 구현 완료를 의미하지 않습니다.
통합 전 브랜치의 보존 상태는 [이관 기록](docs/migration/README.md)에서 확인합니다.
