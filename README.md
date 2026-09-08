# Post-Training Lab

Post-Training Lab은 LLM post-training을 데이터 준비, 학습, 평가와 resource profiling까지 연결해 실습하는 저장소입니다.
실행 코드는 backend 디렉터리에, 공통 데이터 기준은 `docs/datasets`에 모읍니다.
기본 분산 실행 환경은 `spark1`, `spark2` 두 노드이며 실제 workload는 Spark 노드에서 수행합니다.

## Start Here

| 목적 | 시작점 | 범위 |
| --- | --- | --- |
| SFT 시작하기 | [TRL backend](backends/trl/README.md) | QLoRA와 Spark 분산 SFT |
| 분산 학습 구조 익히기 | [Megatron backend](backends/megatron/README.md) | Spark 분산 학습, parallelism과 checkpoint |
| 실행 자원 측정하기 | [Observability lab](observability/README.md) | GPU·host·network·storage 측정 |
| 데이터 기준 확인하기 | [Dataset guides](docs/datasets/) | 공개 dataset과 승인된 service trace |
| 공통 실습 작성하기 | [Lab catalog](labs/README.md) | 실습 템플릿과 확장 계획 |

## Learning Paths

먼저 선택한 backend의 README에서 환경과 실행 순서를 확인합니다.
TRL과 Megatron의 Spark 실행 문서에는 실제로 검증된 범위와 실패 조건을 구분해 기록합니다.
Observability는 GPU나 exporter 없이 실행할 수 있는 CPU 검증과 실제 Spark 수집 경로를 함께 제공합니다.

각 backend의 실행 명령은 해당 backend 디렉터리를 current working directory로 가정합니다.
예를 들어 TRL 명령은 `cd backends/trl` 뒤 실행하고, Megatron 명령은 `cd backends/megatron` 뒤 실행합니다.

## Dataset Guides

- [공개 데이터 가이드](docs/datasets/public-datasets.md): schema, license와 pinned revision
- [사내 데이터 가이드](docs/datasets/internal-data-guide.md): 승인된 service trace의 정제, split과 test 정답 분리

변환 명령과 backend별 학습 연결 방법은 [TRL 데이터 준비](docs/backends/trl/dataset-preparation.md)와 [Megatron 데이터 준비](docs/backends/megatron/dataset-preparation.md)에 있습니다.

## Repository Layout

```text
backends/trl/          TRL SFT implementation and Spark launcher
backends/megatron/     Megatron Bridge Spark implementation
observability/         profiling helpers, examples and monitoring scripts
docs/backends/         backend setup and verification records
docs/observability/    resource profiling guides and metric contract
docs/datasets/         common data policy and provenance guides
tests/                 combined CPU unit tests grouped by backend
```

The system-integration design remains a separate follow-up scope.
