# Post-Training Lab

이미 학습된 LLM에 SFT를 적용하고 데이터 준비, 분산 실행, 저장·재로딩과 자원 계측을 실습하는 저장소입니다.
현재 백엔드는 TRL과 Megatron Bridge이며 공통 실행 환경은 NVIDIA DGX Spark입니다.
Controller가 실행을 조율하고 실제 학습은 설정한 Spark 노드에서 수행합니다.

## Start Here

[Getting Started](docs/getting-started.md)에서 NVIDIA DGX Spark 노드 환경과 입력을 준비한 뒤 첫 분산 SFT를 실행합니다.
Controller에는 Python 3.10 이상과 Git이 필요하며 학습에는 준비된 Spark 노드·CUDA 환경·모델·데이터가 필요합니다.
문서 전체 안내는 [Documentation](docs/README.md), 구성 요소의 책임은 [Architecture](docs/architecture.md)를 확인합니다.

| 할 일 | 안내 |
| --- | --- |
| 학습 입력 만들기 | [Datasets](docs/datasets.md) |
| TRL SFT 실행 | [TRL](docs/backends/trl.md) |
| Megatron SFT·checkpoint 재개 | [Megatron](docs/backends/megatron.md) |
| 반복 비교 | [Experiments](docs/experiments.md) |
| 자원·통신·저장소 계측 | [Observability](docs/observability.md) |
| 결과·한계 확인 | [Verification](docs/verification.md) |

## Repository Layout

| 경로 | 역할 |
| --- | --- |
| `setups/spark/` | 노드·경로·환경 설정 |
| `experiments/` | 공통 runner·학습 preset·반복 측정 |
| `backends/` | 백엔드별 학습·데이터·환경 |
| `observability/` | 계측·baseline·trace 도구 |
| `docs/` | 사용자 가이드·reference·설계 |
| `labs/` | 실습 목표·전제조건·완료 기준 |
| `tests/` | CPU·실행 계약 회귀 검사 |

DPO, RL trainer, registry와 serving 배포는 구현되어 있지 않습니다.
[Design](docs/design.md)은 후속 통합 제안이며 실행 가능한 workflow가 아닙니다.
