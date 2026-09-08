# Post-Training Lab 문서

이 디렉터리는 처음 repository를 사용하는 software engineer를 위한 사용자 문서입니다.
실제 source code, configuration, launcher와 test가 문서의 source of truth입니다.
문서에 없는 option이나 아직 구현되지 않은 workflow를 추측해서 추가하지 않습니다.

## 먼저 읽을 문서

| 목적 | 문서 |
| --- | --- |
| 환경을 준비하고 첫 실행하기 | [Getting Started](getting-started.md) |
| runner와 설정 파일의 책임 이해하기 | [Architecture](architecture.md) |
| 학습 데이터 만들기 | [Datasets](datasets.md) |
| TRL 실행하기 | [TRL backend](backends/trl.md) |
| Megatron Bridge 실행하기 | [Megatron backend](backends/megatron.md) |
| 반복 측정 실행하기 | [Experiments](experiments.md) |
| 자원과 통신 관측하기 | [Observability](observability.md) |
| 구현 전 lifecycle 구상 확인하기 | [Design](design.md) |
| 실행 결과를 판정하기 | [Verification](verification.md) |

## 문서와 코드의 책임 경계

| 영역 | Canonical source |
| --- | --- |
| 공통 원격 실행 | experiments/run.py |
| 반복 Megatron 측정 | experiments/benchmarks.py와 experiments/megatron/benchmark-plan.json |
| 환경·노드 경로 | setups/spark/local.example.json과 gitignored local.json |
| 학습 조건 | experiments/trl/*.json, experiments/megatron/*.json |
| TRL 학습·데이터 | backends/trl/ |
| Megatron 학습·데이터 | backends/megatron/ |
| Profiling 도구와 예제 | observability/ |
| 회귀 검증 | tests/ |

design.md는 현재 구현의 사용 설명서가 아니라 구현 전 설계 문서입니다.
docs/verification/은 과거 실행의 manifest, log, summary와 raw measurement를 보관하는 archive입니다.
Archive의 수치는 새로운 실행의 성능이나 품질을 보증하지 않습니다.

## 권장 학습 순서

1. [Getting Started](getting-started.md)에서 setup 파일을 만들고 TRL smoke를 dry-run합니다.
2. [Datasets](datasets.md)에서 작은 pinned dataset을 준비합니다.
3. [TRL backend](backends/trl.md) 또는 [Megatron backend](backends/megatron.md)의 실행 경로를 선택합니다.
4. [Verification](verification.md)의 성공 조건으로 output과 log를 확인합니다.
5. 병목을 조사할 때만 [Observability](observability.md)를 추가합니다.
6. 기능 비교가 필요할 때 [Experiments](experiments.md)의 반복 측정을 사용합니다.

이 repository는 현재 TRL SFT와 Megatron Bridge SFT를 제공합니다.
DPO, RL trainer와 serving lifecycle은 구현된 실행 경로가 아닙니다.
