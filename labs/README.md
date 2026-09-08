# Lab Catalog

현재 실행 가능한 실습은 [문서 안내](../docs/README.md)에서 시작합니다.
이 디렉터리는 [새 실습 템플릿](template/README.md)을 제공하며 학습 구현이나 실행 preset을 복제하지 않습니다.

## Responsibilities

| 영역 | 책임 |
| --- | --- |
| `backends/` | 학습·평가·checkpoint 구현 |
| `experiments/` | 실행 조건·preset·반복 측정 |
| `setups/` | 노드·환경별 경로 |
| `observability/` | 계측·trace |
| `labs/` | 새 실습의 목표·완료 기준 |

## Planned Work

DPO, RL 학습과 lifecycle 통합은 현재 계획 단계입니다.
외부 verl 설정 예제가 RL backend 구현을 의미하지 않습니다.
새 실습에는 목표·상태·전제조건·실제 명령·예상 결과·검증·정리 절차를 포함합니다.
실행하지 못한 환경을 검증된 경로로 표시하지 않고 대형 가중치와 실행 산출물은 Git에 추가하지 않습니다.
