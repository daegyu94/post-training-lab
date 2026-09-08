# Documentation

이 문서는 처음 프로젝트를 사용하는 소프트웨어 엔지니어를 위한 안내입니다.
코드·설정·스크립트·테스트를 기준으로 현재 실행 경로와 제안·과거 관측을 구분합니다.

## Start Here

[Getting Started](getting-started.md)에서 GPU 없는 계획 확인부터 환경 준비·첫 실행·결과 판정까지 진행합니다.
데이터 변환만 먼저 확인하려면 [가상 서비스 기록 예제](datasets.md#reviewed-service-traces)를 사용합니다.

| 목적 | 문서 |
| --- | --- |
| 환경 준비와 첫 실행 | [Getting Started](getting-started.md) |
| 구성 요소와 책임 | [Architecture](architecture.md) |
| 공개·서비스 데이터 변환 | [Datasets](datasets.md) |
| 학습 환경·backend별 제약 | [TRL](backends/trl.md), [Megatron](backends/megatron.md) |
| Preset과 반복 측정 | [Experiments](experiments.md) |
| Monitoring·baseline·trace 실행 | [Observability](observability.md) |
| 지표 계약·framework 연결 | [Observability Reference](observability-reference.md) |
| 현재 미구현 lifecycle 제안 | [Design](design.md) |
| 판정·실패 한계·과거 기록 | [Verification](verification.md) |

## Information Boundaries

설치·현재 실행 절차는 task guide에, 지표 계약은 reference에, 측정 당시 조건은 `verification/`에 둡니다.
과거 기록의 수치는 새 실행의 성능·품질을 보증하지 않습니다.
대형 checkpoint·모델 가중치는 Git에 포함되지 않습니다.

Setup과 experiment의 책임은 [Architecture](architecture.md), 모델·데이터 준비는 [Getting Started](getting-started.md)와 [Datasets](datasets.md)에서 관리합니다.
백엔드 README는 진입점만 제공하며 같은 명령을 복제하지 않습니다.
DPO, RL trainer, registry, serving 배포는 현재 구현된 workflow가 아닙니다.
