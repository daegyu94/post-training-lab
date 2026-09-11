# Documentation

이 문서는 처음 프로젝트를 사용하는 소프트웨어 엔지니어를 위한 안내입니다.
코드·설정·스크립트·테스트를 기준으로 현재 실행 경로와 제안·과거 관측을 구분합니다.

## Start Here

[Getting Started](getting-started.md)에서 NVIDIA DGX Spark 환경 준비·첫 실행·결과 판정까지 진행합니다.

| 목적 | 문서 |
| --- | --- |
| 환경 준비와 첫 실행 | [Getting Started](getting-started.md) |
| 구성 요소와 책임 | [Architecture](architecture.md) |
| 공개·서비스 데이터 변환 | [Datasets](datasets.md) |
| 학습 환경·backend별 제약 | [TRL](backends/trl.md), [Megatron](backends/megatron.md) |
| Preset과 반복 측정 | [Experiments](experiments.md) |
| Monitoring·baseline·trace 실행 | [Observability](observability.md) |
| 지표 계약·framework 연결 | [Observability Reference](observability-reference.md) |
| Post-training 대시보드 개선 계획 | [Dashboard Plan](dashboard-plan.md) |
| 현재 미구현 lifecycle 제안 | [Design](design.md) |
| 실습 목록·템플릿 | [Lab Catalog](../labs/README.md) |
| 판정·실패 한계·과거 기록 | [Verification](verification.md) |
| 로드맵 Step 1 범위·목표 매핑 | [Step 1](step1.md) |

이 저장소는 더 큰 multi-stage 로드맵의 **Step 1(workflow·monitoring·profiling·baseline)만** 다룹니다 — 범위와 7개 목표별 반영 현황은 [Step 1](step1.md)을 따릅니다.

## Information Boundaries

설치·현재 실행 절차는 task guide에, 지표 계약은 reference에, 실행 판정 기준과 최신 결과 요약은 [Verification](verification.md)에 둡니다.
과거 기록의 수치는 새 실행의 성능·품질을 보증하지 않습니다.
대형 checkpoint·모델 가중치는 Git에 포함되지 않습니다.

Setup과 experiment의 책임은 [Architecture](architecture.md), 모델·데이터 준비는 [Getting Started](getting-started.md)와 [Datasets](datasets.md)에서 관리합니다.
백엔드 README는 진입점만 제공하며 같은 명령을 복제하지 않습니다.
DPO, RL trainer, registry, serving 배포는 현재 구현된 workflow가 아닙니다.
