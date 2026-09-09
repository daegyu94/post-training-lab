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
| 현재 미구현 lifecycle 제안 | [Design](design.md) |
| 실습 목록·템플릿 | [Lab Catalog](../labs/README.md) |
| 판정·실패 한계·과거 기록 | [Verification](verification.md) |

## Roadmap Scope: Step 1

이 저장소는 더 큰 multi-stage 로드맵(Step 1 → Step 2-1 medium-scale distributed post-training → Step 2-2 long-running agent workload → Step 3 1T+ model·4+ node Agentic RL)의 **Step 1(workflow·monitoring·profiling·baseline)만** 다룹니다.
Step 2 이후의 distributed filesystem 선정, train/rollout 분리, agent workload, RL 학습은 이 저장소의 범위가 아니며 별도 단계에서 다룹니다.

| Step 1 목표 | 상태 | 근거 |
| --- | --- | --- |
| GPU/CPU memory·utilization·disk I/O 자동 수집 + 공통 지표 정의 | 완전 반영 | [Observability](observability.md), `observability/config/metrics.json` |
| Container 포함 sandbox 자원 사용 규칙 수립 | 부분 반영 (CPU만 검증) | [Sandbox resource limits](../labs/sandbox-resource-limits/README.md) |
| 30B급 모델 GPU/CPU memory·storage baseline | 완전 반영 | [Verification](verification.md), [30B NVMe](../labs/nvme-30b/README.md) |
| 모델 확장 시 local disk vs remote storage pool 판단 기준 | 부분 반영 (memory-tier 근거는 실측, storage-topology 방향은 설계 논의만) | [30B NVMe: Why NVMe Offload Is Necessary](../labs/nvme-30b/README.md#why-nvme-offload-is-necessary-here), [Direction at Larger Scale](../labs/nvme-30b/README.md#direction-at-larger-scale-local-offload-remote-checkpoint) |
| 이후 단계에 재사용 가능한 profiling·analysis 기술 확보 | 완전 반영 | [Observability](observability.md) |
| 중형 모델급 환경 → 향후 simulation·scaling study baseline | 부분 반영 (방법론만, 모델 크기 다변화 없음) | [Experiments](experiments.md)의 Repeated Megatron Measurements |
| SFT 대표 workload로 end-to-end·distributed 확장 경로 검증 | 완전 반영 | [Verification](verification.md) |

"부분 반영"인 항목들은 알려진 gap이며, 각 문서의 Limitations 절(또는 해당 절)에 원인과 남은 범위가 적혀 있습니다.
Local disk vs remote storage pool 항목의 실제 throughput·contention 실측과 distributed filesystem 선택은 의도적으로 보류했습니다 — 그 결론은 checkpoint·node 규모가 훨씬 커지는 로드맵 Step 2-1/Step 3에서 의미가 생깁니다.

## Information Boundaries

설치·현재 실행 절차는 task guide에, 지표 계약은 reference에, 실행 판정 기준과 최신 결과 요약은 [Verification](verification.md)에 둡니다.
과거 기록의 수치는 새 실행의 성능·품질을 보증하지 않습니다.
대형 checkpoint·모델 가중치는 Git에 포함되지 않습니다.

Setup과 experiment의 책임은 [Architecture](architecture.md), 모델·데이터 준비는 [Getting Started](getting-started.md)와 [Datasets](datasets.md)에서 관리합니다.
백엔드 README는 진입점만 제공하며 같은 명령을 복제하지 않습니다.
DPO, RL trainer, registry, serving 배포는 현재 구현된 workflow가 아닙니다.
