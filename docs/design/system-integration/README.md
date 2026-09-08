# System Integration

이 디렉터리는 Post-Training Lab의 공통 lifecycle 설계 문서입니다.
이 문서는 학습 전의 데이터 준비부터 학습 후 checkpoint의 production 반영까지를 연결하는 **system integration guide**입니다.
TRL과 Megatron-LM 중 어느 backend를 사용하더라도 유지해야 하는 공통 contract, 검증 단계와 운영 원칙을 설명합니다.

학습 framework 실습은 [TRL](../../backends/trl/README.md) 또는 [Megatron](../../backends/megatron/README.md) 가이드를 따릅니다.
이 설계 디렉터리에는 설치 script, 학습 command와 실행 가능한 reference implementation이 없습니다.
문서의 schema와 값은 설계 예시이며, 실제 platform의 저장소, serving engine과 승인 정책에 맞춰 구체화해야 합니다.

## Lifecycle at a Glance

![데이터 준비, 학습, 검증과 배포의 lifecycle](images/lifecycle.svg)

## Recommended Reading Order

| Order | Document | What you should understand |
| --- | --- | --- |
| 1 | [System Architecture](architecture.md) | component 경계, data flow와 최소 구현 단위 |
| 2 | [Data Lifecycle](data-lifecycle.md) | raw trace가 immutable dataset revision이 되는 과정 |
| 3 | [Checkpoint Lifecycle](checkpoint-lifecycle.md) | training request, artifact manifest, promotion과 rollback |
| 4 | [Operations](operations.md) | 공통 identifier, metric, 장애 대응과 운영 checklist |

처음 읽을 때는 위 순서대로 전체 흐름을 확인하고, 실제 system을 설계할 때 각 문서의 contract와 [운영 검증 checklist](operations.md#validation-checklist)를 구현 기준으로 사용하세요.

## Implementation Boundary

데이터 lifecycle, checkpoint promotion과 serving rollback은 설계 범위이며 실행 구현 완료를 의미하지 않습니다.
학습 구현은 `backends/`, 실행 환경은 `setups/`, 자원 측정 도구는 `observability/`에서 관리합니다.

Framework를 바꿔도 유효한 policy와 interface는 이 디렉터리에 둡니다.
특정 Python API, CLI option, configuration 또는 checkpoint 형식에 의존하는 내용은 해당 backend 디렉터리에 둡니다.
