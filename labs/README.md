# Lab Catalog and Extension Guide

현재 실행 가능한 실습과 설계 문서의 시작점은 [프로젝트 안내](../README.md#learning-paths)에서 관리합니다.
이 디렉터리는 공통 실습 작성 기준과 아직 구현되지 않은 확장 계획을 담습니다.

## Ownership

| Location | Responsibility |
| --- | --- |
| `main/labs` | 실습 목록, 준비 상태와 공통 템플릿 |
| `trl` / `megatron` | backend별 dataset 변환, 학습, 평가와 artifact 생성 |
| `system-integration` | framework-independent contract와 promotion·rollback 설계 |
| `profiling` | metric vocabulary, 수집 도구와 resource profiling 실습 |
| Observatory `docs/labs` | 합성 데이터 관측과 run 비교 실습 |

새 실습은 [템플릿](template/README.md)을 해당 구현 브랜치의 문서 디렉터리에 복사해 작성합니다.
동일 backend와 환경을 사용하면 기존 브랜치에 추가하고, 설치 환경이나 실행 lifecycle이 독립적일 때 별도 브랜치를 고려합니다.
계획만으로 빈 framework 브랜치를 만들지 않습니다.

## Planned Exercises

| Exercise | Expected learning outcome | Evidence required before marking runnable |
| --- | --- | --- |
| Preference optimization · DPO | chosen/rejected pair 준비, SFT baseline과 preference 학습 비교 | split 검증, 고정된 base model·dataset revision, 실제 학습 command와 held-out 평가 |
| Reinforcement learning | rollout, reward, policy update와 evaluation의 연결 | 재현 가능한 작은 workload, reward 정의, rollout·update 결과와 resource summary |
| Lifecycle integration | dataset revision에서 candidate 등록·promotion·rollback까지 연결 | 실제 실행 가능한 reference implementation과 실패 경로 검증 |

위 항목은 모두 계획 단계입니다.
기존 profiling의 verl 관측 예제와 Observatory의 합성 RL 화면은 RL 학습 구현을 의미하지 않습니다.

## Completion Criteria

실습 문서는 목표, 현재 상태(`planned`, `design`, `runnable`), prerequisites, 실행 command, 예상 결과, 검증 방법과 cleanup을 포함해야 합니다.
실제 GPU 실행, CPU smoke check, configuration dry run과 합성 replay 중 무엇을 검증했는지 명시합니다.

학습 실습의 작은 결과 summary에는 run ID, model·dataset revision, recipe, seed, hardware, 실행 시간과 metric 단위를 기록합니다.
원본 데이터와 큰 artifact는 저장하지 않습니다.
공통 metric 정의는 [`profiling`의 metric contract](../docs/observability/metric-schema.md)를 참조합니다.

새 실습의 실행 command와 실패 경로를 검증한 뒤 프로젝트 안내에 entry point와 검증 범위를 등록합니다.
실행하지 못한 환경은 명시하고, 예정된 기능을 실행 가능한 것으로 표시하지 않습니다.
