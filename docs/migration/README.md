# Repository Integration

이 문서는 backend별 장기 브랜치를 `main`으로 통합한 구조와 기존 작업의 보존 위치를 안내합니다.
환경 설정은 `setups/`, 학습 조건은 `experiments/`에서 관리하므로 환경을 추가할 때 학습 코드를 복제하지 않아도 됩니다.
기존 작업의 보존 커밋은 [preserved-branches.json](preserved-branches.json)에 기록합니다.
`archive/pre-unify-20260908/<branch>` 태그는 미커밋 작업을 포함한 보존 상태이며, `<branch>-head`는 통합 전 원래 커밋입니다.
모든 보존 태그를 remote에 push한 뒤 기존 브랜치를 정리합니다.

데이터·모델 cache와 대형 runtime 산출물은 Git에 추가하지 않고 원래 위치에 보존합니다.

## Boundaries

| 영역 | 책임 |
| --- | --- |
| `backends/trl/`, `backends/megatron/` | 기존 학습·평가·checkpoint 구현과 독립 dependency |
| `setups/spark/` | 노드, 통신, Python 환경과 실행 노드의 경로 |
| `experiments/` | 모델·데이터 revision, 학습 옵션과 반복 실행 |
| `observability/` | 자원 측정과 로그 도구 |
| `docs/design/system-integration/` | 구현 전 lifecycle·serving 설계 |
| `tests/` | backend별 회귀와 공통 실행 계약 |

Megatron RTX Setup1 전용 코드는 복원하지 않습니다.
Spark 단일 노드 실행은 topology 옵션으로 다루며 backend·parallelism 조합의 제약을 먼저 검사합니다.
구조 이동과 실행 기능 추가를 별도 커밋으로 보존합니다.

과거 실험 로그의 수치·원래 경로는 기록 그대로 유지하며 구조 이동 후 재검증 결과와 구분합니다.
성능 비교, checkpoint 재개 correctness와 durability는 각각 별도 검증입니다.

별도 Megatron review worktree의 미커밋 문서도 `archive/pre-unify-20260908/worktree-megatron-review-changes`에 보존했습니다.
나머지 detached worktree의 기준 커밋은 같은 prefix의 `worktree-*` 태그에 보존했습니다.
