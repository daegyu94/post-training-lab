# Lab Catalog

이 디렉터리는 새 실습을 추가할 때 사용할 기준과 아직 구현되지 않은 확장 계획을 담습니다.
현재 실행 가능한 workflow의 시작점은 [문서 입구](../docs/README.md)입니다.

## 책임 경계

| 영역 | 책임 |
| --- | --- |
| backends/ | 실제 학습·평가·checkpoint 구현 |
| experiments/ | 실행 조건, preset과 반복 측정 |
| setups/ | node와 runtime path |
| observability/ | 자원 측정과 trace 도구 |
| labs/ | 새 실습의 목표와 완료 기준 |

## 계획 단계인 항목

DPO, reinforcement learning과 lifecycle integration은 현재 planned 상태입니다.
Observability의 verl 문서나 합성 화면은 RL training 구현을 의미하지 않습니다.

새 실습은 목표, status, prerequisites, 실제 command, expected result, verification과 cleanup을 포함해야 합니다.
실행하지 못한 환경은 runnable로 표시하지 않습니다.
큰 model weight와 runtime artifact는 Git에 추가하지 않습니다.

