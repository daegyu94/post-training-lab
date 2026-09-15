# Observability

이 문서는 분산 학습 실행의 상태와 성능을 관측하는 방법을 설명합니다.
먼저 host·GPU·통신·저장소와 학습 지표에서 이상이 발생한 시간·node·rank를 찾고, 원인 분석이 필요할 때만 해당 구간의 짧은 trace를 수집합니다.
Synthetic demo는 dashboard 동작을 보여 주기 위한 예시이며 실제 LLM 학습 결과와 구분합니다.

## 시작점

분산 실행을 관측하려면 먼저 [분산 실행 모니터링](observability/monitoring.md)에서 collector와 dashboard 설정 방법을 확인합니다.
이상이 발견되면 [실행 분석](observability/analysis.md)에 따라 run history를 확인하고, 필요한 구간의 selected-rank trace를 수집하거나 hardware baseline과 비교합니다.
지표를 추가하거나 해석할 때는 [Metrics Contract](observability/metrics.md)에 정의된 이름·단위·측정 범위를 따릅니다.

| 목적 | 문서 |
| --- | --- |
| node collector, monitoring server, dashboard, SSD health, framework metrics | [분산 실행 모니터링](observability/monitoring.md) |
| 과거 실행 요약, selected-rank PyTorch trace, NCCL baseline | [실행 분석](observability/analysis.md) |
| metric 이름·단위·scope, label, workflow phase, framework integration | [Metrics Contract](observability/metrics.md) |

## 로컬 검증

GPU workload를 실행하기 전에 기본 도구 상태와 observability 테스트를 로컬 환경에서 확인합니다.
다음 명령은 저장소 루트에서 실행합니다.
`setup.sh`는 `.venv`와 pytest만 준비하며 CUDA PyTorch, NCCL Tests, Python package는 설치하지 않습니다.

```bash
cd observability
bash scripts/setup.sh
. .venv/bin/activate
bash scripts/check_tools.sh
python -m pytest -q ../tests/observability
```

미설치 도구 표시는 해당 기능을 아직 사용할 수 없다는 뜻이며 다른 로컬 검사는 계속 실행할 수 있습니다.
