# Observability

관측은 상시 지표로 이상이 발생한 시간·node·rank를 좁힌 뒤, 필요한 경우에만 짧은 trace를 수집하는 과정입니다.
host·GPU·통신·저장소 지표와 framework가 내보내는 학습 지표를 함께 보되, synthetic demo와 실제 LLM 실행 결과는 분리해 해석합니다.

## 시작점

처음에는 [분산 실행 모니터링](observability/monitoring.md)의 collector와 dashboard를 실행합니다.
문제가 보이면 [실행 분석](observability/analysis.md)에서 run history, selected-rank trace, hardware baseline을 순서대로 사용합니다.
새 지표를 추가하거나 값의 단위·범위를 판단할 때는 [Metrics Contract](observability/metrics.md)를 기준으로 합니다.

| 목적 | 문서 |
| --- | --- |
| node collector, monitoring server, dashboard, SSD health, framework metrics | [분산 실행 모니터링](observability/monitoring.md) |
| 과거 실행 요약, selected-rank PyTorch trace, NCCL baseline | [실행 분석](observability/analysis.md) |
| metric 이름·단위·scope, label, workflow phase, framework integration | [Metrics Contract](observability/metrics.md) |

## CPU Checks

저장소 루트에서 실행합니다.
`setup.sh`는 `.venv`와 pytest만 준비하며 CUDA PyTorch, NCCL Tests, Python package는 설치하지 않습니다.

```bash
cd observability
bash scripts/setup.sh
. .venv/bin/activate
bash scripts/check_tools.sh
python -m pytest -q ../tests/observability
```

미설치 도구 표시는 해당 기능을 아직 사용할 수 없다는 뜻이며, 다른 CPU 검사는 계속 실행할 수 있습니다.
