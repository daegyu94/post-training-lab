# Observability Tools

Host·GPU 계측, Spark monitoring, NCCL baseline과 PyTorch trace 예제입니다.
실행은 [Observability](../docs/observability.md), 지표 계약과 framework 연결은 [Reference](../docs/observability-reference.md)를 따릅니다.

| 경로 | 역할 |
| --- | --- |
| `config/metrics.json` | 지표 이름·단위·수집 범위 |
| `examples/pytorch/` | Synthetic DDP와 selected-rank trace |
| `profiling_lab/` | 계측과 Grafana textfile bridge, 서버 없는 run history CLI(`show_run`) |
| `scripts/` | 환경 준비·수집·baseline 실행 |

Framework hook은 사용자가 연결하는 예제이며 모든 LLM 학습에 자동 적용되지 않습니다.
Synthetic 결과와 실제 학습 결과는 구분합니다.
