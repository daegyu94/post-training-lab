# Resource Profiling Lab

관측 기능의 실행 방법은 [Observability guide](../docs/observability.md)에 모읍니다.
이 directory에는 telemetry helper, Prometheus/Grafana example, NCCL·fio baseline과 PyTorch profiler example이 있습니다.

```text
config/metrics.json                 metric vocabulary
examples/observability/             Compose and Prometheus configuration
examples/pytorch/                   synthetic DDP and selected-rank trace
profiling_lab/                      validation and telemetry modules
scripts/                            setup and profiling launchers
```

이 directory의 도구는 실제 LLM training과 별도의 synthetic 또는 framework adapter 경로입니다.
실행 결과는 해당 run의 model, topology와 host 환경에 한정해 해석합니다.
