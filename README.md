# Post-Training Lab

Post-Training Lab은 이미 학습된 LLM을 대상으로 SFT와 분산 post-training workflow를 실습하는 repository입니다.
데이터 준비, bounded training run, checkpoint와 resource observability를 한 구조에서 확인할 수 있습니다.
현재 공통 실행 환경은 Spark이며 실제 workload는 설정한 Spark node에서 실행됩니다.

## Start Here

처음 사용하는 경우 [문서 입구](docs/README.md)에서 학습 순서를 확인하세요.
가장 짧은 실행 경로는 [Getting Started](docs/getting-started.md)에서 setup을 만들고 TRL smoke를 dry-run하는 것입니다.

| 목적 | 시작점 |
| --- | --- |
| TRL SFT와 QLoRA | [TRL backend](docs/backends/trl.md) |
| Megatron Bridge와 checkpoint | [Megatron backend](docs/backends/megatron.md) |
| 공개·서비스 데이터 변환 | [Datasets](docs/datasets.md) |
| 반복 feature 측정 | [Experiments](docs/experiments.md) |
| GPU·host·network 관측 | [Observability](docs/observability.md) |
| 검증 결과 판정 | [Verification](docs/verification.md) |

## Repository Layout

```text
setups/spark/       Spark hosts, paths and runtime environment
experiments/        runner, backend presets and benchmark plans
backends/trl/       TRL SFT, data preparation and Spark launcher
backends/megatron/  Megatron Bridge SFT, data preparation and launcher
observability/      telemetry, baseline and profiling tools
docs/               canonical user and design documentation
docs/verification/  historical manifests, logs and measurement archive
tests/              CPU and integration regression tests
```

DPO, RL trainer, serving deployment과 lifecycle promotion API는 현재 구현 범위가 아닙니다.
[Design](docs/design.md)은 이러한 통합을 위한 구현 전 설계이며 runnable workflow가 아닙니다.
