# Observability

observability/는 host·GPU telemetry, Prometheus/Grafana validation, NCCL·fio baseline과 선택 rank PyTorch trace 예제를 제공합니다.
Framework 자체나 exporter를 재구현하지 않으며 일부 연동은 사용자가 실제 training loop에 연결해야 합니다.

## CPU 환경과 정적 검증

```bash
cd observability
./scripts/setup.sh
. .venv/bin/activate
./scripts/check_tools.sh
python -m pytest -q ../tests/observability
```

setup.sh는 observability Python helper와 pytest를 설치합니다.
CUDA PyTorch, Docker, NCCL Tests와 fio는 자동으로 설치하지 않습니다.

## Prometheus와 Grafana validation

Target JSON을 실제 endpoint로 바꾼 뒤 target file만 먼저 검사할 수 있습니다.

```bash
cd observability
. .venv/bin/activate
python -m profiling_lab.observability check-targets \
  --target-dir examples/observability/targets
```

Monitoring control plane은 Docker Compose v2가 필요합니다.
GRAFANA_ADMIN_PASSWORD를 지정하면 stack을 시작하고 readiness, health, up query를 검사합니다.

```bash
export GRAFANA_ADMIN_PASSWORD=<strong-password>
./scripts/validate_observability.sh
```

기본 validation은 target이 down이어도 stack health를 확인합니다.
실제 target도 모두 up이어야 하면 REQUIRE_TARGETS_UP=1을 추가합니다.

```bash
REQUIRE_TARGETS_UP=1 ./scripts/validate_observability.sh
```

결과는 기본적으로 artifacts/observability-validation/summary.json에 저장됩니다.
Compose service를 중지하려면 observability/examples/observability에서 docker compose down을 실행합니다.

## Spark telemetry와 trace

run_spark_observability.sh는 node 또는 server role로 실행합니다.
Node role은 node exporter와 nvidia-smi 기반 GPU sampler를 실행합니다.
Server role은 Prometheus와 Grafana를 실행하며 SPARK1_ADDR, SPARK2_ADDR, GRAFANA_ADMIN_PASSWORD가 필요합니다.

PyTorch synthetic DDP trace는 CUDA 환경에서 run_spark_profile.sh capture로 실행합니다.
양 node에서 같은 PROFILE_RUN_ID, 올바른 NODE_RANK, MASTER_ADDR와 node-local PYTHON을 설정해야 합니다.
이 예제는 실제 LLM workload가 아니라 synthetic workload입니다.

```bash
PROFILE_RUN_ID=profile-001 \
NODE_RANK=0 \
MASTER_ADDR=spark1 \
PYTHON=/path/to/cuda-python \
./scripts/run_spark_profile.sh capture
```

다른 node에서는 NODE_RANK=1로 실행합니다.
저장되는 trace와 summary는 OUTPUT_DIR로 위치를 지정할 수 있습니다.

## Baseline 범위

run_nccl_baseline.sh는 NCCL Tests의 all_reduce_perf binary를 요구합니다.
run_fio_baseline.sh는 FIO_DIRECTORY와 filesystem에 남는 test file을 요구합니다.
두 결과는 application-independent baseline이며 training throughput이나 checkpoint durability를 자동으로 설명하지 않습니다.

Metric vocabulary와 JSON Schema의 source는 observability/config/metrics.json과 metrics.schema.json입니다.
Metric이 실제로 자동 수집되는지는 exporter, framework hook과 실행 환경에 따라 달라집니다.
