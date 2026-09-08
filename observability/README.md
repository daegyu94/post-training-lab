# Large-scale LLM Post-training Resource Profiling Lab

이 저장소는 Megatron 또는 verl 기반 post-training workload를 멀티노드에서 실행할 때 resource bottleneck을 찾는 오픈소스 profiling 실습입니다.
특정 command의 부모 PID를 sampling하는 wrapper 대신, 실제 cluster의 node, GPU, network, storage, distributed rank와 agentic rollout을 같은 run으로 연결합니다.

## Start Here

처음에는 **01 → 02 → 06** 순서로 telemetry, hardware baseline, 작은 DDP trace를 익힙니다.
그다음 실제 workload에 따라 **03 (Megatron)** 또는 **04 (verl)**를 선택하고, 더 자세한 증거가 필요할 때 **05 (selected trace)**를 적용합니다.
Lab 번호는 문서 식별자이며 실행 순서와는 다릅니다.

```bash
git clone -b profiling https://github.com/daegyu94/post-training-lab.git
cd post-training-lab
./scripts/setup.sh
. .venv/bin/activate
python -m pytest -q
```

`setup.sh`는 CPU 검증용 environment와 pytest만 설치합니다.
Docker/Compose, exporter, CUDA용 PyTorch, NCCL Tests, fio와 framework는 각 실습의 실행 호스트에 별도로 준비합니다.
아래 명령은 `observability` 디렉터리를 current working directory로 가정합니다.
Spark cluster에서는 controller가 실행을 조율하고 실제 DDP·LLM 연산은 `spark1`, `spark2`에서 수행합니다.
공유 저장소 경로는 controller에서 `/home/daegyu/shared/post-training-lab`, Spark 노드에서 `/home/spark/shared/post-training-lab`입니다.
Lab 06의 2-GPU 및 Slurm 예제는 일반적인 구성 예시이므로 실제 GPU 수와 launcher에 맞춰 적용합니다.

## What Is Included

| 구성 | 제공 범위 | 사용자가 연결할 부분 |
| --- | --- | --- |
| Cluster telemetry | Prometheus·Grafana Compose, target 예제, host/GPU dashboard, health 검증 | 각 node의 exporter 설치와 실제 target 주소 |
| Hardware baseline | NCCL Tests·fio 실행 script와 raw 결과 저장 | benchmark binary, node allocation, storage 경로 |
| DDP trace | synthetic workload, selected-rank helper, Chrome trace JSON | CUDA PyTorch와 분산 실행 환경 |
| Megatron | 기존 timer 값을 `.prom`으로 내보내는 hook 예제 | training logging loop 호출, textfile collector 설정 |
| verl / agent loop | 기존 framework 설정과 계측 가이드 | 실제 endpoint, role dashboard, OpenTelemetry 계측 |
| Metric contract | 공통 이름·단위·phase 정의와 validator | 원본 metric 변환, phase marker, derived metric 계산 |

Metric contract에 나열된 항목이 모두 자동 수집되는 것은 아닙니다.
기본 dashboard는 host/GPU resource를 보여주며, role·phase 분석과 Tempo/OpenTelemetry 배포는 추가 통합 범위입니다.

## How Evidence Flows

![compute node의 metric 수집과 별도 trace 파일 분석 경로](../docs/observability/profiling-architecture.svg)

Prometheus는 각 node의 exporter endpoint를 주기적으로 읽고, Grafana는 Prometheus를 조회합니다.
선택한 rank의 trace와 NCCL/fio 결과는 별도 파일로 남으며 Prometheus에 자동으로 들어가지 않습니다.
`run_id`, 실행 시간대와 rank map을 보존해 metric과 파일을 함께 해석합니다.
도구별 관측 범위는 [tool 선택과 사각지대](../docs/observability/tooling.md)를 참고하세요.

![baseline에서 selected trace와 재검증으로 이어지는 실습 흐름](../docs/observability/profiling-workflow.svg)

## Metric Contract and Data Movement

![전송 경로별 관측 신호와 추가 계측이 필요한 부분](../docs/observability/data-movement-profiling.svg)

Storage, host memory, GPU와 node 간 전송은 workflow phase와 path별로 구분합니다.
Exporter는 node/device 전체 신호를 제공하고, framework marker와 selected trace가 해당 구간의 의미를 보완합니다.
Bytes와 duration을 실제로 얻은 경로에만 effective bandwidth를 계산합니다.
Canonical metric은 [`config/metrics.json`](config/metrics.json)에 정의하며, 작성 규칙은 [Profiling metric contract](../docs/observability/metric-schema.md)에서 설명합니다.

## Public Dashboard Demo

합성 데이터 기반의 profiling dashboard 데모는 [Post-Training Lab Observatory](https://daegyu94.github.io/post-training-lab-observatory/)에서 확인할 수 있습니다.
실제 exporter, cluster, training run에는 연결하지 않으며, 실습과 profiling 구성은 `observability/`에서 제공합니다.

## Labs

| Lab | Outcome |
| --- | --- |
| [01. Cluster telemetry](../docs/observability/labs/01-cluster-telemetry.md) | 실제 멀티노드의 host/GPU metric을 Prometheus와 Grafana에서 조회 |
| [02. Hardware baselines](../docs/observability/labs/02-hardware-baselines.md) | NCCL과 storage 성능의 application-independent 기준 확보 |
| [03. Megatron profiling](../docs/observability/labs/03-megatron.md) | parallel rank imbalance, communication, pipeline bubble과 checkpoint 병목 분석 |
| [04. verl agentic RL profiling](../docs/observability/labs/04-verl.md) | rollout, training role, Ray scheduling과 tool/environment wait 분석 |
| [05. Selected trace](../docs/observability/labs/05-selected-trace.md) | 이상 rank와 짧은 step window만 trace하고 HTA/Perfetto로 분석 |
| [06. Distributed PyTorch profiling](../docs/observability/labs/06-distributed-pytorch.md) | 작은 DDP workload로 single-node에서 multi-node까지 profiler 흐름을 검증 |

처음에는 Lab 01, 02와 Lab 06을 순서대로 실행합니다.
이후 실제 Megatron 또는 verl run에 맞는 Lab 03, 04를 적용하고, 이상이 발견된 경우에만 Lab 05로 들어갑니다.

## Validate the Monitoring Stack

`examples/observability/targets/*.json`의 예시 주소를 실제 compute node 주소로 바꾼 후 `observability` 디렉터리에서 validation을 실행합니다.

```bash
export GRAFANA_ADMIN_PASSWORD=<strong-password>
./scripts/validate_observability.sh
```

script는 target file과 Compose configuration을 검사하고 Prometheus와 Grafana를 시작한 뒤 readiness, Grafana database health, Prometheus `up` query와 active target 상태를 확인합니다.
결과는 `artifacts/observability-validation/summary.json`에 공통 summary schema로 저장합니다.

기본 실행은 monitoring control plane만 검증하므로 exporter target이 `DOWN`이어도 상태를 기록하고 실패로 처리하지 않습니다.
각 training node에 exporter를 배치한 뒤 모든 configured target까지 검증하려면 다음을 실행합니다.

```bash
REQUIRE_TARGETS_UP=1 ./scripts/validate_observability.sh
```

Prometheus는 `http://<monitoring-host>:9090`, Grafana는 `http://<monitoring-host>:3000`에서 확인합니다.
validation 후 service는 계속 실행됩니다.
종료하려면 `examples/observability`에서 `docker compose down`을 실행합니다.
이 Compose 예제는 monitoring control plane만 실행하며, exporter는 각 training node에서 별도로 배치해야 합니다.

## Repository Scope

- `examples/observability`: Prometheus file discovery, Grafana provisioning과 resource dashboard
- `examples/pytorch`: 선택 rank/step용 PyTorch Profiler helper와 DDP demo
- `scripts/check_tools.sh`: 오픈소스 도구와 vendor fallback의 설치 여부 확인
- `scripts/validate_observability.sh`: monitoring stack 실행과 validation orchestration
- `config/metrics.json`: framework에 관계없는 metric vocabulary, phase와 collection policy
- `config/metrics.schema.json`: metric contract의 JSON Schema
- `docs/metric-schema.md`: label, manifest field, phase와 data movement metric 작성 규칙
- `profiling_lab/observability.py`: target, readiness, health와 query validation
- `profiling_lab/schema.py`: metric schema validation
- `run_summary.py`: framework 공통 summary schema

이 저장소는 Megatron, verl, Ray 또는 exporter 자체를 재구현하지 않습니다.
workload별 adapter는 framework가 이미 제공하는 timer와 metric을 재사용하고, 없는 semantic signal만 얇게 추가하는 것을 원칙으로 합니다.

## Validate the Repository

GPU나 exporter 없이 Python helper, metric schema, shell script syntax를 확인할 수 있습니다.

```bash
.venv/bin/python -m pytest -q
for script in scripts/*.sh; do bash -n "$script" || exit; done
```
