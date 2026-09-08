# Lab 01: Cluster Telemetry

## Goal

모든 training node의 host와 NVIDIA GPU metric을 중앙 Prometheus가 수집하고, 제공된 Grafana dashboard에서 node/GPU imbalance를 확인합니다.
이 단계에는 training code 수정이나 profiler 활성화가 필요하지 않습니다.

## Prerequisites

- monitoring host에서 Compose-compatible container runtime 사용 가능
- Prometheus가 각 compute node의 TCP 9100과 9400에 접근 가능
- 각 compute node에 NVIDIA driver와 호환되는 DCGM Exporter 배치 가능
- production 적용 전 exporter endpoint의 인증, TLS와 network policy 별도 설계

## 1. Run Exporters on Every Compute Node

Node Exporter는 각 node에서 host namespace를 볼 수 있도록 systemd 또는 container로 실행합니다.
아래 image tag는 예시이므로 조직에서 검증한 immutable version으로 고정합니다.

```bash
export NODE_EXPORTER_IMAGE=quay.io/prometheus/node-exporter:v1.9.1

docker run -d \
  --name node-exporter \
  --network host \
  --pid host \
  --restart unless-stopped \
  -v /:/host:ro,rslave \
  "$NODE_EXPORTER_IMAGE" \
  --path.rootfs=/host
```

DCGM Exporter도 각 GPU node에 하나씩 실행합니다.
DCGM/driver compatibility matrix에 맞는 image tag를 선택합니다.

```bash
export DCGM_EXPORTER_IMAGE=nvcr.io/nvidia/k8s/dcgm-exporter:<validated-tag>

docker run -d \
  --name dcgm-exporter \
  --network host \
  --gpus all \
  --cap-add SYS_ADMIN \
  --restart unless-stopped \
  "$DCGM_EXPORTER_IMAGE"
```

각 node에서 endpoint를 확인합니다.

```bash
curl --fail http://127.0.0.1:9100/metrics
curl --fail http://127.0.0.1:9400/metrics
```

Kubernetes에서는 standalone container 대신 Node Exporter와 DCGM Exporter DaemonSet, Prometheus Operator의 `ServiceMonitor`를 사용합니다.
NVIDIA GPU Operator가 이미 DCGM Exporter를 관리한다면 중복 배치하지 않습니다.

## 2. Configure Targets

[`nodes.json`](../../examples/observability/targets/nodes.json)과 [`gpus.json`](../../examples/observability/targets/gpus.json)의 주소를 실제 node로 바꾸고 `cluster`와 `run_id` label을 설정합니다.
allocation이 바뀔 때마다 static file을 손으로 관리하기보다 Slurm host list 또는 Kubernetes service discovery에서 생성하는 것이 production에 적합합니다.

## 3. Start Prometheus and Grafana

```bash
export GRAFANA_ADMIN_PASSWORD=<strong-password>
REQUIRE_TARGETS_UP=1 ./scripts/validate_observability.sh
```

script가 성공하면 target file, Compose configuration, Prometheus와 Grafana health, 기본 Prometheus query와 모든 configured target을 확인한 것입니다.
결과는 `artifacts/observability-validation/summary.json`에 저장됩니다.
이후 Grafana의 `Profiling Lab / Multinode LLM Cluster Resources` dashboard를 엽니다.

## 4. Exercises

Prometheus expression browser에서 다음 query를 실행합니다.

CPU busy percentage:

```promql
100 * (1 - avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[1m])))
```

GPU utilization spread:

```promql
max(DCGM_FI_DEV_GPU_UTIL) - min(DCGM_FI_DEV_GPU_UTIL)
```

Node별 network receive throughput:

```promql
sum by (instance) (rate(node_network_receive_bytes_total{device!~="lo|docker.*|veth.*"}[1m]))
```

Disk I/O busy ratio:

```promql
rate(node_disk_io_time_seconds_total[1m])
```

실제 workload를 실행하며 다음을 확인합니다.

- 특정 node/GPU만 지속적으로 낮은 utilization을 보이는가?
- 낮은 GPU utilization 시점에 CPU, NIC 또는 disk가 포화되는가?
- power/clock/throttling과 throughput 저하가 같은 시점에 발생하는가?
- job이 끝난 뒤에도 같은 label의 unrelated process traffic이 섞이지 않는가?

## Expected Result

한 dashboard에서 allocation 전체의 node/GPU 상태를 비교할 수 있고 missing target과 실제 0 utilization을 구분할 수 있어야 합니다.
여기서 찾은 이상 시점과 node가 이후 Megatron/verl semantic metric과 selected trace의 대상이 됩니다.
