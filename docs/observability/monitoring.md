# Distributed Run Monitoring

이 문서는 각 node에서 host·GPU 지표를 수집하고 Prometheus와 Grafana에서 확인하는 방법을 설명합니다.
제공하는 helper script는 ARM64와 x86_64 Linux를 지원하며 특정 workload launcher나 cluster setup을 가정하지 않습니다.
관측 대상은 `이름=주소` 형식으로 지정하므로 node 구성에 맞게 확장할 수 있습니다.

## Monitoring Flow

1. 관측할 각 node에서 `node` role을 실행해 host·GPU 지표를 노출합니다.
2. monitoring host에서 `server` role을 실행해 Prometheus와 Grafana를 시작합니다.
3. browser에서 dashboard를 열고 시간 범위, cluster, node, run을 선택합니다.
4. storage node가 분리되어 있으면 `storage` role을 추가하고, framework 지표가 있으면 node role에 경로를 전달합니다.

## Start Collectors and Dashboards

### 1. Prepare Tools

`scripts/install_observability_tools.sh`는 host architecture에 맞는 ARM64 또는 x86_64 userspace 도구를 내려받으며 driver나 system package를 설치하지 않습니다.
관측할 node마다 실행하고, monitoring host에서는 `server` 인자를 추가합니다.

```bash
cd observability
bash scripts/install_observability_tools.sh
bash scripts/install_observability_tools.sh server
```

### 2. Start Collectors on Each Node

각 node의 management address를 지정합니다.
`node` role은 node exporter를 19100 포트에 열고 GPU sampler를 기본 900초 동안 실행합니다.

```bash
NODE_ADDR='<node-management-address>' \
  bash scripts/run_observability.sh node
```

### 3. Start the Monitoring Server

`OBSERVABILITY_TARGETS`는 `이름=주소` 항목을 쉼표로 연결한 일반 target 목록입니다.
node 수는 고정하지 않습니다.

```bash
CLUSTER_NAME='<cluster-name>' \
OBSERVABILITY_TARGETS='trainer-0=<first-node-address>,rollout-0=<second-node-address>' \
  bash scripts/run_observability.sh server
```

Prometheus는 monitoring host의 loopback 19090, Grafana는 loopback 13000에서 실행합니다.
작업이 끝나면 server 세션을 종료합니다.
Grafana는 anonymous Viewer이며, 관리자 계정이 필요할 때만 `GRAFANA_ADMIN_PASSWORD`를 지정합니다.
`SERVER_CONFIG_ONLY=1`은 service를 시작하지 않고 provisioning 파일만 생성합니다.

### 4. Open the Dashboards

client에서 controller를 거쳐 monitoring host로 SSH forwarding을 엽니다.

```bash
ssh -NT -L 13000:127.0.0.1:13000 -J <user>@<controller-ssh-alias> <user>@<monitoring-host>
```

브라우저에서 `http://localhost:13000`을 엽니다.
`server` role은 `examples/observability/`의 세 dashboard를 provisioning 경로로 복사합니다.

| Dashboard | 확인할 내용 |
| --- | --- |
| Run Overview (`run-overview.json`, uid `observability-overview`) | target 상태·sample age, GPU utilization matrix, rank별 throughput·step time·loss |
| Compute & Communication | GPU health·memory, rank timer, interface throughput, GPU allocation·compute topology |
| Data & Storage | node-local device·filesystem 성능, storage topology, 선택적 SSD SMART |

화면 링크는 시간·cluster·node·run 선택을 유지합니다.
같은 경로의 `grafana/`와 `compose.yaml`은 별도 Docker Compose 예시입니다.

## Synthetic Live Demo

실제 GPU나 storage를 사용하지 않고 dashboard 동작을 보여 주려면 monitoring host에서 `DEMO_LIVE=1`을 지정합니다.

```bash
cd observability
DEMO_LIVE=1 bash scripts/run_observability.sh server
```

`demo-b300` cluster는 GPU node 4개와 node당 B300 GPU 8개, storage node 8개와 node당 SSD 4개를 합성합니다.
각 node의 synthetic traffic은 800Gbps RoCE를 넘지 않으며, GPU 내부 NVLINK와 node-to-fabric RoCE 관계는 topology matrix에서 확인합니다.

세 dashboard는 `cluster`, `node`, `run_id` filter와 시간 범위를 공유합니다.
Live demo에서는 `cluster=demo-b300`, `node=All`, `run_id=live-demo`를 선택합니다.

| 관찰 구간 | 기대 변화 |
| --- | --- |
| 정상 학습 | Run Overview의 target 상태·GPU utilization·학습 지표가 갱신됨 |
| `collective` | communication timer와 RoCE traffic이 상승함 |
| `data_wait` | read·busy time이 증가함 |
| `checkpoint` | storage write가 증가함 |

### Run Overview Example

아래 30초 GIF는 `DEMO_LIVE=1`의 `demo-b300` Run Overview를 재생합니다.
Exporter target 상태, training·GPU sample age, GPU utilization matrix가 synthetic 값의 변화에 따라 갱신됩니다.
실제 LLM 학습의 측정 결과가 아니므로 dashboard 구성과 이상 구간 확인 흐름을 설명하는 용도로만 사용합니다.

![30초 Run Overview synthetic live demo](../figures/post-training-run-overview-30s.gif)

Data & Storage는 storage system·storage node·SSD filter를 추가로 제공하며, 모든 storage node를 보려면 기본 `All`을 유지합니다.
Topology panel은 전달된 연결 관계를 표시하며, 직접 측정한 link bandwidth나 endpoint 쌍별 traffic matrix는 아닙니다.

Simulator는 100초마다 정상 학습, data wait, collective 통신, checkpoint, recovery를 반복합니다.
기존 dashboard JSON은 변경하지 않으며 Prometheus가 `observability`와 `storage-smart` job으로 scrape하는 metric만 합성합니다.
`examples/live-demo/gpu_topology.yaml`과 `storage_topology.yaml`은 JSON-compatible YAML이라 추가 Python package 없이 읽습니다.
`DEMO_TOPOLOGY_DIR`, `DEMO_ADDR`, `DEMO_PORT`로 fixture와 listen address를 바꿀 수 있습니다.

해석할 때 주의할 것:

- 일부 GB10 NVML 값은 unavailable/null이며 이를 사용량 0으로 읽지 않습니다.
- 시스템 메모리와 학습 process의 CUDA allocated/reserved peak를 구분합니다.
- GPU 행렬은 sampler가 보고한 index와 utilization이며 allocation이 아닙니다.
- 통신 패널은 interface·RDMA port counter이고 endpoint 쌍별 traffic matrix가 아닙니다.
- Data & Storage의 device·filesystem 지표를 특정 run의 단독 사용량으로 읽지 않습니다.

## SSD Health

SSD health는 실험별 write attribution이 아니라 장치 이상과 장기 열화를 확인하는 선택 기능입니다.
`smartctl_exporter`는 기본 설치 script에 포함되지만 `smartctl`은 운영체제의 `smartmontools` package로 설치해야 합니다.

일부 NVMe 환경에서는 controller device(`/dev/nvmeN`)의 admin-passthrough ioctl에 root 권한이 필요합니다.
`node`와 `storage` role은 이를 감지해 필요하면 passwordless sudo로 `smartctl`만 실행하며 exporter 자체를 root로 올리지 않습니다.
sudo 경로에서는 시작 전에 `sudo -n smartctl --scan`으로 비대화식 권한을 확인하고, 실패하면 실행을 중단합니다.
이 검사는 장치별 SMART 조회 성공을 보장하지 않으므로 exporter log와 실제 SMART metric도 확인합니다.
자동 감지를 강제로 켜거나 끄려면 `SMARTCTL_SUDO=1` 또는 `SMARTCTL_SUDO=0`을 지정합니다.

로컬 SSD를 관측하려면 기존 node role에 기능을 추가합니다.

```bash
NODE_ADDR='<node-management-address>' \
ENABLE_SSD_HEALTH=1 \
  bash scripts/run_observability.sh node
```

storage node가 분리된 구성에서는 GPU sampler를 실행하지 않는 `storage` role을 각 storage node에서 사용합니다.
helper script가 맞지 않는 host에서는 `smartctl_exporter`를 따로 설치하고 `SMARTCTL_EXPORTER`에 실행 파일 경로를 지정합니다.

```bash
NODE_ADDR='<storage-node-management-address>' \
SMARTCTL_EXPORTER='<smartctl-exporter-path>' \
  bash scripts/run_observability.sh storage
```

기본 exporter port는 19633이고 SMART 조회 주기는 60초입니다.
필요하면 `SMARTCTL_PORT`, `SMARTCTL_INTERVAL`, `SMARTCTL_EXPORTER`, `SMARTCTL`을 지정합니다.
짧은 scrape 주기를 사용해도 SSD firmware가 내부 SMART 값을 같은 주기로 갱신한다는 보장은 없습니다.

Monitoring server에는 compute node와 별도로 storage target을 전달합니다.

```bash
CLUSTER_NAME='<cluster-name>' \
OBSERVABILITY_TARGETS='trainer-0=<trainer-address>,trainer-1=<trainer-address>' \
STORAGE_SYSTEM='3fs' \
STORAGE_TARGETS='storage-0=<storage-address>,storage-1=<storage-address>' \
  bash scripts/run_observability.sh server
```

`STORAGE_SYSTEM`은 `local`, `3fs`, `pnfs`처럼 배치를 식별하는 값입니다.
Data & Storage dashboard는 `cluster → storage system → storage node → SSD` 순서로 필터링합니다.
분산 filesystem의 replication이나 data-server layout 때문에 client write와 개별 SSD write는 일대일로 대응하지 않으므로 health metric을 특정 run이나 client에 귀속하지 않습니다.
Storage topology의 component·edge와 SMART 시계열의 `instance` label은 같은 storage node 이름을 사용합니다.

Dashboard는 NVMe critical warning·collection status·현재 온도·percentage used·available spare·unrecovered media error·lifetime host bytes written을 표시합니다.
Critical warning은 정상 metric이 있으면 0, metric 자체가 없으면 N/A로 표시됩니다.
`smartctl_device_bytes_written`은 host가 controller에 기록한 누계이며, 내부 NAND write나 SSD write amplification을 뜻하지 않습니다.
`percentage_used`는 짧은 실행에서 변하지 않을 수 있으므로 장기 추세에 사용합니다.

Docker Compose 예시는 `targets/storage.json`을 읽습니다.
exporter를 배치한 뒤 target과 topology label을 다음처럼 추가합니다.

```json
[
  {
    "targets": ["storage-0.example:19633"],
    "labels": {
      "cluster": "<cluster-name>",
      "storage_system": "pnfs",
      "instance": "storage-0"
    }
  }
]
```

## Live Framework Metrics

`node` role에 launcher와 같은 `FRAMEWORK_METRICS_DIR`를 지정하면 [`framework_metrics_textfile`](../../observability/profiling_lab/framework_metrics_textfile.py)이 rank JSON을 주기적으로 읽습니다.
`training_loss`, `training_tokens_per_second`, `training_step_time_seconds`, `training_step`과 Megatron의 `training_timer_seconds{timer="..."}`를 GPU sampler와 같은 textfile collector에 씁니다.

```bash
NODE_ADDR='<node-management-address>' \
FRAMEWORK_METRICS_DIR='<launcher output-directory>/framework-metrics' \
  bash scripts/run_observability.sh node
```

생략하면 framework 수집만 꺼지고 host/GPU monitoring은 계속됩니다.
각 node의 node-local 경로를 사용해야 하며, 두 node가 같은 NFS 파일을 읽으면 같은 rank가 두 `instance`로 중복 노출됩니다.

GPU allocation matrix는 framework snapshot에 `CUDA_VISIBLE_DEVICES`와 `LOCAL_RANK`가 있는 rank만 표시합니다.
행은 cluster·node·run·framework·rank로 구분하며 열의 device ID는 보고된 index 또는 UUID 그대로입니다.
GPU utilization matrix의 index와 UUID를 자동 매칭하지 않습니다.
GPU 지표는 30초, 학습 지표·할당은 `Training sample max age (s)`(기본 300초)를 넘으면 숨깁니다.
긴 step에서는 이 값을 늘리며 오래되거나 없는 할당 정보를 GPU가 비어 있다는 뜻으로 해석하지 않습니다.
Overview의 rank별 sample age는 오래된 값도 표시하므로 일부 rank의 갱신 중단을 확인할 수 있습니다.

`TOPOLOGY_DIR`를 node role에 지정하면 `compute-topology.json`·`storage-topology.json`의 component와 edge를 Grafana에 표시합니다.
전달된 관계를 보여 줄 뿐 bandwidth·latency 측정값은 아니며 topology는 Cluster filter만 적용합니다.

dashboard JSON을 갱신한 뒤에는 monitoring host의 `server` role을 다시 실행해야 합니다.
수집기 변경은 각 node의 `node` role을 재시작해 적용합니다.
