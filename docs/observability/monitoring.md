# Distributed Run Monitoring

각 node의 host·GPU 지표를 수집하고 Prometheus와 Grafana에서 확인하는 방법을 설명합니다.
helper script는 ARM64와 x86_64 Linux를 지원하며 특정 workload launcher나 cluster setup을 가정하지 않습니다.
관측 대상은 `이름=주소` 형식으로 지정하므로 node 구성에 맞게 확장할 수 있습니다.

## Monitoring Flow

1. 각 node에서 collector를 실행합니다.
2. 필요하면 같은 node에서 framework metrics와 topology 수집을 추가합니다.
3. monitoring host에서 Prometheus와 Grafana를 시작합니다.
4. GUI가 있는 client에서 Grafana dashboard를 엽니다.

## Start Monitoring

### 1. Prepare Tools

`install_observability_tools.sh`는 host architecture에 맞는 userspace 도구를 내려받습니다.
driver와 system package는 설치하지 않습니다.

관측할 node에서는 기본 도구를, monitoring host에서는 server 도구를 설치합니다.

```bash
cd observability
bash scripts/install_observability_tools.sh
```

```bash
cd observability
bash scripts/install_observability_tools.sh server
```

### 2. Start Collectors on Each Node

`node` role은 다음 process를 시작합니다.

- node exporter: host 지표를 19100 포트에 노출
- GPU sampler: GPU 지표를 수집하고 textfile metric과 JSONL을 생성
- 선택 기능: framework metrics, topology, local SSD health

GPU sampler의 기본 실행 시간은 15분입니다.
시간이 지나면 sampler가 종료되고 script가 함께 시작한 exporter와 선택적 collector를 정리한 뒤 `node` role도 종료됩니다.
`DURATION`은 초 단위로 변경할 수 있습니다.

```bash
NODE_ADDR='<node-management-address>' \
DURATION=3600 \
  bash scripts/run_observability.sh node
```

#### Live Framework Metrics

학습 loss·처리량·step time을 dashboard에 표시하려면 launcher가 쓰는 framework metrics 디렉터리를 같은 node의 collector에 전달합니다.

```bash
NODE_ADDR='<node-management-address>' \
FRAMEWORK_METRICS_DIR='<launcher-output>/framework-metrics' \
DURATION=3600 \
  bash scripts/run_observability.sh node
```

[`framework_metrics_textfile`](../../observability/profiling_lab/framework_metrics_textfile.py)은 rank JSON을 읽어 다음 metric을 node exporter의 textfile collector로 전달합니다.

- 공통: `training_loss`, `training_tokens_per_second`, `training_step_time_seconds`, `training_step`
- Megatron: `training_timer_seconds{timer="..."}`

`FRAMEWORK_METRICS_DIR`를 생략하면 framework metrics만 수집하지 않으며 host·GPU monitoring은 계속됩니다.
각 node에는 node-local 경로를 지정해야 합니다.
여러 node가 같은 NFS 디렉터리를 읽으면 동일한 rank가 여러 `instance`에 중복됩니다.

Dashboard의 freshness 처리는 다음과 같습니다.

| 대상 | 기본 유효 시간 | 시간이 지난 뒤 |
| --- | --- | --- |
| GPU metric | 30초 | panel에서 숨김 |
| 학습 metric·GPU allocation | `Training sample max age (s)` 300초 | panel에서 숨김 |
| rank별 sample age | 제한 없이 표시 | 갱신이 중단된 rank 확인에 사용 |

긴 step에서는 `Training sample max age (s)`를 늘립니다.
오래되거나 없는 allocation 정보를 GPU가 비어 있다는 뜻으로 해석하지 않습니다.

`TOPOLOGY_DIR`를 지정하면 `compute-topology.json`과 `storage-topology.json`의 component·edge를 Grafana에 표시합니다.
이 정보는 연결 관계일 뿐 bandwidth나 latency 측정값은 아닙니다.

### 3. Start the Monitoring Server

`OBSERVABILITY_TARGETS`에 `이름=주소` 항목을 쉼표로 연결합니다.
node 수와 역할은 고정하지 않습니다.

```bash
CLUSTER_NAME='<cluster-name>' \
OBSERVABILITY_TARGETS='trainer-0=<first-node-address>,rollout-0=<second-node-address>' \
  bash scripts/run_observability.sh server
```

| 항목 | 동작 |
| --- | --- |
| Prometheus | monitoring host의 `127.0.0.1:19090`에서 실행 |
| Grafana | monitoring host의 `127.0.0.1:13000`에서 실행 |
| 기본 접근 권한 | anonymous Viewer |
| 관리자 비밀번호 | 필요할 때만 `GRAFANA_ADMIN_PASSWORD`로 지정 |
| 설정만 생성 | `SERVER_CONFIG_ONLY=1`이면 service를 시작하지 않고 provisioning 파일만 생성 |
| 종료 | server를 실행한 terminal에서 세션을 종료하면 함께 시작한 service를 정리 |

### 4. Open the Dashboards

controller에는 GUI browser가 없으므로, browser가 있는 client에서 controller를 jump host로 사용합니다.
다음 명령은 monitoring host의 Grafana port를 client의 `localhost:13000`으로 전달합니다.

```bash
ssh -NT -L 13000:127.0.0.1:13000 -J <user>@<controller-ssh-alias> <user>@<monitoring-host>
```

client browser에서 `http://localhost:13000`을 엽니다.

| Dashboard | 확인할 내용 |
| --- | --- |
| Run Overview (`run-overview.json`, uid `observability-overview`) | target 상태·sample age, GPU utilization matrix, rank별 throughput·step time·loss |
| Compute & Communication | GPU health·memory, rank timer, interface throughput, GPU allocation·compute topology |
| Data & Storage | node-local device·filesystem 성능, storage topology, 선택적 SSD SMART |

화면 링크는 시간·cluster·node·run 선택을 유지합니다.
`server` role은 `examples/observability/`의 세 dashboard를 provisioning 경로로 복사합니다.
같은 경로의 `grafana/`와 `compose.yaml`은 별도 Docker Compose 예시입니다.

## Synthetic Live Demo

실제 GPU나 storage 없이 dashboard 동작을 확인하려면 monitoring host에서 실행합니다.

```bash
cd observability
DEMO_LIVE=1 bash scripts/run_observability.sh server
```

| 구성 | Demo 값 |
| --- | --- |
| Cluster | `demo-b300` |
| Compute | GPU node 4개, node당 B300 GPU 8개 |
| Storage | storage node 8개, node당 SSD 4개 |
| Network | node당 synthetic traffic 최대 800Gbps RoCE |
| 반복 주기 | 정상 학습 → data wait → collective → checkpoint → recovery, 100초 |

Dashboard에서는 `cluster=demo-b300`, `node=All`, `run_id=live-demo`를 선택합니다.

| 관찰 구간 | 기대 변화 |
| --- | --- |
| 정상 학습 | GPU utilization과 학습 지표 갱신 |
| `data_wait` | storage read·busy time 상승 |
| `collective` | communication timer·RoCE traffic 상승 |
| `checkpoint` | storage write 상승 |

GPU 내부 NVLINK와 node-to-fabric RoCE 관계는 topology matrix에서 확인합니다.
Topology는 전달된 연결 관계이며 link bandwidth나 endpoint별 traffic 측정값은 아닙니다.

### Run Overview Example

아래 GIF는 30초 동안 exporter 상태, sample age, GPU utilization matrix가 갱신되는 모습을 보여 줍니다.
Synthetic demo이므로 실제 LLM 학습 결과로 해석하지 않습니다.

![30초 Run Overview synthetic live demo](../figures/post-training-run-overview-30s.gif)

해석할 때 주의할 점:

- unavailable/null인 NVML 값을 사용량 0으로 읽지 않습니다.
- system memory와 학습 process의 CUDA allocated/reserved peak를 구분합니다.
- GPU utilization matrix의 index는 sampler가 보고한 device index이며 allocation이 아닙니다.
- interface·RDMA counter는 endpoint별 traffic matrix가 아닙니다.
- device·filesystem 지표는 특정 run의 단독 사용량이 아닙니다.

Simulator는 Prometheus의 `observability`와 `storage-smart` job에 metric을 제공합니다.
fixture는 `examples/live-demo/`에 있으며 `DEMO_TOPOLOGY_DIR`, `DEMO_ADDR`, `DEMO_PORT`로 경로와 listen address를 바꿀 수 있습니다.

## SSD Health

SSD health는 특정 run의 write 양이 아니라 장치 이상과 장기 열화를 확인하는 선택 기능입니다.

### Prerequisites and Permissions

| 항목 | 확인할 내용 |
| --- | --- |
| Exporter | `smartctl_exporter`는 observability 도구 설치에 포함 |
| System package | `smartctl`을 제공하는 `smartmontools`는 별도 설치 |
| NVMe 권한 | `/dev/nvmeN`의 admin-passthrough ioctl 접근 필요 |
| 자동 권한 처리 | 필요하면 exporter가 아닌 `smartctl`만 passwordless sudo로 실행 |
| 권한 강제 설정 | `SMARTCTL_SUDO=1` 또는 `SMARTCTL_SUDO=0` |

sudo를 사용할 때는 시작 전에 `sudo -n smartctl --scan`을 실행하며 실패하면 collector를 시작하지 않습니다.
이 검사가 성공해도 exporter log와 실제 SMART metric을 함께 확인합니다.

### Collect a Local SSD

compute node의 local SSD는 기존 `node` role에서 활성화합니다.

```bash
NODE_ADDR='<node-management-address>' \
ENABLE_SSD_HEALTH=1 \
  bash scripts/run_observability.sh node
```

### Collect Dedicated Storage Nodes

별도 storage node에서는 GPU sampler 없이 `storage` role만 실행합니다.
helper script를 사용할 수 없는 host에서는 `smartctl_exporter`를 따로 설치하고 실행 파일 경로를 지정합니다.

```bash
NODE_ADDR='<storage-node-management-address>' \
SMARTCTL_EXPORTER='<smartctl-exporter-path>' \
  bash scripts/run_observability.sh storage
```

| 설정 | 기본값 |
| --- | --- |
| Exporter port | `SMARTCTL_PORT=19633` |
| SMART 조회 주기 | `SMARTCTL_INTERVAL=60s` |
| Exporter 경로 | `SMARTCTL_EXPORTER`로 재정의 |
| smartctl 경로 | `SMARTCTL`로 재정의 |

짧은 scrape 주기를 사용해도 SSD firmware가 SMART 값을 같은 주기로 갱신한다는 보장은 없습니다.

### Add Storage Targets

Monitoring server에는 compute target과 storage target을 함께 전달합니다.

```bash
CLUSTER_NAME='<cluster-name>' \
OBSERVABILITY_TARGETS='trainer-0=<trainer-address>,trainer-1=<trainer-address>' \
STORAGE_SYSTEM='3fs' \
STORAGE_TARGETS='storage-0=<storage-address>,storage-1=<storage-address>' \
  bash scripts/run_observability.sh server
```

`STORAGE_SYSTEM`에는 `local`, `3fs`, `pnfs`처럼 배치를 식별하는 값을 사용합니다.
Data & Storage dashboard는 `cluster → storage system → storage node → SSD` 순서로 필터링합니다.

### Read SSD Metrics

| Metric | 의미 |
| --- | --- |
| Critical warning | 0이면 정상 metric, N/A이면 metric 없음 |
| Temperature | 현재 장치 온도 |
| Percentage used | vendor가 추정한 endurance 사용률 |
| Available spare | 남은 spare 비율 |
| Media errors | 복구되지 않은 media error 누계 |
| Lifetime host bytes written | host가 controller에 기록한 누계 |

`smartctl_device_bytes_written`에는 garbage collection과 wear leveling의 내부 NAND write가 포함되지 않습니다.
따라서 SSD write amplification으로 해석하지 않습니다.
`percentage_used`는 짧은 실행에서 변하지 않을 수 있으므로 장기 추세에 사용합니다.

분산 filesystem에서는 replication과 data-server layout 때문에 client write와 개별 SSD write가 일대일로 대응하지 않습니다.
SSD metric을 특정 run이나 client에 귀속하지 않습니다.
Storage topology component·edge와 SMART 시계열의 `instance` label에는 같은 storage node 이름을 사용합니다.

Docker Compose 예시는 `targets/storage.json`을 읽습니다.

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

dashboard나 collector 구성을 변경한 뒤에는 해당 `server` 또는 `node` role을 다시 시작합니다.
