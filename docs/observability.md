# Observability

`observability/`의 host·GPU 계측, Spark monitoring, 통신·저장소 baseline과 PyTorch trace 사용법을 다룹니다.
상시 지표로 이상 구간을 찾은 뒤 필요한 rank만 짧게 trace하며, synthetic 결과와 실제 LLM 학습 결과는 구분합니다.
지표 이름·단위·범위는 [Metrics Contract](#metrics-contract)를 따릅니다.

## CPU Checks

저장소 루트에서 실행합니다.
`setup.sh`는 `.venv`를 만들고 pytest를 설치하지만 CUDA PyTorch, NCCL Tests나 Python package 자체를 설치하지 않습니다.

```bash
cd observability
bash scripts/setup.sh
. .venv/bin/activate
bash scripts/check_tools.sh
python -m pytest -q ../tests/observability
```

미설치 도구 표시는 해당 기능만 준비되지 않았다는 뜻입니다.

## Spark Monitoring

`scripts/install_spark_tools.sh`가 ARM64 userspace 도구를 다운로드합니다(`TOOLS_DIR`로 경로 지정 가능, driver나 system package 설치는 대신하지 않음).

준비된 각 노드의 `observability`에서 node role을 실행합니다.
지정 주소의 19100 포트에 node exporter를 열고 GPU sampler를 기본 900초 동안 실행합니다.

```bash
NODE_ADDR='<node-management-address>' \
  bash scripts/run_spark_observability.sh node
```

ARM64 monitoring host에서는 server role을 별도 실행합니다.
`SPARK_TARGETS`는 `이름=주소` 항목을 쉼표로 연결하므로 노드 수를 고정하지 않습니다.

```bash
CLUSTER_NAME='<cluster-name>' \
SPARK_TARGETS='trainer-0=<first-node-address>,rollout-0=<second-node-address>' \
  bash scripts/run_spark_observability.sh server
```

Prometheus는 loopback 19090, Grafana는 loopback 13000에서 실행하며 작업 후 server 세션을 종료합니다.
Grafana는 anonymous Viewer이고 관리자 계정이 필요할 때만 `GRAFANA_ADMIN_PASSWORD`를 지정합니다.
`SERVER_CONFIG_ONLY=1`은 service 없이 provisioning 파일만 생성합니다.

Grafana를 보려면 client에서 controller를 jump host로 거쳐 server 노드(아래는 spark1)에 SSH forwarding을 엽니다.

```bash
ssh -NT -L 13000:127.0.0.1:13000 -J <user>@<controller-ssh-alias> spark@spark1
```

브라우저에서 `http://localhost:13000`을 엽니다.

`server`는 `examples/observability/`의 세 dashboard를 복사합니다.

- Run Overview(`spark-resources.json`, uid `spark-profiling`): 실행 진행·GPU 행렬
- Compute & Communication: GPU health·rank timer·인터페이스 통신
- Data & Storage: node-local device·filesystem 성능과 선택적 storage-node SSD health

화면 링크는 시간·cluster·node·run 선택을 유지합니다.
같은 경로의 `grafana/`와 `compose.yaml`은 별도 Docker Compose 예시이며 이 ARM64 경로에서는 쓰지 않습니다.

### Synthetic Live Demo

실제 GPU와 스토리지를 사용하지 않고 기존 Grafana dashboard에서 발표용 변화를 재생하려면 server role에 `DEMO_LIVE=1`을 지정합니다.
`demo-b300` cluster를 선택하면 GPU node 4개와 각 B300 GPU 8개, storage node 8개와 각 SSD 4개가 보입니다.
각 node는 800Gbps RoCE를 넘지 않는 synthetic traffic을 내보내며, GPU 내부 NVLINK와 node-to-fabric RoCE 연결은 기존 topology matrix에서 확인합니다.

```bash
cd observability
DEMO_LIVE=1 bash scripts/run_spark_observability.sh server
```

controller에서 ARM64 server host로 실행을 넘기려면 hostname을 지정합니다.

```bash
DEMO_LIVE=1 DEMO_SERVER_HOST=spark1 bash scripts/run_spark_observability.sh server
```

Grafana는 이 명령을 실행한 host의 loopback에 열리므로, SSH forwarding의 최종 host도 그 host여야 합니다.
예를 들어 controller에서 실행했다면 위의 `spark1` forwarding 예시 대신 controller로 연결합니다.

### Dashboard 구성

세 dashboard는 `cluster`, `node`, `run_id` filter와 시간 범위를 공유하며 화면 링크로 선택값을 넘깁니다.
Live demo에서는 `cluster=demo-b300`, `node=All`, `run_id=live-demo`를 선택합니다.

| Dashboard | 주요 panel | Live demo에서 볼 구간 |
| --- | --- | --- |
| Run Overview | target 상태·sample age, GPU utilization matrix, rank별 throughput·step time·loss | 시작 화면과 정상 학습 대비 |
| Compute & Communication | GPU health·memory, rank timer, RoCE interface throughput, GPU allocation·compute topology | `collective`에서 communication timer와 RoCE traffic 상승 |
| Data & Storage | node disk throughput·IOPS·busy time·filesystem, storage topology, SSD SMART | `data_wait`의 read/busy time과 `checkpoint`의 write 증가 |

Data & Storage는 storage system·storage node·SSD filter를 추가로 제공하며, 모든 storage node를 보려면 기본 `All`을 유지합니다.
Topology panel은 전달된 연결 관계를 표시하며, 직접 측정한 link bandwidth나 endpoint 쌍별 traffic matrix는 아닙니다.

Simulator는 100초마다 정상 학습, data wait, collective 통신, checkpoint, recovery를 반복합니다.
기존 dashboard JSON은 변경하지 않으며, Prometheus가 `spark`와 `storage-smart` job으로 scrape하는 metric만 합성합니다.
`examples/live-demo/gpu_topology.yaml`과 `storage_topology.yaml`은 JSON-compatible YAML이라 추가 Python package 없이 읽습니다.
`DEMO_TOPOLOGY_DIR`, `DEMO_ADDR`, `DEMO_PORT`로 fixture와 listen address를 바꿀 수 있습니다.

해석할 때 주의할 것:

- 일부 GB10 NVML 값은 unavailable/null이며 이를 사용량 0으로 읽지 않습니다.
- 시스템 메모리와 학습 process의 CUDA allocated/reserved peak를 구분합니다.
- GPU 행렬은 sampler가 보고한 index와 utilization이며 allocation이 아닙니다.
- 통신 패널은 interface·RDMA port counter이고 endpoint 쌍별 traffic matrix가 아닙니다.
- Data & Storage의 device·filesystem 지표를 특정 run의 단독 사용량으로 읽지 않습니다.

### SSD Health

SSD health는 실험별 write attribution이 아니라 장치 이상과 장기 열화를 확인하는 선택 기능입니다.
`smartctl_exporter`는 기본 설치 스크립트에 포함되지만 `smartctl`은 운영체제의 `smartmontools` package로 설치해야 합니다.

Spark 검증 환경에서 NVMe SMART 조회는 controller device(`/dev/nvmeN`)에 admin-passthrough ioctl을 열며 이 장치는 sibling block device(`/dev/nvmeXn1`, disk group 읽기 가능)와 달리 group 권한이 없는 root:root 0600으로 남습니다.
`node`/`storage` role은 이를 자동 감지해 필요하면 passwordless sudo로 `smartctl`만 감싸 실행합니다(`smartctl_exporter` 자체는 root로 올리지 않습니다).
sudo 경로를 선택하면 exporter 시작 전에 `sudo -n smartctl --scan`으로 비대화식 실행 권한을 확인하고 실패 시 중단합니다.
이 검사는 각 장치의 SMART 조회 성공을 보장하지 않으므로 exporter log와 실제 SMART metric도 확인합니다.
자동 감지를 강제로 켜거나 끄려면 `SMARTCTL_SUDO=1` 또는 `SMARTCTL_SUDO=0`을 지정합니다.
sudo 없이 이미 권한이 있는 환경(예: 별도 udev rule이나 capability 설정)에서는 자동 감지가 sudo를 건너뜁니다.

실제 SSD가 장착된 노드에서 health 수집을 활성화합니다.
로컬 SSD라면 기존 Spark node role에 함께 실행합니다.

```bash
NODE_ADDR='<spark-node-management-address>' \
ENABLE_SSD_HEALTH=1 \
  bash scripts/run_spark_observability.sh node
```

3FS·pNFS처럼 storage node가 분리된 구성에서는 GPU sampler를 실행하지 않는 storage role을 각 storage node에서 사용합니다.
ARM64 Spark 설치 스크립트가 맞지 않는 storage node에서는 `smartctl_exporter`를 따로 설치하고 `SMARTCTL_EXPORTER`에 실행 파일 경로를 지정합니다.

```bash
NODE_ADDR='<storage-node-management-address>' \
SMARTCTL_EXPORTER='<smartctl-exporter-path>' \
  bash scripts/run_spark_observability.sh storage
```

기본 exporter port는 19633이고 SMART 조회 주기는 60초입니다.
필요하면 `SMARTCTL_PORT`, `SMARTCTL_INTERVAL`, `SMARTCTL_EXPORTER`, `SMARTCTL`을 지정합니다.
짧은 scrape 주기를 사용해도 SSD firmware가 내부 SMART 값을 같은 주기로 갱신한다는 보장은 없습니다.

Monitoring server에는 compute node와 별도로 storage target을 전달합니다.

```bash
CLUSTER_NAME='<cluster-name>' \
SPARK_TARGETS='trainer-0=<trainer-address>,trainer-1=<trainer-address>' \
STORAGE_SYSTEM='3fs' \
STORAGE_TARGETS='storage-0=<storage-address>,storage-1=<storage-address>' \
  bash scripts/run_spark_observability.sh server
```

`STORAGE_SYSTEM`은 `local`, `3fs`, `pnfs`처럼 배치를 식별하는 값입니다.
Data & Storage dashboard는 `cluster → storage system → storage node → SSD` 순서로 필터링합니다.
3FS의 replication과 pNFS의 data-server layout 때문에 client write와 개별 SSD write는 일대일로 대응하지 않으므로 health metric을 특정 run이나 client에 귀속하지 않습니다.
Storage topology component·edge는 기존 `storage-topology.json`으로 공급하고, SMART 시계열의 `instance` label(화면의 `storage_node` 변수)과 topology component ID는 같은 이름을 사용합니다.

Critical-warning 개수는 정상 metric이 있으면 0, metric 자체가 없으면 N/A로 표시합니다.
Dashboard는 다음 값을 표시합니다.

- NVMe critical warning과 SMART collection status
- 현재 온도
- vendor가 추정한 endurance percentage used
- available spare
- unrecovered media error 누계
- 장치가 보고한 lifetime host bytes written

`smartctl_device_bytes_written`은 host가 controller에 기록한 누계입니다.
Garbage collection과 wear leveling에서 발생한 내부 NAND write는 포함하지 않으며 SSD write amplification으로 해석하지 않습니다.
`percentage_used`는 짧은 실험에서 변하지 않을 수 있으므로 장기 추세에 사용합니다.

Docker Compose 예시는 `targets/storage.json`을 읽습니다.
기본 파일은 빈 목록이며 exporter를 배치한 뒤 다음처럼 target과 topology label을 추가합니다.

```json
[
  {
    "targets": ["storage-0.example:19633"],
    "labels": {
      "cluster": "spark-cluster",
      "storage_system": "pnfs",
      "instance": "storage-0"
    }
  }
]
```

## Live Framework Metrics

`node` role에 launcher와 같은 `FRAMEWORK_METRICS_DIR`를 지정하면 [`framework_metrics_textfile`](../observability/profiling_lab/framework_metrics_textfile.py)이 rank JSON을 주기적으로 읽습니다.
`training_loss`, `training_tokens_per_second`, `training_step_time_seconds`, `training_step`과 Megatron의 `training_timer_seconds{timer="..."}`를 GPU sampler와 같은 textfile collector에 씁니다.

```bash
NODE_ADDR='<node-management-address>' \
FRAMEWORK_METRICS_DIR='<launcher output-directory>/framework-metrics' \
  bash scripts/run_spark_observability.sh node
```

생략하면 framework 수집만 꺼지고 host/GPU 모니터링은 계속됩니다.
**각 노드의 node-local 경로를 사용합니다.**
두 노드가 같은 NFS 파일을 읽으면 같은 rank가 두 `instance`로 중복 노출됩니다.

GPU allocation matrix는 framework snapshot에 `CUDA_VISIBLE_DEVICES`와 `LOCAL_RANK`가 있는 rank만 표시합니다.
행은 cluster·node·run·framework·rank로 구분하며, 열의 device ID는 보고된 index 또는 UUID 그대로입니다.
GPU 사용률 Matrix의 index와 UUID를 자동 매칭하지 않습니다.
GPU 지표는 30초, 학습 지표·할당은 `Training sample max age (s)`(기본 300초)를 넘으면 숨깁니다.
긴 step에서는 이 값을 늘리며, 오래되거나 없는 할당 정보를 GPU가 비어 있다는 뜻으로 해석하지 않습니다.
Overview의 rank별 sample age는 오래된 값도 표시하므로 일부 rank의 갱신 중단을 확인할 수 있습니다.
`TOPOLOGY_DIR`를 한 node role에 지정하면 `compute-topology.json`·`storage-topology.json`의 `components`(각 `id`와 선택적 `role`)와 `edges`(`source`, `destination`, 선택적 `relation`)를 Grafana에 표시합니다 — 전달된 관계를 보여줄 뿐 bandwidth·latency 측정값이 아닙니다.
Topology는 Cluster 필터만 적용하며 Node·Run 선택과 독립적입니다.

저장소 갱신 후 monitoring host의 `server` role을 다시 실행해야 새 dashboard JSON이 provisioning 경로에 복사됩니다.
수집기 변경은 각 노드의 `node` role 재시작으로 적용합니다.

## Run History

[`show_run`](../observability/profiling_lab/show_run.py)은 서버 없이 output-dir의 Megatron `run-metadata-<stage>.json`과 TRL `summary-<stage>.json`을 읽습니다.
`OBSERVATORY_RUN_ID`/`FRAMEWORK_METRICS_DIR`가 설정됐다면 `framework-metrics/<framework>-rank-<rank>.json`의 마지막 step도 표시합니다.

```bash
PYTHONPATH=observability python3 -m profiling_lab.show_run '<output-dir>'
```

output-dir가 controller에서 보이지 않는 node-local 경로면 해당 Spark 노드에서 실행합니다.
`framework-metrics`는 학습 중에는 최신 step 값이고 끝나면 마지막 값에 고정되므로, 전체 loss 추이가 필요하면 `logs/`의 학습 로그를 확인합니다.

## Distributed Trace

각 노드의 `observability`에서 같은 `PROFILE_RUN_ID`를 지정하고 첫 노드는 `NODE_RANK=0`, 다른 노드는 `1`로 실행합니다.
이 명령은 GPU 작업을 시작하며 preflight가 기존 GPU 작업과 최소 가용 메모리를 검사합니다 — 다른 작업이 있으면 임의 종료하지 않고 끝난 후 실행합니다.

```bash
PROFILE_RUN_ID=profile-001 \
NODE_RANK=0 \
MASTER_ADDR='<first-node-data-address>' \
PYTHON='<cuda-python>' \
  bash scripts/run_spark_profile.sh baseline
```

모드만 `capture`로 바꾸면 rank 0·1 trace를 수집합니다.
기본 24 step, 실행 제한 300초이며 `STEPS`·`RUN_TIMEOUT`으로 조정하고, 출력은 기본 `artifacts/spark/<run-id>/<mode>/`입니다.
양 rank 로그·manifest·JSON 결과와 capture의 `traces/`를 확인합니다.
Baseline과 capture를 동시에 실행하지 말고 profiler overhead를 비교합니다.
이 예제는 synthetic DDP이며 실제 LLM trace가 아닙니다.

기존 PyTorch loop에 직접 삽입하는 selected-rank profiler 사용법은 [Framework Integration](#framework-integration)에 있습니다.

## Hardware Baselines

NCCL baseline은 MPI 지원 `all_reduce_perf`, `mpirun`과 할당받은 GPU 노드를 요구하며 모든 지정 노드에 GPU 부하를 발생시킵니다.

```bash
NCCL_TEST_BINARY='<all-reduce-perf-path>' \
HOSTS='<first-host>,<second-host>' GPUS_PER_NODE=1 \
  bash scripts/run_nccl_baseline.sh
```

`artifacts/nccl-baseline/manifest.env`와 `all-reduce.log`에서 조건·correctness 오류·대역폭을 확인합니다.
NCCL baseline은 학습 throughput이 아닙니다.

<a id="metrics-contract"></a>

## Metrics Contract

비교 가능한 지표의 **이름·단위·측정 범위**는 [`config/metrics.json`](../observability/config/metrics.json)을 기준으로 합니다.
파일 형식은 [`config/metrics.schema.json`](../observability/config/metrics.schema.json)에 정의하며 `observability/profiling_lab/schema.py`가 추가 의존성 없이 runtime validation을 수행합니다.

`metrics.json`의 `schema_version`은 소비자가 이해하는 계약 version입니다.
기존 metric의 의미나 단위를 바꾸는 호환성 파괴 변경에만 version을 올리고, 새 metric 추가는 같은 version에서 합니다.

### Metric Entry

각 metric은 여섯 field를 가집니다.

```json
{
  "name": "data_movement_effective_bandwidth_bytes_per_second",
  "category": "data_movement",
  "unit": "bytes/s",
  "scope": "phase and path",
  "source": "derived from bytes and duration",
  "policy": "phase profiling"
}
```

| Field | 의미 |
| --- | --- |
| `name` | Prometheus와 summary에서 쓰는 stable snake_case 이름 |
| `category` | training, GPU, host, container, network, data movement, storage, checkpoint 영역 |
| `unit` | 저장 단위. dashboard에서 GiB·Gbps로 변환하기 전의 canonical unit |
| `scope` | run, node, GPU, rank, phase, device, mount, parallel group 등 값이 속한 범위 |
| `source` | exporter, framework timer, selected trace, manifest 또는 derived summary |
| `policy` | always-on, workload-specific, baseline, diagnostic 수집 조건 |

**계약은 목표 vocabulary이며 자동 수집 목록이 아닙니다.**
Dashboard는 exporter 원본 이름을 조회하고 학습 지표는 rank JSON을 textfile collector로 재발행합니다([Live Framework Metrics](#live-framework-metrics)).
Canonical name 변환·phase 집계는 workload adapter에 추가 구현해야 합니다.
Derived metric은 원본을 보존하고 계산 window·source metric을 summary에 기록합니다.

### Labels and Manifest Fields

`recommended_labels`는 필터·비교 기준이며 시계열 증가를 제한하도록 값의 종류를 제한합니다.
공통 후보는 `run_id`, `cluster`, `job`, `node`, `gpu`, `framework`, `role`, `phase`, `device`, `interface`, `operation`, `parallel_group`이고, Megatron hook의 `rank`·`local_rank`·`tp_rank`·`pp_rank`·`dp_rank`·`timer`는 범위를 제한한 예제 확장입니다.

Commit, image digest, model/dataset/checkpoint URI, rank map, profiler option, precision, batch/sequence 설정, storage path type, filesystem, cache state, node topology는 `manifest_only_fields`에 기록합니다.
**Prompt, request ID, timestamp, trace ID처럼 계속 늘어나는 값은 Prometheus label로 쓰지 않습니다.**

### Workflow Phases

`phase_vocabulary`는 framework가 달라도 같은 lifecycle 구간을 비교하기 위한 이름입니다.

| Phase | 주 신호 |
| --- | --- |
| `dataset_loading` | storage read, metadata operation, preprocessing, data wait |
| `model_loading` | checkpoint read, host staging, host-to-GPU copy |
| `training_input` | pinned memory, host-to-GPU bytes/time, compute overlap |
| `forward_backward` | GPU compute, activation/gradient movement, collective |
| `optimizer_step` | AllReduce, ReduceScatter, AllGather, rank synchronization |
| `checkpoint_save` | training pause, GPU-to-host staging, write bandwidth와 volume |
| `checkpoint_restore` | read bandwidth, host-to-GPU restore, rank synchronization |
| `evaluation` | inference compute, input transfer, idle time |

Phase marker에는 최소한 `run_id`, `phase`, 시작/종료 시각과 성공 여부를 기록합니다.
Bytes와 duration을 모두 얻으면 유효 대역폭을 계산하고 같은 path의 NCCL Tests baseline과 비교해 utilization ratio를 만듭니다.

### Data Movement Paths

| Path | Always-on 증거 | Diagnostic 증거 |
| --- | --- | --- |
| 저장소 → host memory | Node Exporter disk·filesystem·mountstats | iostat 또는 selected I/O trace |
| Host memory → GPU | framework data wait, pinned memory | PyTorch Profiler memory copy event, Nsight |
| 노드 내 GPU ↔ GPU | DCGM utilization, framework collective timer | NCCL Tests single-node baseline |
| 노드 ↔ 노드 | NIC/InfiniBand counter, communication timer | NCCL Tests multi-node baseline, GPUDirect RDMA |
| GPU/host → checkpoint storage | checkpoint timer, storage throughput·volume | writer/rank coordination trace |

Node Exporter와 DCGM만으로는 bytes가 어떤 phase나 rank에서 발생했는지 알 수 없습니다.
Framework phase marker와 rank map을 같은 `run_id`로 연결하고, 원인이 남을 때만 selected trace를 수집합니다.

**RoCE는 TCP/IP와 RDMA counter를 함께 봅니다.**
소켓 트래픽은 `node_network_*`, kernel을 우회하는 RDMA verbs(NCCL IB transport 포함)는 `/sys/class/infiniband`에서 집계됩니다.
기본 활성화된 Node Exporter `infiniband` collector는 `node_infiniband_port_data_{received,transmitted}_bytes_total`을 노출합니다.
계약의 `rdma_receive_bytes_per_second`·`rdma_transmit_bytes_per_second`·`rdma_errors_total`과 dashboard의 RDMA 패널을 확인하며 `network_*`만으로 판단하지 않습니다.

### Validate the Contract

```bash
python -m pytest -q tests/observability/test_schema.py
```

JSON 문법, 허용된 이름·분류, 필수 field와 중복 이름을 확인합니다.
새 metric을 추가할 때는 exporter나 framework에서 실제로 얻을 수 있는 source를 먼저 확인하고, canonical unit과 scope를 정한 뒤 `metrics.json`·adapter·summary·dashboard를 같은 변경에서 갱신합니다.

### Framework Integration

Runner는 output 이름을 `OBSERVATORY_RUN_ID`로 설정하고 TRL·Megatron callback은 `<output>/framework-metrics/`의 rank JSON을 atomic replace합니다.
실시간 조회는 [textfile collector](#live-framework-metrics), 과거 조회는 [`show_run`](#run-history)이 읽습니다.

**TRL tokens/s는 Trainer의 누적 입력 token 차이이고, Megatron tokens/s는 `global_batch_size * max_length`를 callback wall time으로 나눈 configured-token 처리율입니다** — variable-length 실행의 실제 non-padding 처리율로 해석하지 않습니다.
Megatron timer는 `timing_log_level=1`에서 이미 계산된 timer의 rank-local `active_time` 차이를 읽으며 adapter 때문에 추가 collective를 실행하지 않습니다.

[Selected-rank helper](../observability/examples/pytorch/selected_rank_profiler.py)는 선택하지 않은 rank에 no-op profiler를 돌려줍니다.
아래는 완성된 실행 명령이 아니라 기존 PyTorch loop에 삽입하는 예시이며, `observability`가 import 경로에 있어야 합니다.

```python
from pathlib import Path
from examples.pytorch.selected_rank_profiler import selected_rank_profile

with selected_rank_profile(
    Path("artifacts/traces/run-001"),
    ranks={0, 1},
    skip_first=4,
    wait=1,
    warmup=1,
    active=2,
) as profiler:
    for batch in train_loader:
        train_step(batch)
        profiler.step()
```

모든 iteration에서 `profiler.step()`을 호출해야 schedule이 진행됩니다.
기본적으로 shape·memory·stack 수집은 꺼져 있으며 필요한 질문에 한해 켭니다.
출력은 `rank-<rank>/trace-<index>.json`이고 비교할 rank는 같은 run·capture 구간이어야 합니다.
CPU의 kernel 제출 지연, NCCL과 compute의 겹침, rank별 collective 도착 시점, copy·동기화 집중 구간을 확인한 뒤, 원인을 수정하면 profiler를 끈 실행에서 효과를 다시 검증합니다.

[verl profiler 설정](../observability/examples/verl/torch-profiler.yaml)은 외부 framework 연동 참고이며 이 저장소에 verl backend가 있다는 뜻이 아닙니다.
