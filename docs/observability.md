# Observability

`observability/`는 host·GPU 계측, Spark monitoring, 통신·저장소 baseline과 PyTorch trace 예제를 제공합니다.
먼저 상시 지표로 이상 구간을 찾고 필요한 rank만 짧게 trace하는 순서를 씁니다.
Synthetic workload 결과와 실제 LLM 학습 결과는 구분합니다.

이 문서는 도구를 어떻게 실행하는지와 지표 이름·단위·수집 범위를 맞추는 계약([Metrics Contract](#metrics-contract))을 함께 관리합니다.

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

도구 검사에서 미설치 도구가 표시되면 해당 기능의 준비가 안 된 것이며 전체 기능 실패로 해석하지 않습니다.

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

Prometheus는 loopback 19090, Grafana는 loopback 13000에 바인딩합니다.
Grafana는 anonymous Viewer로 열리므로 SSH tunnel만 있으면 비밀번호 없이 볼 수 있고, 관리자 계정이 필요할 때만 `GRAFANA_ADMIN_PASSWORD`를 지정합니다.
Server는 종료할 때까지 실행하므로 작업을 마치면 세션을 종료합니다.
`SERVER_CONFIG_ONLY=1`은 service를 시작하지 않고 생성된 provisioning 파일만 써서 target 구성을 확인합니다.

Grafana가 loopback에만 열리므로 브라우저로 보려면 SSH local forwarding이 필요합니다.
Controller에는 GUI가 없고 사용자의 client 머신에서 접속하는 host이므로, client → controller → spark1 순으로 두 번 거치며 controller는 순수 jump host로 씁니다.

```bash
ssh -NT -L 13000:127.0.0.1:13000 -J <user>@<controller-ssh-alias> spark@spark1
```

브라우저에서 `http://localhost:13000`을 엽니다.

`server` role은 `examples/observability/`의 세 dashboard를 함께 복사합니다 — Run Overview(`spark-resources.json`, uid `spark-profiling`)는 실행 진행과 GPU 행렬을, Compute & Communication은 GPU health·rank timer·인터페이스별 통신을, Data & Storage는 node-local device·filesystem을 다룹니다.
화면 링크는 시간 범위와 cluster·node·run 선택을 전달합니다.
저장소의 `examples/observability/grafana/`와 `compose.yaml`은 Docker Compose로 Prometheus·Grafana를 띄우는 별도의 일반 예시이며 이 ARM64 워크플로우에서는 쓰지 않습니다.

해석할 때 주의할 것:

- 일부 GB10 NVML 값은 unavailable/null이며 이를 사용량 0으로 읽지 않습니다.
- 시스템 메모리와 학습 process의 CUDA allocated/reserved peak를 구분합니다.
- GPU 행렬은 sampler가 보고한 index와 utilization이며 allocation이 아닙니다.
- 통신 패널은 interface·RDMA port counter이고 endpoint 쌍별 traffic matrix가 아닙니다.
- Data & Storage의 device·filesystem 지표를 특정 run의 단독 사용량으로 읽지 않습니다.

## Live Framework Metrics

`node` role을 실행할 때 같은 shell에 `FRAMEWORK_METRICS_DIR`를 학습 launcher가 쓰는 값과 똑같이 지정하면, 그 노드가 [`framework_metrics_textfile`](../observability/profiling_lab/framework_metrics_textfile.py)을 추가로 띄웁니다.
이 process는 `<framework>-rank-<rank>.json` 스냅샷을 주기적으로 읽어 `training_loss`, `training_tokens_per_second`, `training_step_time_seconds`, `training_step`, (Megatron이면) `training_timer_seconds{timer="..."}`를 GPU sampler와 같은 textfile collector에 씁니다.

```bash
NODE_ADDR='<node-management-address>' \
FRAMEWORK_METRICS_DIR='<launcher output-directory>/framework-metrics' \
  bash scripts/run_spark_observability.sh node
```

새 서비스가 아니라 launcher가 이미 쓰는 node-local 경로를 가리키기만 하면 되므로, 별도 NFS 공유나 collector 없이 그 노드의 rank가 같은 Grafana 대시보드에 나타납니다.
지정하지 않으면 이 process만 시작되지 않고 나머지 host/GPU 모니터링은 그대로 동작합니다.

> **`FRAMEWORK_METRICS_DIR`를 NFS 공유 경로로 두 노드 모두에 지정하지 않습니다.**
> 두 노드가 같은 파일을 각자 읽어 같은 rank가 두 `instance` label로 중복 노출됩니다.
> 각 노드가 자신의 node-local 경로(기본값)만 읽게 둡니다.

GPU allocation matrix는 framework snapshot에 `CUDA_VISIBLE_DEVICES`와 `LOCAL_RANK`가 있는 rank만 표시합니다.
`TOPOLOGY_DIR`를 한 node role에 지정하면 `compute-topology.json`·`storage-topology.json`의 `components`(각 `id`와 선택적 `role`)와 `edges`(`source`, `destination`, 선택적 `relation`)를 Grafana에 표시합니다 — 전달된 관계를 보여줄 뿐 bandwidth·latency 측정값이 아닙니다.

## Run History

과거 run 조회에는 서버가 필요 없습니다.
학습 launcher가 이미 output-dir에 파일을 남깁니다: Megatron은 `run-metadata-<stage>.json`, TRL은 `summary-<stage>.json`, 그리고 `OBSERVATORY_RUN_ID`/`FRAMEWORK_METRICS_DIR`가 설정됐다면 `framework-metrics/<framework>-rank-<rank>.json`(마지막 step 값)도 남습니다.
[`show_run`](../observability/profiling_lab/show_run.py)은 이 파일들을 읽어 요약을 출력하는 CLI입니다.

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

같은 지표를 다른 이름이나 단위로 기록하면 실행 결과를 비교할 수 없습니다.
이 절은 학습 코드와 자원 수집 도구가 지표의 **이름·단위·측정 범위**를 맞추는 규칙을 설명합니다.
실제 source of truth는 [`config/metrics.json`](../observability/config/metrics.json)이고 [`config/metrics.schema.json`](../observability/config/metrics.schema.json)이 파일 형식을 검증합니다.
`observability/profiling_lab/schema.py`는 별도 dependency 없이 test와 script에서 runtime validation을 수행합니다.

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

**이 계약은 목표 vocabulary이지 자동 수집 목록이 아닙니다.**
현재 dashboard는 exporter가 노출하는 원본 이름을 그대로 조회합니다.
host/GPU/network는 Node Exporter·DCGM이 직접 exporter이고, 학습 지표는 TRL·Megatron 콜백이 rank별로 쓴 JSON 스냅샷을 `framework_metrics_textfile`이 같은 노드의 textfile collector로 재발행한 값입니다 — push gateway가 아니라 GPU sampler와 같은 패턴입니다([Live Framework Metrics](#live-framework-metrics)).
Canonical name 변환과 phase별 집계는 workload adapter에서 추가 구현해야 합니다.
Derived metric은 원본 값을 덮어쓰지 않고 계산에 쓴 window와 source metric을 summary에 함께 기록합니다.

### Labels and Manifest Fields

`recommended_labels`는 결과를 필터링·비교하는 분류 기준입니다.
값의 종류가 제한된 항목만 써야 시계열 수가 폭증하지 않습니다.
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

**RoCE NIC는 같은 물리 포트에서 두 경로를 따로 셉니다.**
일반 TCP/IP 소켓 트래픽은 kernel netdev 경로를 지나 `node_network_*`로 잡히고, RDMA verbs 트래픽(NCCL의 IB transport 포함)은 kernel bypass 경로를 지나 `/sys/class/infiniband`의 counter로 잡힙니다.
Node Exporter의 `infiniband` collector는 기본으로 켜져 있어 이 두 번째 경로를 `node_infiniband_port_data_{received,transmitted}_bytes_total`로 노출하며, `metrics.json`은 이를 `rdma_receive_bytes_per_second`·`rdma_transmit_bytes_per_second`·`rdma_errors_total`로 정의합니다.
`network_*`만 보면 RDMA 트래픽이 잡히지 않으므로, 노드 간 collective가 실제로 RDMA를 쓰는지 확인하려면 `rdma_*`(또는 dashboard의 RDMA throughput 패널)를 함께 봐야 합니다.

### Validate the Contract

```bash
python -m pytest -q tests/observability/test_schema.py
```

JSON 문법, 허용된 이름·분류, 필수 field와 중복 이름을 확인합니다.
새 metric을 추가할 때는 exporter나 framework에서 실제로 얻을 수 있는 source를 먼저 확인하고, canonical unit과 scope를 정한 뒤 `metrics.json`·adapter·summary·dashboard를 같은 변경에서 갱신합니다.

### Framework Integration

공통 runner는 `OBSERVATORY_RUN_ID`를 실행 output 이름으로 설정합니다.
TRL callback과 Megatron Bridge callback은 collector와 통신하지 않고 rank별 최신 JSON을 `<output>/framework-metrics/`에 atomic replace합니다.
이 JSON은 상시 관측이면 같은 노드의 textfile collector가 재발행하고([Live Framework Metrics](#live-framework-metrics)), 과거 조회면 `show_run`이 직접 읽습니다([Run History](#run-history)).

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
