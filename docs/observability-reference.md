# Profiling Metric Contract

같은 지표를 서로 다른 이름이나 단위로 기록하면 실행 결과를 비교하기 어렵습니다.
이 문서는 학습 코드와 자원 수집 도구가 지표의 이름·단위·측정 범위(scope)를 맞추는 규칙을 설명합니다.
실제 source of truth는 [`config/metrics.json`](../observability/config/metrics.json)이며 [`config/metrics.schema.json`](../observability/config/metrics.schema.json)은 파일 형식을 검증하는 JSON Schema입니다.

## Contract Files

| File | Role |
| --- | --- |
| `config/metrics.json` | 수집·파생 metric, label, manifest field와 phase vocabulary의 canonical definition |
| `config/metrics.schema.json` | metric object의 필수 field, category와 naming rule을 정의하는 JSON Schema |
| `observability/profiling_lab/schema.py` | 별도 dependency 없이 repository test와 script에서 수행하는 runtime validation |

`metrics.json`의 `schema_version`은 소비자가 이해하는 계약 version입니다.
기존 metric의 의미나 단위를 바꾸는 호환성 파괴 변경이 있을 때만 version을 올리고, 새 metric을 추가하는 변경은 같은 version에서 수행합니다.

## Metric Entry

지표를 추가할 때는 무엇을 어디서 어떤 단위로 측정했는지 함께 정의합니다.
아래 예시는 전송량을 시간으로 나눈 유효 대역폭이며, 각 metric은 다음 여섯 field를 가집니다.

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

| Field | Meaning |
| --- | --- |
| `name` | Prometheus와 summary에서 사용하는 stable snake_case name |
| `category` | training, GPU, host, container, network, data movement, storage, checkpoint 또는 agentic RL 영역 |
| `unit` | 값을 해석하는 canonical unit. dashboard에서 GiB, Gbps 등으로 변환하기 전 저장 단위 |
| `scope` | run, node, GPU, rank, phase, device, mount, parallel group 등 값이 속한 범위 |
| `source` | exporter, framework timer, selected trace, manifest 또는 derived summary |
| `policy` | always-on, workload-specific, baseline 또는 diagnostic 수집 조건 |

이 contract는 목표 vocabulary이며 자동 수집 목록이 아닙니다.
현재 dashboard는 exporter가 노출하는 원본 이름을 그대로 조회합니다.
host/GPU/network 지표는 Node Exporter·DCGM이 직접 exporter이고, 학습 지표(`training_loss`, `training_tokens_per_second`, `training_step_time_seconds`)는 TRL·Megatron 콜백이 rank별로 쓴 JSON 스냅샷을 `profiling_lab.framework_metrics_textfile`이 같은 노드의 Node Exporter textfile collector로 재발행한 값입니다 — 별도 exporter나 push gateway가 아니라 GPU sampler(`spark_telemetry.py`)가 이미 쓰는 것과 같은 패턴이며, 자세한 구조는 [Local Viewing 절](observability.md#local-viewing)을 참고합니다.
Megatron의 rank-local timer(`forward-backward` 등)는 `training_timer_seconds{timer="..."}`로 노출됩니다.
Canonical name 변환과 phase별 bytes/time 집계는 workload adapter에서 추가 구현해야 합니다.
Derived metric은 원본 값을 덮어쓰지 않으며 계산에 사용한 window와 source metric을 summary에 함께 기록합니다.

## Labels and Manifest Fields

`recommended_labels`는 실행 결과를 필터링하고 비교할 때 사용하는 분류 기준입니다.
예를 들어 `node`로 특정 노드를 고르고 `phase`로 학습 단계만 볼 수 있습니다.
값의 종류가 제한된 항목을 사용해야 시계열 수가 과도하게 늘어나지 않습니다.
`run_id`, `cluster`, `job`, `node`, `gpu`, `framework`, `role`, `phase`, `device`, `interface`, `operation`, `parallel_group`을 공통 후보로 사용합니다.
Megatron hook의 `rank`, `local_rank`, `tp_rank`, `pp_rank`, `dp_rank`, `timer`는 allocation과 timer 목록으로 범위를 제한하는 예제 확장입니다.

Commit, image digest, model/dataset/checkpoint URI, complete rank map, profiler option, precision, batch/sequence configuration, storage path type, filesystem, cache state와 node topology는 `manifest_only_fields`에 기록합니다.
Prompt, request ID, timestamp와 trace ID처럼 계속 늘어나는 값은 Prometheus label로 사용하지 않습니다.

## Workflow Phases

`phase_vocabulary`는 framework가 달라도 같은 lifecycle 구간을 비교하기 위한 이름입니다.

| Phase | Main signal |
| --- | --- |
| `dataset_loading` | storage read, metadata operation, preprocessing과 data wait |
| `model_loading` | checkpoint read, host staging, host-to-GPU copy |
| `training_input` | pinned memory, host-to-GPU bytes/time과 compute overlap |
| `forward_backward` | GPU compute, activation/gradient movement와 collective |
| `optimizer_step` | AllReduce, ReduceScatter, AllGather와 rank synchronization |
| `checkpoint_save` | training pause, GPU-to-host staging, write bandwidth와 volume |
| `checkpoint_restore` | read bandwidth, host-to-GPU restore와 rank synchronization |
| `evaluation` | inference compute, input transfer와 idle time |
| `rollout`, `tool_interaction`, `reward`, `weight_sync` | agentic RL의 generation, external wait, reward/evaluation과 policy distribution |

Phase marker에는 최소한 `run_id`, `phase`, 시작/종료 시각과 성공 여부를 기록합니다.
Bytes와 duration을 모두 얻을 수 있으면 `data_movement_effective_bandwidth_bytes_per_second`를 계산하고, 동일 path의 NCCL Tests baseline과 비교해 utilization ratio를 만듭니다.

## Data Movement Paths

| Path | Always-on evidence | Diagnostic evidence |
| --- | --- | --- |
| Local/remote storage → host memory | Node Exporter disk, filesystem, mountstats와 storage client metric | iostat 또는 selected I/O trace |
| Host memory → GPU | framework data wait와 pinned memory | PyTorch Profiler memory copy event 또는 Nsight Systems |
| GPU ↔ GPU in one node | DCGM utilization과 framework collective timer | NCCL Tests single-node baseline, selected trace |
| GPU node ↔ GPU node | NIC/InfiniBand counter와 communication timer | NCCL Tests multi-node baseline, GPUDirect RDMA 확인 |
| GPU/host → checkpoint storage | checkpoint timer, storage throughput와 volume | writer/rank coordination trace |

Node Exporter와 DCGM만으로는 bytes가 어떤 framework phase나 rank에서 발생했는지 알 수 없습니다.
Framework phase marker와 rank map을 같은 `run_id`로 연결하고, 원인이 남을 때만 selected trace를 수집합니다.

RoCE NIC는 같은 물리 포트에서 두 경로를 분리해서 셉니다: 일반 TCP/IP 소켓 트래픽은 kernel netdev 경로를 지나 `node_network_*`(`network_receive_bytes_per_second`, `network_transmit_bytes_per_second`, `network_errors_total`)로 잡히고, RDMA verbs 트래픽(NCCL의 IB transport 포함)은 kernel bypass 경로를 지나 `/sys/class/infiniband`의 InfiniBand counter로 잡힙니다.
Node Exporter의 `infiniband` collector는 기본으로 켜져 있어 별도 flag 없이 이 두 번째 경로를 `node_infiniband_port_data_received_bytes_total`, `node_infiniband_port_data_transmitted_bytes_total`(및 오류 counter)로 노출하며, `metrics.json`은 이를 `rdma_receive_bytes_per_second`, `rdma_transmit_bytes_per_second`, `rdma_errors_total`로 정의합니다.
`network_*` metric만 보면 RDMA 트래픽이 잡히지 않으므로, GPU node 간 collective가 RDMA를 실제로 쓰는지 확인하려면 `rdma_*` metric(또는 dashboard의 "RDMA (InfiniBand/RoCE) throughput" panel)을 함께 봐야 합니다.

## Validate the Contract

관측 도구의 Python 환경을 활성화한 뒤 저장소 루트에서 지표 정의를 검사합니다.
다음 명령은 JSON 문법, 허용된 이름·분류, 필수 필드와 중복 지표 이름을 확인합니다.

```bash
python -m pytest -q tests/observability/test_schema.py
```

새 metric을 추가할 때는 exporter 또는 framework에서 실제로 얻을 수 있는 source를 먼저 확인하고, canonical unit과 scope를 결정한 뒤 `metrics.json`, adapter, summary와 dashboard를 같은 변경에서 갱신합니다.

## Framework Integration

공통 runner는 `OBSERVATORY_RUN_ID`를 실행 output 이름으로 설정합니다.
TRL callback과 Megatron Bridge callback은 collector와 통신하지 않고 rank별 최신 JSON을 `<output>/framework-metrics/`에 atomic replace합니다.
이 JSON을 실제로 어디서 읽는지는 목적에 따라 다릅니다 — host·GPU·network·RDMA까지 한 화면에서 상시로 보려면 같은 노드의 `profiling_lab.framework_metrics_textfile`이 이를 Node Exporter textfile collector로 재발행해 Grafana에서 바로 보이고([Local Viewing](observability.md#local-viewing)), Prometheus·Grafana 없이 CPU·메모리·NIC·학습 지표만 가볍게 보려면 `profiling_lab.node_agent`가 (loopback 또는 SSH reverse tunnel로) `profiling_lab.collector`에 전송합니다([Controller에서 결과 수집과 표시](observability.md#controller에서-결과-수집과-표시)).
TRL tokens/s는 Trainer의 누적 입력 token 차이이고 Megatron tokens/s는 `global_batch_size * max_length`를 callback wall time으로 나눈 configured-token 처리율이므로 variable-length 실행의 실제 non-padding token 처리율로 해석하지 않습니다.
Megatron timer는 `timing_log_level=1`에서 이미 계산한 timer의 rank-local `active_time` 차이를 읽으며 adapter 때문에 추가 collective를 실행하지 않습니다.

[verl profiler 설정](../observability/examples/verl/torch-profiler.yaml)은 외부 framework 연동 참고이며 이 저장소에 verl 학습 backend가 있다는 뜻은 아닙니다.
실제 사용하는 framework 버전에 맞춰 설정을 검증한 후 적용합니다.

[Selected-rank helper](../observability/examples/pytorch/selected_rank_profiler.py)는 선택하지 않은 rank에 no-op profiler를 돌려줍니다.
아래는 완성된 실행 명령이 아니라 기존 PyTorch loop에 삽입하는 예시입니다.
`observability`가 import 경로에 있어야 하며 `train_loader`와 `train_step`은 사용자의 학습 코드가 제공합니다.

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
기본적으로 shape·memory·stack 수집은 꺼져 있으며 필요한 질문에 한해서 켭니다.
출력은 `rank-<rank>/trace-<index>.json`이며 비교할 rank는 같은 run과 capture 구간이어야 합니다.
CPU의 kernel 제출 지연, NCCL과 compute의 겹침, rank별 collective 도착 시점, copy·동기화 집중 구간을 확인합니다.
원인을 수정한 뒤에는 profiler를 끈 실행에서 효과를 다시 검증합니다.
