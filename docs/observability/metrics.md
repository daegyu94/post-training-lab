# Metrics Contract

비교 가능한 지표의 이름·단위·측정 범위는 [`config/metrics.json`](../../observability/config/metrics.json)을 기준으로 합니다.
파일 형식은 [`config/metrics.schema.json`](../../observability/config/metrics.schema.json)에 정의하며 `observability/profiling_lab/schema.py`가 추가 의존성 없이 runtime validation을 수행합니다.

`metrics.json`의 `schema_version`은 소비자가 이해하는 계약 version입니다.
기존 metric의 의미나 단위를 바꾸는 호환성 파괴 변경에만 version을 올리고, 새 metric 추가는 같은 version에서 합니다.

## Metric entry

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
| `unit` | dashboard에서 GiB·Gbps로 변환하기 전의 canonical 저장 단위 |
| `scope` | run, node, GPU, rank, phase, device, mount, parallel group 등 값이 속한 범위 |
| `source` | exporter, framework timer, selected trace, manifest 또는 derived summary |
| `policy` | always-on, workload-specific, baseline, diagnostic 수집 조건 |

계약은 목표 vocabulary이며 자동 수집 목록이 아닙니다.
Dashboard는 exporter 원본 이름을 조회하고 학습 지표는 rank JSON을 textfile collector로 재발행합니다([Live framework metrics](monitoring.md#live-framework-metrics)).
Canonical name 변환·phase 집계는 workload adapter에 구현해야 합니다.
Derived metric은 원본을 보존하고 계산 window·source metric을 summary에 기록합니다.

## Labels and manifest fields

`recommended_labels`는 filter·비교 기준이며 시계열 증가를 제한하도록 값의 종류를 제한합니다.
공통 후보는 `run_id`, `cluster`, `job`, `node`, `gpu`, `framework`, `role`, `phase`, `device`, `interface`, `operation`, `parallel_group`입니다.
Megatron hook의 `rank`·`local_rank`·`tp_rank`·`pp_rank`·`dp_rank`·`timer`는 범위를 제한한 예제 확장입니다.

Commit, image digest, model/dataset/checkpoint URI, rank map, profiler option, precision, batch/sequence 설정, storage path type, filesystem, cache state, node topology는 `manifest_only_fields`에 기록합니다.
Prompt, request ID, timestamp, trace ID처럼 계속 늘어나는 값은 Prometheus label로 쓰지 않습니다.

## Workflow phases

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

Phase marker에는 최소한 `run_id`, `phase`, 시작·종료 시각, 성공 여부를 기록합니다.
Bytes와 duration을 모두 얻으면 유효 대역폭을 계산하고 같은 path의 NCCL Tests baseline과 비교해 utilization ratio를 만듭니다.

## Data movement paths

| Path | Always-on 증거 | Diagnostic 증거 |
| --- | --- | --- |
| 저장소 → host memory | Node Exporter disk·filesystem·mountstats | iostat 또는 selected I/O trace |
| Host memory → GPU | framework data wait, pinned memory | PyTorch Profiler memory copy event, Nsight |
| node 내 GPU ↔ GPU | DCGM utilization, framework collective timer | NCCL Tests single-node baseline |
| node ↔ node | NIC/InfiniBand counter, communication timer | NCCL Tests multi-node baseline, GPUDirect RDMA |
| GPU/host → checkpoint storage | checkpoint timer, storage throughput·volume | writer/rank coordination trace |

Node Exporter와 DCGM만으로는 bytes가 어떤 phase나 rank에서 발생했는지 알 수 없습니다.
Framework phase marker와 rank map을 같은 `run_id`로 연결하고, 원인이 남을 때만 selected trace를 수집합니다.

RoCE는 TCP/IP와 RDMA counter를 함께 봅니다.
소켓 traffic은 `node_network_*`, kernel을 우회하는 RDMA verbs(NCCL IB transport 포함)는 `/sys/class/infiniband`에서 집계됩니다.
기본 활성화된 Node Exporter `infiniband` collector는 `node_infiniband_port_data_{received,transmitted}_bytes_total`을 노출합니다.
계약의 `rdma_receive_bytes_per_second`·`rdma_transmit_bytes_per_second`·`rdma_errors_total`과 dashboard의 RDMA panel을 확인하며 `network_*`만으로 판단하지 않습니다.

## Validate the contract

```bash
python -m pytest -q tests/observability/test_schema.py
```

JSON 문법, 허용된 이름·분류, 필수 field, 중복 이름을 확인합니다.
새 metric을 추가할 때는 exporter나 framework에서 실제로 얻을 수 있는 source를 먼저 확인합니다.
그 다음 canonical unit과 scope를 정하고 `metrics.json`·adapter·summary·dashboard를 같은 변경에서 갱신합니다.

## Framework integration

Runner는 output 이름을 `OBSERVATORY_RUN_ID`로 설정하고 TRL·Megatron callback은 `<output>/framework-metrics/`의 rank JSON을 atomic replace합니다.
실시간 조회는 [textfile collector](monitoring.md#live-framework-metrics), 과거 조회는 [`show_run`](analysis.md#run-history)이 읽습니다.

TRL tokens/s는 Trainer의 누적 입력 token 차이입니다.
Megatron tokens/s는 `global_batch_size * max_length`를 callback wall time으로 나눈 configured-token 처리율이므로 variable-length 실행의 실제 non-padding 처리율로 해석하지 않습니다.
Megatron timer는 `timing_log_level=1`에서 이미 계산된 timer의 rank-local `active_time` 차이를 읽으며 adapter 때문에 추가 collective를 실행하지 않습니다.

[Selected-rank helper](../../observability/examples/pytorch/selected_rank_profiler.py)는 선택하지 않은 rank에 no-op profiler를 돌려줍니다.
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
CPU의 kernel 제출 지연, NCCL과 compute의 겹침, rank별 collective 도착 시점, copy·동기화 집중 구간을 확인합니다.
원인을 수정하면 profiler를 끈 실행에서 효과를 다시 검증합니다.

[verl profiler 설정](../../observability/examples/verl/torch-profiler.yaml)은 외부 framework 연동 참고이며 이 저장소에 verl backend가 있다는 뜻이 아닙니다.
