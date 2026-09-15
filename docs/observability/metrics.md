# Metrics Contract

Metrics Contract는 관측 데이터를 기록하고 비교할 때 사용할 이름·단위·측정 범위를 정의합니다.
기준 파일은 [`config/metrics.json`](../../observability/config/metrics.json)이며, 서로 다른 collector와 framework가 같은 의미의 값을 같은 방식으로 표현하도록 돕습니다.

이 파일은 모든 metric을 자동으로 수집하거나 Prometheus 이름을 자동 변환하는 runtime registry가 아닙니다.
현재 dashboard는 exporter의 원본 metric을 직접 조회하고, framework metric은 textfile collector가 별도로 발행합니다.

## Who Uses the Contract

| 사용자 | 사용 목적 | 현재 연결 방식 |
| --- | --- | --- |
| Collector·framework adapter 개발자 | 새 metric의 canonical name, unit, scope 결정 | 계약을 구현 기준으로 사용 |
| Summary·분석 코드 개발자 | 서로 다른 run과 backend의 값을 같은 단위로 비교 | canonical vocabulary를 출력 기준으로 사용 |
| Dashboard 작성자 | exporter 원본 metric의 의미와 변환 단위 확인 | 원본 Prometheus metric을 직접 query |
| Schema validator·test | 계약 파일의 구조, 필수 field, 중복 검사 | [`schema.py`](../../observability/profiling_lab/schema.py)가 JSON을 검증 |

즉, 계약은 생산자와 소비자가 따라야 할 공통 규칙입니다.
metric을 실제로 수집하려면 collector나 adapter 구현이 별도로 필요합니다.

## Contract Structure

| Top-level field | 역할 |
| --- | --- |
| `schema_version` | 소비자가 이해하는 계약 형식 version |
| `metrics` | canonical metric 정의 목록 |
| `recommended_labels` | 비교와 filtering에 사용할 label 후보 |
| `manifest_only_fields` | Prometheus label 대신 manifest에 기록할 값 |
| `phase_vocabulary` | framework 간에 공통으로 사용할 실행 단계 이름 |

기존 metric의 의미나 단위를 바꾸는 호환성 파괴 변경에만 `schema_version`을 올립니다.
같은 의미를 유지한 새 metric 추가는 현재 version에서 처리합니다.

## Metric Entry

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

| Field | 답하는 질문 | 예시 의미 |
| --- | --- | --- |
| `name` | 어떤 이름으로 저장하는가? | stable snake_case canonical name |
| `category` | 어느 관측 영역에 속하는가? | training, GPU, network, storage, checkpoint |
| `unit` | 어떤 단위로 저장하는가? | dashboard 변환 전의 bytes/s, seconds 등 |
| `scope` | 값이 어디에 속하는가? | run, node, GPU, rank, phase, device, path |
| `source` | 원본은 무엇인가? | exporter, framework timer, trace, manifest, derived value |
| `policy` | 언제 수집하는가? | always-on, workload-specific, baseline, diagnostic |

Derived metric은 원본 값을 보존하고 계산 window와 source metric을 summary에 기록합니다.
예를 들어 bytes와 duration으로 대역폭을 계산했다면 두 원본과 계산 구간을 함께 남깁니다.

## Labels and Manifest Fields

Label은 Prometheus에서 시계열을 filtering하고 비교하는 데 사용합니다.
값의 종류가 계속 늘어나면 시계열 수가 급증하므로 제한된 값만 label로 사용합니다.

권장 label 후보:

- 실행: `run_id`, `cluster`, `job`, `framework`, `role`, `phase`
- 위치: `node`, `gpu`, `device`, `interface`
- 병렬 실행: `rank`, `local_rank`, `tp_rank`, `pp_rank`, `dp_rank`, `parallel_group`
- 작업 종류: `operation`, `timer`

다음 값은 종류가 많거나 문자열이 길어 Prometheus label로 적합하지 않습니다.
`manifest_only_fields`에 기록해 실행 재현과 사후 분석에 사용합니다.

- commit과 image digest
- model·dataset·checkpoint URI
- rank map과 profiler option
- precision, batch, sequence 설정
- storage path type, filesystem, cache state, node topology

Prompt, request ID, timestamp, trace ID처럼 계속 새 값이 생기는 항목도 Prometheus label로 사용하지 않습니다.

## Workflow Phases

`phase_vocabulary`는 framework가 달라도 같은 lifecycle 구간을 비교하기 위한 이름입니다.

| Phase | 대표 신호 |
| --- | --- |
| `dataset_loading` | storage read, metadata operation, preprocessing, data wait |
| `model_loading` | checkpoint read, host staging, host-to-GPU copy |
| `training_input` | pinned memory, host-to-GPU bytes/time, compute overlap |
| `forward_backward` | GPU compute, activation·gradient movement, collective |
| `optimizer_step` | AllReduce, ReduceScatter, AllGather, rank synchronization |
| `checkpoint_save` | training pause, GPU-to-host staging, write bandwidth·volume |
| `checkpoint_restore` | read bandwidth, host-to-GPU restore, rank synchronization |
| `evaluation` | inference compute, input transfer, idle time |

Phase marker에는 최소한 `run_id`, `phase`, 시작·종료 시각, 성공 여부를 기록합니다.
bytes와 duration을 모두 얻으면 유효 대역폭을 계산하고 같은 path의 hardware baseline과 비교할 수 있습니다.

## Data Movement Paths

항상 수집하는 지표로 범위를 좁힌 뒤, 원인이 남을 때만 diagnostic 도구를 사용합니다.

| Path | Always-on evidence | Diagnostic evidence |
| --- | --- | --- |
| storage → host memory | Node Exporter disk·filesystem·mountstats | iostat 또는 selected I/O trace |
| host memory → GPU | framework data wait, pinned memory | PyTorch Profiler copy event, Nsight |
| node 내부 GPU ↔ GPU | DCGM utilization, framework collective timer | NCCL Tests single-node baseline |
| node ↔ node | NIC·InfiniBand counter, communication timer | NCCL Tests multi-node baseline, GPUDirect RDMA |
| GPU·host → checkpoint storage | checkpoint timer, storage throughput·volume | writer·rank coordination trace |

Node Exporter와 DCGM만으로는 bytes가 어떤 phase나 rank에서 발생했는지 알 수 없습니다.
Framework phase marker와 rank map을 같은 `run_id`로 연결해야 합니다.

RoCE는 TCP/IP와 RDMA counter를 함께 확인합니다.

- socket traffic: `node_network_*`
- RDMA verbs: `node_infiniband_port_data_{received,transmitted}_bytes_total`
- canonical contract: `rdma_receive_bytes_per_second`, `rdma_transmit_bytes_per_second`, `rdma_errors_total`

NCCL IB transport처럼 kernel network stack을 우회하는 traffic은 `network_*`만으로 판단할 수 없습니다.

## Framework Integration

Runner는 output 이름을 `OBSERVATORY_RUN_ID`로 설정합니다.
TRL·Megatron callback은 `<output>/framework-metrics/`의 rank JSON을 atomic replace합니다.

| 사용 경로 | Reader |
| --- | --- |
| 실시간 dashboard | [textfile collector](monitoring.md#live-framework-metrics) |
| 종료된 run 요약 | [`show_run`](analysis.md#run-history) |

처리량의 의미는 framework마다 다릅니다.

- TRL tokens/s: Trainer가 보고한 누적 입력 token의 차이
- Megatron tokens/s: `global_batch_size * max_length`를 callback wall time으로 나눈 configured-token 처리율

Megatron 값은 variable-length 실행의 실제 non-padding token 처리율이 아닙니다.
Megatron timer는 `timing_log_level=1`에서 계산된 rank-local `active_time` 차이를 읽으며 adapter 때문에 추가 collective를 실행하지 않습니다.

[Selected-rank helper](../../observability/examples/pytorch/selected_rank_profiler.py)는 선택하지 않은 rank에 no-op profiler를 돌려줍니다.
다음 코드는 기존 PyTorch loop에 profiler를 삽입하는 예시입니다.

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
shape·memory·stack 수집은 기본적으로 꺼져 있으며 필요한 질문이 있을 때만 켭니다.
비교할 rank는 같은 run과 capture 구간을 사용해야 합니다.
원인을 수정한 뒤에는 profiler를 끈 실행에서 효과를 다시 검증합니다.

[verl profiler 설정](../../observability/examples/verl/torch-profiler.yaml)은 외부 framework 연동 참고이며 이 저장소에 verl backend가 있다는 뜻이 아닙니다.

## Change and Validate the Contract

새 metric을 추가할 때는 다음 순서로 진행합니다.

1. exporter나 framework에서 실제 source를 얻을 수 있는지 확인합니다.
2. canonical name, unit, scope, collection policy를 정합니다.
3. `metrics.json`, adapter, summary, dashboard를 필요한 범위에서 함께 갱신합니다.
4. schema validation을 실행합니다.

```bash
python -m pytest -q tests/observability/test_schema.py
```

검사는 JSON 문법, 허용된 이름·분류, 필수 field, 중복 이름을 확인합니다.
