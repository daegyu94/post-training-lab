# Profiling Metric Contract

이 문서는 post-training workload와 exporter가 같은 이름, 단위, scope로 metric을 기록하도록 하는 공통 계약을 설명합니다.
실제 source of truth는 [`config/metrics.json`](../config/metrics.json)이며 [`config/metrics.schema.json`](../config/metrics.schema.json)은 파일 형식을 검증하는 JSON Schema입니다.

## Contract Files

| File | Role |
| --- | --- |
| `config/metrics.json` | 수집·파생 metric, label, manifest field와 phase vocabulary의 canonical definition |
| `config/metrics.schema.json` | metric object의 필수 field, category와 naming rule을 정의하는 JSON Schema |
| `profiling_lab/schema.py` | 별도 dependency 없이 repository test와 script에서 수행하는 runtime validation |

`metrics.json`의 `schema_version`은 소비자가 이해하는 계약 version입니다.
기존 metric의 의미나 단위를 바꾸는 호환성 파괴 변경이 있을 때만 version을 올리고, 새 metric을 추가하는 변경은 같은 version에서 수행합니다.

## Metric Entry

각 metric은 다음 여섯 field를 가집니다.

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
현재 dashboard는 exporter 원본 이름을 조회하고, Megatron hook은 `llm_training_*`와 `llm_megatron_timer_seconds`를 내보냅니다.
Canonical name 변환과 phase별 bytes/time 집계는 workload adapter에서 추가 구현해야 합니다.
Derived metric은 원본 값을 덮어쓰지 않으며 계산에 사용한 window와 source metric을 summary에 함께 기록합니다.

## Labels and Manifest Fields

`recommended_labels`는 run 비교와 drill-down에 필요한 bounded-cardinality dimension입니다.
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
Bytes와 duration을 모두 얻을 수 있으면 `data_movement_effective_bandwidth_bytes_per_second`를 계산하고, 동일 path의 NCCL Tests 또는 fio baseline과 비교해 utilization ratio를 만듭니다.

## Data Movement Paths

| Path | Always-on evidence | Diagnostic evidence |
| --- | --- | --- |
| Local/remote storage → host memory | Node Exporter disk, filesystem, mountstats와 storage client metric | fio, iostat 또는 selected I/O trace |
| Host memory → GPU | framework data wait와 pinned memory | PyTorch Profiler memory copy event 또는 Nsight Systems |
| GPU ↔ GPU in one node | DCGM utilization과 framework collective timer | NCCL Tests single-node baseline, selected trace |
| GPU node ↔ GPU node | NIC/InfiniBand counter와 communication timer | NCCL Tests multi-node baseline, GPUDirect RDMA 확인 |
| GPU/host → checkpoint storage | checkpoint timer, storage throughput와 volume | fio checkpoint pattern, writer/rank coordination trace |

Node Exporter와 DCGM만으로는 bytes가 어떤 framework phase나 rank에서 발생했는지 알 수 없습니다.
Framework phase marker와 rank map을 같은 `run_id`로 연결하고, 원인이 남을 때만 selected trace를 수집합니다.

## Validate the Contract

Repository root에서 다음 명령으로 JSON syntax, vocabulary, category, required field와 중복 metric name을 확인합니다.

```bash
.venv/bin/python -m pytest -q tests/test_schema.py
```

새 metric을 추가할 때는 exporter 또는 framework에서 실제로 얻을 수 있는 source를 먼저 확인하고, canonical unit과 scope를 결정한 뒤 `metrics.json`, adapter, summary와 dashboard를 같은 변경에서 갱신합니다.
