# Lab 03: Megatron Resource Profiling

## Goal

이 실습은 Megatron 학습에서 느린 프로세스(rank)와 학습 단계를 찾는 방법을 설명합니다.
Rank는 분산 작업에 참여하는 프로세스의 번호이고, straggler는 다른 rank보다 반복해서 늦는 프로세스입니다.

기존 Megatron launch command는 그대로 사용하면서 cluster telemetry와 Megatron이 이미 계산하는 timer/straggler 값을 같은 run으로 연결합니다.
parent launcher PID가 아니라 node, GPU, rank와 parallel group의 분포를 분석합니다.

이 문서의 명령과 Python 예제는 `observability` 디렉터리를 작업 디렉터리로 사용합니다.
실제 GPU 연산은 Spark 노드에서 실행하고, 다른 클러스터용 예시는 해당 환경에 맞춰 적용합니다.

## 1. Record the Parallel Topology

실행 전에 `run_id`를 만들고 manifest에 world size, node list, rank placement, DP/TP/PP/CP/EP size와 software version을 기록합니다.
다음 환경 변수는 metric hook이 공통 identity를 붙이는 예입니다.

```bash
export PROFILE_RUN_ID=<stable-run-id>
export NODE_EXPORTER_TEXTFILE_DIR=/var/lib/node_exporter/textfile_collector
```

`RANK`와 `LOCAL_RANK`는 `torchrun`이 설정합니다.
`TP_RANK`, `PP_RANK`, `DP_RANK`는 Megatron parallel state에서 얻어 process 시작 시 환경 또는 hook argument로 전달합니다.
topology는 Prometheus label만 믿지 말고 완전한 rank map을 artifact로도 보존합니다.

## 2. Enable Megatron's Existing Signals

사용 중인 Megatron version의 `--help`에서 다음 option을 확인한 뒤 기존 launch command에 추가합니다.

```text
--log-throughput
--timing-log-level 1
--timing-log-option minmax
--tensorboard-dir <run-artifact-dir>/tensorboard
--log-timers-to-tensorboard
--log-straggler
--straggler-minmax-count <N>
```

`timing-log-option=minmax`는 rank spread를 낮은 비용으로 보여주고 `all`은 진단 시점에만 사용합니다.
`--log-straggler`는 지원되는 Megatron-LM 경로에서 GPU timing과 rank별 차이를 찾는 데 사용합니다.
version과 training integration에 따라 option이 달라질 수 있으므로 실제 binary의 argument 목록을 manifest에 함께 보존합니다.

## 3. Export Selected Metrics to Prometheus

Node Exporter를 다음 textfile directory mount와 option으로 실행해야 합니다.

```text
-v /var/lib/node_exporter/textfile_collector:/var/lib/node_exporter/textfile_collector
--collector.textfile.directory=/var/lib/node_exporter/textfile_collector
```

[`metric_hook.py`](../../../observability/examples/megatron/metric_hook.py)의 `export_megatron_step()`을 기존 logging interval에서 호출합니다.
새 timer를 만들지 않고 Megatron이 이미 계산한 iteration time, throughput과 timer 결과만 전달합니다.
각 rank는 자기 `.prom` 파일만 원자적으로 교체하므로 port 충돌과 부분 scrape를 피합니다.

```python
from examples.megatron.metric_hook import export_megatron_step

export_megatron_step(
    iteration=iteration,
    step_time_seconds=step_time,
    tokens_per_second=tokens_per_second,
    timer_seconds={
        "forward-compute": forward_seconds,
        "backward-compute": backward_seconds,
        "optimizer": optimizer_seconds,
        "data-loader": data_loader_seconds,
    },
)
```

예제 import path는 `observability` 디렉터리가 `PYTHONPATH`에 있는 실습 환경을 가정합니다.
production에서는 helper를 training image의 package에 포함합니다.
stale metric을 피하려면 job epilogue에서 현재 run의 정확한 rank 파일만 제거하거나 별도 directory를 allocation마다 mount합니다.

## 4. Observe a Baseline Run

profiler를 켜지 않은 run에서 다음을 비교합니다.

- tokens/s와 step p50/p95/p99
- rank step time의 min/max spread와 반복되는 straggler rank
- GPU utilization/power/clock과 rank placement
- data loader, forward, backward, optimizer와 checkpoint timer
- network throughput과 collective 시간이 동시에 증가하는지 여부

Megatron timer가 TensorBoard에는 rank 최대값만 기록할 수 있으므로 rank distribution이 필요하면 console/raw artifact 또는 adapter에서 `minmax/all` 결과를 별도로 보존합니다.

## 5. Interpret Common Patterns

| Pattern | Likely hypothesis | Next evidence |
| --- | --- | --- |
| 한 rank가 느리고 해당 GPU clock/power가 낮음 | GPU throttling 또는 hardware anomaly | DCGM health와 straggler history |
| 모든 rank에서 collective 증가, NCCL baseline 저하 | fabric 문제 | NIC error/drop, topology와 NCCL logs |
| collective 증가, NCCL baseline 정상 | 늦게 collective에 도착하는 rank | selected rank trace와 data/compute timer |
| 첫/마지막 PP stage의 idle이 큼 | pipeline bubble 또는 stage imbalance | PP stage별 trace와 layer placement |
| checkpoint 때 모든 GPU idle | serialization/storage/barrier | checkpoint bytes/time, fio와 rank coordination |

## Expected Result

throughput regression을 특정 node/GPU/rank와 Megatron phase까지 좁힐 수 있어야 합니다.
원인이 여전히 compute/communication 내부에 남을 때만 Lab 05의 selected trace를 수행합니다.

## Reference

- [Megatron Core timers](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.timers.html)
- [Megatron StragglerDetector](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/README_STRAGGLER.md)
