# Lab 05: Selected-rank Trace

## Goal

상시 metric에서 확인한 이상 node/rank/role과 짧은 step window만 trace합니다. 전체 run과 모든 rank를 profile하여 기준 성능을 오염시키거나 shared storage를 trace로 포화시키지 않습니다.

## 1. Choose the Capture Set

capture 전에 다음을 manifest에 기록합니다.

- baseline에서 이상을 보인 global rank와 그 rank의 node, GPU, DP/TP/PP/CP/EP group 또는 verl role/replica
- capture할 global step과 warm-up/wait/active schedule
- CPU/CUDA, shape, memory와 stack option
- PyTorch, Kineto, CUDA와 framework version
- node clock synchronization 상태와 trace 저장 경로

Megatron에서는 straggler rank 외에 같은 parallel group의 정상 rank를 하나 포함해야 비교가 가능합니다. pipeline 문제라면 각 PP stage 대표 rank를 선택합니다. verl은 가능하면 Lab 04의 built-in profiler를 사용하고 actor와 rollout을 별도로 선택합니다.

## 2. Integrate the Helper in a PyTorch Loop

Megatron custom loop 또는 일반 distributed PyTorch loop에서는 [`selected_rank_profiler.py`](../../examples/pytorch/selected_rank_profiler.py)를 사용할 수 있습니다.

```python
from pathlib import Path

from examples.pytorch.selected_rank_profiler import selected_rank_profile

with selected_rank_profile(
    Path("artifacts/traces/run-123"),
    ranks={0, 8},
    skip_first=10,
    wait=1,
    warmup=1,
    active=2,
    record_shapes=False,
    profile_memory=False,
) as profiler:
    for batch in train_loader:
        train_step(batch)
        profiler.step()
```

`profiler.step()`은 모든 loop iteration에서 호출해야 schedule이 진행됩니다. `record_shapes`, `profile_memory`와 `with_stack`은 파일 크기와 overhead가 커지므로 질문에 필요한 option만 켭니다.

## 3. Inspect an Individual Timeline

생성된 Chrome trace JSON을 [Perfetto UI](https://ui.perfetto.dev/)에서 열고 다음 순서로 확인합니다.

1. CPU가 CUDA kernel을 늦게 submit하여 GPU queue가 비는지 확인합니다.
2. NCCL collective가 compute와 overlap되는지 또는 serialize되는지 확인합니다.
3. 정상 rank와 straggler rank가 같은 collective에 도착하는 시각을 비교합니다.
4. memory allocation/copy 또는 synchronization이 iteration boundary에 집중되는지 확인합니다.
5. user annotation이 data, forward, backward, optimizer, rollout과 checkpoint 구간을 구분하는지 확인합니다.

## 4. Analyze Multiple Rank Traces With HTA

모든 selected rank trace를 하나의 run directory 아래 모은 뒤 별도 analysis environment에서 Holistic Trace Analysis를 실행합니다.

```bash
python -m pip install HolisticTraceAnalysis
```

```python
from hta.trace_analysis import TraceAnalysis

analysis = TraceAnalysis(trace_dir="artifacts/traces/run-123")
print(analysis.get_temporal_breakdown())
```

HTA로 compute/communication/idle breakdown, kernel duration distribution, rank imbalance와 communication overlap을 비교합니다. trace 파일명이 rank를 안정적으로 나타내고 모든 trace가 같은 capture window를 포함하는지 먼저 검증합니다.

## 5. Decide Whether a Vendor Tool Is Needed

다음 질문이 남을 때만 제한된 reproduction에 vendor tool을 사용합니다.

- CUDA runtime, kernel과 NCCL을 포함한 system-wide timeline의 정확한 causality가 필요한가?
- 특정 kernel의 occupancy, memory throughput 또는 stall 원인을 알아야 하는가?
- framework 밖의 native thread/process와 GPU activity를 함께 정렬해야 하는가?

첫 번째와 세 번째에는 Nsight Systems, 두 번째에는 Nsight Compute가 보통 필요합니다. 해당 결과는 오픈소스 상시 stack과 분리된 diagnostic artifact로 보존합니다.

## Expected Result

trace에서 확인한 원인이 Prometheus의 시간대, framework timer와 rank placement에 연결되어야 합니다. trace 하나만 보고 결론을 내리지 않고 변경 후 profiler를 끈 baseline run에서 throughput과 품질 개선을 다시 확인합니다.
