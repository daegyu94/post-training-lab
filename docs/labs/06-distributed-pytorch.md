# Lab 06: End-to-end Distributed PyTorch Profiling

## Goal

한 번의 기준 run에서 cluster telemetry, rank별 step 시간과 PyTorch trace를 연결하는 최소 멀티 GPU 실습입니다. Megatron이나 verl을 아직 설치하지 않은 환경에서도 `torchrun`과 제공된 작은 DDP workload로 profiling 흐름을 검증할 수 있습니다. 실제 학습으로 바꿀 때도 관측 지점과 artifact 형식은 유지합니다.

이 실습은 GPU가 있는 한 node에서 시작하고, 같은 command를 scheduler가 할당한 여러 node로 확장합니다. profiler는 선택한 rank와 짧은 step window에서만 켜므로 모든 rank를 장시간 capture하지 않습니다.

## What This Lab Produces

```text
artifacts/ddp-profile/<run-id>/
├── run-manifest.txt
├── torchrun.log
└── traces/
    ├── rank-0/trace-0.json
    └── rank-<selected-rank>/trace-0.json
```

`trace-*.json`은 Perfetto와 HTA에서 열 수 있습니다. Node Exporter textfile collector를 설정했다면 같은 `run_id`로 host/GPU telemetry와 trace 시점을 비교할 수 있습니다.

## Prerequisites

- CUDA를 사용할 수 있는 PyTorch environment
- 한 node에서 GPU 두 장 이상 또는 여러 node에 걸친 scheduler allocation
- repository root가 `PYTHONPATH`에 포함된 shell
- multi-node일 때 node 간 rendezvous port 접근 가능
- 선택 사항: Lab 01에서 준비한 Node Exporter, DCGM Exporter, Prometheus

PyTorch Profiler는 PyTorch에 포함되어 있으므로 별도 profiler package가 필요하지 않습니다. HTA는 trace 분석을 할 별도 environment에만 설치합니다.

```bash
python -m pip install HolisticTraceAnalysis
```

## Single-node: Two-GPU Capture

다음 command는 global rank 0과 1을 대상으로, 처음 4 step을 건너뛴 뒤 한 번의 짧은 capture를 생성합니다.

```bash
export PYTHONPATH="$PWD"
export PROFILE_RUN_ID=ddp-$(date +%Y%m%d-%H%M%S)
export TRACE_OUTPUT_DIR="artifacts/ddp-profile/$PROFILE_RUN_ID"
mkdir -p "$TRACE_OUTPUT_DIR"

torchrun --standalone --nproc_per_node=2 \
  examples/pytorch/ddp_profile_demo.py \
  --steps 16 \
  --profile-ranks 0,1 \
  --trace-dir "$TRACE_OUTPUT_DIR/traces" \
  2>&1 | tee "$TRACE_OUTPUT_DIR/torchrun.log"
```

이 예제는 작은 TransformerEncoder와 synthetic batch를 사용합니다. 모델 품질이나 throughput benchmark가 아니라 rank별 GPU work, DDP gradient synchronization, CPU submit gap을 볼 수 있도록 만든 관측용 workload입니다.

다음처럼 run의 조건을 함께 보존합니다.

```bash
{
  printf 'run_id=%s\n' "$PROFILE_RUN_ID"
  python -c 'import torch; print(f"torch={torch.__version__}, cuda={torch.version.cuda}")'
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
  nvidia-smi topo -m
} | tee "$TRACE_OUTPUT_DIR/run-manifest.txt"
```

## Multi-node: Scheduler Allocation

multi-node 실행은 scheduler가 각 node에 한 process group member를 시작하게 해야 합니다. 아래는 Slurm의 한 가지 형태이며, node당 GPU 수와 launcher option은 클러스터 정책에 맞게 바꿉니다.

```bash
export PYTHONPATH="$PWD"
export PROFILE_RUN_ID=ddp-2nodes-$(date +%Y%m%d-%H%M%S)
export TRACE_OUTPUT_DIR="<shared-artifact-path>/$PROFILE_RUN_ID"

srun --nodes=2 --ntasks-per-node=1 --gpus-per-task=8 \
  torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node=8 \
    --node_rank="$SLURM_NODEID" \
    --master_addr="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)" \
    --master_port=29500 \
    examples/pytorch/ddp_profile_demo.py \
      --steps 24 \
      --profile-ranks 0,8 \
      --trace-dir "$TRACE_OUTPUT_DIR/traces"
```

공유 filesystem이 없다면 각 node의 local trace directory에 저장한 뒤, job 종료 전에 중앙 artifact store로 복사합니다. 전송 완료 전에는 trace를 삭제하지 않습니다. 노드 간 trace의 시간 정렬이 필요하므로 NTP/PTP 상태와 정확한 rank-to-node mapping도 manifest에 남깁니다.

## Inspect the Result

### Perfetto

브라우저에서 [Perfetto UI](https://ui.perfetto.dev/)를 열어 `trace-0.json`을 올립니다. 정상 rank와 느린 rank를 나란히 놓고 다음을 확인합니다.

- `aten::` CPU operator와 CUDA kernel 사이에 긴 빈 구간이 있는가
- `nccl:` 또는 `ncclKernel` 구간이 backward와 overlap되는가
- 한 rank가 collective에 다른 rank보다 늦게 도착하는가
- memory allocation, device-to-device copy 또는 synchronization이 step 경계에 몰리는가

### HTA

같은 capture window의 rank trace만 한 directory에 모은 후 분석합니다.

```python
from hta.trace_analysis import TraceAnalysis

analysis = TraceAnalysis(trace_dir="artifacts/ddp-profile/<run-id>/traces")
print(analysis.get_temporal_breakdown())
```

HTA의 결과는 trace가 같은 world size, rank naming, capture schedule을 사용했을 때만 비교합니다. 한 rank만 capture한 경우에는 HTA의 cross-rank 결론을 내리지 말고 Perfetto와 baseline telemetry를 함께 봅니다.

## Scale-up Decision Flow

1. profiler 없이 기준 run을 실행해 step p50/p95, tokens/s, node/GPU utilization과 NCCL/fio baseline을 수집합니다.
2. 느린 node 또는 반복되는 straggler rank를 Megatron timer, verl role metric, Ray metric으로 좁힙니다.
3. 정상 비교군 한 개를 포함한 selected rank와 2~5 step만 capture합니다.
4. trace에서 compute, collective, CPU submit, data/checkpoint 중 어느 계층인지 확인합니다.
5. 변경 후 profiler를 끈 기준 run을 다시 실행해 throughput과 quality/reward가 실제로 개선됐는지 확인합니다.

## Expected Result

single-node에서는 rank별 DDP communication과 compute timeline을, multi-node에서는 node 0의 rank 0과 node 1의 rank 8 trace를 동일 run에서 비교할 수 있어야 합니다. trace를 통해 원인을 확정하지 못하면 `docs/tooling.md`의 coverage gap에 따라 Nsight Systems 또는 Nsight Compute가 필요한 질문인지 판단합니다.
