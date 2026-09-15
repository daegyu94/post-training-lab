# 실행 분석

상시 지표에서 시간 범위·node·rank를 먼저 좁힙니다.
그 다음 과거 run 요약, 짧은 selected-rank trace, hardware baseline을 차례로 사용해 원인을 확인합니다.
trace는 원인을 확인하는 진단 도구이므로, 수정 효과는 profiler를 끈 실행에서 다시 검증합니다.

## Run history

[`show_run`](../../observability/profiling_lab/show_run.py)은 monitoring server 없이 output directory의 Megatron `run-metadata-<stage>.json`과 TRL `summary-<stage>.json`을 읽습니다.
`OBSERVATORY_RUN_ID`와 `FRAMEWORK_METRICS_DIR`가 설정됐다면 `framework-metrics/<framework>-rank-<rank>.json`의 마지막 step도 표시합니다.

```bash
PYTHONPATH=observability python3 -m profiling_lab.show_run '<output-dir>'
```

output directory가 실행 node의 local path라 controller에서 보이지 않으면 해당 node에서 실행합니다.
`framework-metrics`는 학습 중 최신 step 값이며 끝나면 마지막 값에 고정됩니다.
전체 loss 추이가 필요하면 `logs/`의 학습 log를 확인합니다.

## Distributed trace

각 참여 node에서 같은 `PROFILE_RUN_ID`를 지정하고 첫 node는 `NODE_RANK=0`, 다음 node는 `1`로 실행합니다.
이 명령은 GPU 작업을 시작하며 preflight가 기존 GPU 작업과 최소 가용 메모리를 검사합니다.
다른 작업이 있으면 임의로 종료하지 않고, 작업이 끝난 뒤 실행합니다.

```bash
PROFILE_RUN_ID=profile-001 \
NODE_RANK=0 \
MASTER_ADDR='<first-node-data-address>' \
PYTHON='<cuda-python>' \
  bash scripts/run_profile.sh baseline
```

모드만 `capture`로 바꾸면 rank 0·1 trace를 수집합니다.
기본값은 24 step, 실행 제한은 300초이며 `STEPS`와 `RUN_TIMEOUT`으로 조정합니다.
출력은 기본 `artifacts/observability/<run-id>/<mode>/`이고, 각 rank의 log·manifest·JSON 결과와 capture의 `traces/`를 확인합니다.
baseline과 capture를 동시에 실행하지 말고 profiler overhead를 비교합니다.
이 예제는 synthetic DDP이며 실제 LLM trace가 아닙니다.

기존 PyTorch loop에 직접 삽입하는 selected-rank profiler는 [Framework integration](metrics.md#framework-integration)을 따릅니다.

## Hardware baselines

NCCL baseline은 MPI 지원 `all_reduce_perf`, `mpirun`, 할당받은 GPU node를 요구하며 지정한 모든 node에 GPU 부하를 발생시킵니다.

```bash
NCCL_TEST_BINARY='<all-reduce-perf-path>' \
HOSTS='<first-host>,<second-host>' GPUS_PER_NODE=1 \
  bash scripts/run_nccl_baseline.sh
```

`artifacts/nccl-baseline/manifest.env`와 `all-reduce.log`에서 조건·correctness 오류·대역폭을 확인합니다.
NCCL baseline은 학습 throughput이 아니며, metric 해석 기준을 위한 별도 통신 측정입니다.
