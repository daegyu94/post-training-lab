# Run Analysis

이 문서는 dashboard에서 발견한 이상 징후의 원인을 좁히는 방법을 설명합니다.
실행 결과를 먼저 확인하고, 상시 지표만으로 원인을 판단할 수 없을 때 trace나 hardware baseline을 추가로 수집합니다.

## Choose the Evidence

조사하려는 질문에 맞는 가장 작은 증거부터 사용합니다.

| 질문 | 먼저 사용할 도구 | 확인할 내용 |
| --- | --- | --- |
| 실행이 어느 stage까지 진행됐는가? | `show_run` | stage 상태, rank별 마지막 step, output 위치 |
| 특정 구간에서 CPU·GPU·통신이 어떻게 겹치는가? | selected-rank PyTorch trace | kernel 제출, copy, collective, synchronization |
| 통신 성능이 hardware 한계에 가까운가? | NCCL baseline | topology 조건, correctness, collective bandwidth |

Trace와 baseline은 항상 필요한 절차가 아닙니다.
run summary와 [monitoring dashboard](monitoring.md)만으로 답할 수 없을 때 추가합니다.

## Analysis Workflow

1. Dashboard에서 이상이 발생한 시간 범위와 node·rank·run을 기록합니다.
2. `show_run`으로 실행 상태와 마지막 application metric을 확인합니다.
3. 원인이 남아 있으면 같은 조건에서 짧은 trace를 수집합니다.
4. 통신 병목이 의심되면 별도의 NCCL baseline과 비교합니다.
5. 원인을 수정한 뒤 profiler를 끈 실행에서 효과를 다시 측정합니다.

Synthetic trace나 NCCL baseline을 실제 LLM throughput으로 해석하지 않습니다.
비교할 실행은 model·batch·sequence·topology 등 성능에 영향을 주는 조건을 같게 유지합니다.

## Inspect Run State

[`show_run`](../../observability/profiling_lab/show_run.py)은 monitoring server 없이 한 output directory의 실행 상태를 요약합니다.

| 입력 | 표시하는 정보 |
| --- | --- |
| Megatron `run-metadata-<stage>.json` | stage별 실행 metadata와 상태 |
| TRL `summary-<stage>.json` | stage별 summary |
| `observatory-metrics/<producer>-<role>-<worker>.json` | worker별 마지막 step과 metric |

```bash
PYTHONPATH=observability python3 -m profiling_lab.show_run '<output-dir>'
```

Application metric을 보려면 실행 시 `OBSERVATORY_RUN_ID`와 `OBSERVATORY_METRICS_DIR`가 설정되어 있어야 합니다.
output directory가 node-local이라 controller에서 보이지 않으면 해당 node에서 명령을 실행합니다.

`show_run`이 보여 주는 application metric은 마지막 snapshot입니다.
전체 loss 추이나 step별 변화는 `logs/`의 학습 log를 확인합니다.

## Capture a Focused Trace

`run_profile.sh`는 profiler overhead를 비교할 수 있도록 synthetic DDP workload를 같은 조건에서 실행합니다.

| Mode | 동작 | 용도 |
| --- | --- | --- |
| `baseline` | profiler 없이 synthetic DDP 실행 | trace overhead 비교 기준 |
| `capture` | 선택한 rank의 PyTorch trace 수집 | CPU·GPU operation과 synchronization 확인 |
| `collective` | synthetic collective 실행 | 분산 통신 경로 확인 |

각 참여 node에서 같은 `PROFILE_RUN_ID`를 사용하고 `NODE_RANK`만 다르게 지정합니다.
`MASTER_ADDR`은 rank 0 node의 data-interface 주소입니다.

```bash
PROFILE_RUN_ID=profile-001 \
NODE_RANK=0 \
MASTER_ADDR='<rank-0-data-address>' \
PYTHON='<cuda-python>' \
  bash scripts/run_profile.sh baseline
```

`capture`를 수집할 때는 mode만 바꾸고 baseline과 동시에 실행하지 않습니다.

```bash
PROFILE_RUN_ID=profile-001 \
NODE_RANK=0 \
MASTER_ADDR='<rank-0-data-address>' \
PYTHON='<cuda-python>' \
  bash scripts/run_profile.sh capture
```

| 설정 | 기본값 |
| --- | --- |
| Step 수 (`baseline`, `capture`) | `STEPS=24` |
| 실행 제한 | `RUN_TIMEOUT=300`초 |
| 출력 | `artifacts/observability/<run-id>/<mode>/` |

명령은 기존 GPU process와 최소 가용 memory를 확인한 뒤 workload를 시작합니다.
다른 GPU 작업이 있으면 종료될 때까지 기다리며 임의로 중지하지 않습니다.

각 rank의 log·manifest·JSON 결과를 확인하고, `capture`에서는 `traces/`도 확인합니다.
이 workload는 trace 절차를 검증하는 synthetic DDP이며 실제 LLM 실행이 아닙니다.

실제 PyTorch training loop에 profiler를 넣는 방법은 [Application Integration](metrics.md#application-integration)을 따릅니다.

## Compare a Hardware Baseline

NCCL baseline은 training code와 분리된 collective 통신 성능을 측정합니다.
MPI를 지원하는 `all_reduce_perf`, `mpirun`, 할당된 GPU node가 필요하며 지정한 모든 node에 GPU 부하를 발생시킵니다.

```bash
NCCL_TEST_BINARY='<all-reduce-perf-path>' \
HOSTS='<first-host>,<second-host>' GPUS_PER_NODE=1 \
  bash scripts/run_nccl_baseline.sh
```

| 결과 | 확인할 내용 |
| --- | --- |
| `artifacts/nccl-baseline/manifest.env` | host, GPU 수, 환경변수 등 측정 조건 |
| `artifacts/nccl-baseline/all-reduce.log` | correctness 오류와 collective bandwidth |

NCCL baseline은 학습 throughput이 아닙니다.
같은 node·GPU·network 조건에서 측정한 값만 training communication metric의 비교 기준으로 사용합니다.

## Verify the Fix

분석이 끝나면 다음 항목을 확인합니다.

- 수정 전후 실행의 model·batch·sequence·topology 조건이 같은가?
- profiler를 끈 실행에서도 개선이 유지되는가?
- throughput 개선이 loss·correctness 오류와 맞바뀌지 않았는가?
- synthetic 결과와 실제 LLM 실행 결과를 구분해 기록했는가?
