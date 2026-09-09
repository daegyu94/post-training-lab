# Observability

`observability/`는 host·GPU 계측, Spark monitoring, 통신·저장소 baseline과 PyTorch trace 예제를 제공합니다.
먼저 상시 지표로 이상 구간을 찾고 필요한 rank만 짧게 trace합니다.
Synthetic workload와 실제 LLM 학습 결과는 구분합니다.

## CPU Checks

저장소 루트에서 아래 순서로 실행합니다.
Setup은 `.venv`를 만들고 pytest를 설치하지만 CUDA PyTorch, NCCL Tests, fio나 Python package 자체를 설치하지 않습니다.
모듈은 `observability`를 작업 디렉터리로 사용해 import합니다.

```bash
cd observability
bash scripts/setup.sh
. .venv/bin/activate
bash scripts/check_tools.sh
python -m pytest -q ../tests/observability
```

도구 검사에서 미설치 도구가 표시되면 해당 기능의 준비가 안 된 것이며 전체 기능 실패로 해석하지 않습니다.
지표 이름·단위·수집 범위와 framework 연결은 [Observability Reference](observability-reference.md)를 확인합니다.

## Spark Monitoring

`scripts/install_spark_tools.sh`는 ARM64 userspace 도구를 다운로드합니다.
실제 설치 경로는 `TOOLS_DIR`로 지정할 수 있으며 driver나 system package 설치를 대신하지 않습니다.
준비된 도구가 있는 각 노드의 `observability`에서 node role을 실행합니다.
이 명령은 지정 주소의 19100 포트에 node exporter를 열고 GPU sampler를 기본 900초 동안 실행합니다.

```bash
NODE_ADDR='<node-management-address>' \
  bash scripts/run_spark_observability.sh node
```

도구가 설치된 ARM64 monitoring host에서는 server role을 별도 실행합니다.
`GRAFANA_ADMIN_PASSWORD`를 설정하고 실제 두 주소를 지정합니다.

```bash
SPARK1_ADDR='<first-node-management-address>' \
SPARK2_ADDR='<second-node-management-address>' \
  bash scripts/run_spark_observability.sh server
```

Prometheus는 loopback 19090, Grafana는 loopback 13000에 바인딩합니다.
Server는 종료할 때까지 실행하므로 작업을 마치면 해당 세션을 종료합니다.
일부 GB10 NVML 값은 unavailable/null이며 이를 사용량 0으로 해석하지 않습니다.
시스템 메모리와 학습 process의 CUDA allocated/reserved peak도 구분합니다.

## Controller에서 결과 수집과 표시

각 Spark 노드의 GPU sampler는 원본 값을 JSONL로 저장하고 Node Exporter가 읽을 지표 파일도 갱신합니다.
학습 launcher는 같은 `run_id` 아래에 rank별 로그, summary와 measurement JSONL을 남깁니다.

```text
spark1 telemetry --+
                  +--> controller collector --> SQLite --> 실시간 확인
spark2 telemetry --+                            |
                                                +--> JSON export --> GitHub Pages
```

[Post-Training Lab Observatory](https://github.com/daegyu94/post-training-lab-observatory)는 Python 표준 라이브러리로 만든 controller collector를 제공합니다.
Spark 노드는 SSH reverse tunnel을 통해 측정값을 보내므로 collector port를 외부에 열지 않고도 controller 화면에서 최신 값을 확인할 수 있습니다.

GitHub Pages는 정적 사이트라 측정값을 직접 받을 수 없습니다.
공개할 run은 controller에서 JSON snapshot으로 내보내 Observatory 저장소에 반영한 뒤 Pages에서 확인합니다.

공통 runner는 output 이름을 `run_id`로 사용하고 TRL과 Megatron adapter가 rank별 최신 metric을 `<output>/framework-metrics/`에 atomic JSON으로 기록합니다.
TRL은 Trainer가 집계한 loss와 실제 누적 입력 token 차이를 사용하고, Megatron은 callback loss, step wall time, 설정된 batch·sequence 상한 기반 tokens/s와 rank timer를 기록합니다.

Observatory node agent에 같은 `run_id`와 `--framework-metrics-dir <output>/framework-metrics`를 주면 노드 전체 CPU·메모리·NIC와 framework metric이 함께 collector로 전송됩니다.
학습 process는 collector에 직접 접속하지 않으므로 collector 또는 tunnel 장애가 학습 step을 막지 않습니다.
직접 backend launcher를 실행할 때에는 `OBSERVATORY_RUN_ID`와 `FRAMEWORK_METRICS_DIR`를 함께 설정해야 adapter가 활성화됩니다. 공통 runner를 쓰면 두 값을 자동으로 설정하므로 따로 지정할 필요가 없습니다.

### Lightweight Local Viewer (History Server Pattern)

위 Observatory 경로는 SSH reverse tunnel과 별도 collector 프로세스가 필요합니다.
Controller와 두 Spark 노드가 이미 같은 NFS 공유 디렉터리를 보고 있으므로(`AGENTS.md`의 NFS 절 참고), `FRAMEWORK_METRICS_DIR`를 node-local NVMe 대신 그 공유 경로로 지정하면 collector나 SSH tunnel 없이도 controller가 파일을 직접 읽을 수 있습니다.
Apache Spark나 MapReduce의 History Server와 같은 pull 방식입니다: 애플리케이션은 잘 알려진 공유 경로에 쓰기만 하고, 뷰어는 그 경로를 그냥 읽습니다.

```text
Observatory (push, 위 경로)                   Local viewer (pull, 이 경로)

spark1 --SSH tunnel--> collector --> SQLite    spark1 --+
spark2 --SSH tunnel--> collector       |                +--> NFS 공유 경로 (같은 파일, 복사 없음)
                        |                       spark2 --+          |
                        +--> JSON export                            v
                             --> GitHub Pages          controller가 직접 읽음 (SSH·tunnel·token 불필요)
                                                                     |
                                                                     v
                                                     정적 HTTP server + 브라우저 polling
```

`backends/trl/scripts/run_spark_cluster.sh`와 `backends/megatron/scripts/run_spark_cluster.sh`는 `FRAMEWORK_METRICS_DIR`가 이미 설정돼 있으면 그 값을 그대로 쓰고, 없으면 기존처럼 `<output>/framework-metrics`(node-local NVMe)를 기본값으로 씁니다.
공유 경로를 가리키게 하려면 launcher 호출 전에 `FRAMEWORK_METRICS_DIR`를 NFS 경로로 export하면 됩니다. `<output>/model`처럼 checkpoint 저장 경로는 이 값의 영향을 받지 않으므로 30B NVMe 실습처럼 checkpoint 자체는 node-local NVMe에 남기고 metric만 공유 경로로 보낼 수 있습니다.
뷰어는 그 경로의 `<framework>-rank-<rank>.json`을 주기적으로 `fetch`하는 간단한 정적 HTML이면 충분하며, 저장소에는 포함돼 있지 않습니다.
이 경로는 GitHub Pages에 값을 공개하지 않으며 controller에서만 보입니다. 외부에 공개하려면 위 Observatory push·export 절차를 그대로 따릅니다.
2026-09-09에 Qwen2.5-0.5B DDP smoke를 60 step으로 돌리며 이 방식을 실제로 검증했습니다: launcher가 공유 경로에 쓴 `trl-rank-0.json`을 controller가 SSH 없이 직접 읽었고, 약 8초 동안 step 4→60까지 13번의 서로 다른 값을 관찰했습니다.

## Distributed Trace

각 Spark 노드의 `observability`에서 CUDA Python과 같은 `PROFILE_RUN_ID`를 지정합니다.
첫 노드는 `NODE_RANK=0`, 다른 노드는 `1`로 실행합니다.
명령은 GPU 작업을 시작하며 preflight는 기존 GPU 작업과 최소 가용 메모리를 검사합니다.
다른 작업이 있으면 임의 종료하지 않고 끝난 후 실행합니다.

```bash
PROFILE_RUN_ID=profile-001 \
NODE_RANK=0 \
MASTER_ADDR='<first-node-data-address>' \
PYTHON='<cuda-python>' \
  bash scripts/run_spark_profile.sh baseline
```

같은 두 노드에서 모드만 `capture`로 바꾸면 rank 0·1 trace를 수집합니다.
기본 24 step이고 실행 제한은 300초이며 `STEPS`, `RUN_TIMEOUT`으로 조정합니다.
기본 출력은 `artifacts/spark/<run-id>/<mode>/`이고 `OUTPUT_DIR`로 변경합니다.
양 rank 로그·manifest·JSON 결과와 capture의 `traces/`를 확인합니다.
Baseline과 capture를 동시에 실행하지 말고 profiler overhead를 비교합니다.
이 예제는 synthetic DDP이며 실제 LLM trace가 아닙니다.

## Hardware Baselines

NCCL baseline은 MPI 지원 `all_reduce_perf`, `mpirun`과 할당받은 GPU 노드를 요구합니다.
아래 명령은 `observability`에서 실행하며 모든 지정 노드에 GPU 부하를 발생시킵니다.

```bash
NCCL_TEST_BINARY='<all-reduce-perf-path>' \
HOSTS='<first-host>,<second-host>' GPUS_PER_NODE=1 \
  bash scripts/run_nccl_baseline.sh
```

`artifacts/nccl-baseline/manifest.env`와 `all-reduce.log`에서 조건·correctness 오류·대역폭을 확인합니다.
NCCL baseline은 학습 throughput이 아닙니다.

fio는 저장소에 쓰기 부하를 주며 기본 16 GiB 테스트 파일을 남깁니다.
사용자 데이터가 없는 전용 시험 디렉터리와 여유 공간을 먼저 확인하고 `observability`에서 실행합니다.

```bash
FIO_DIRECTORY='<dedicated-test-directory>' \
  bash scripts/run_fio_baseline.sh
```

결과는 `artifacts/fio-baseline/checkpoint.json`입니다.
남은 `profiling-lab-checkpoint.bin`은 자동 삭제되지 않으므로 시험 후 정확한 파일을 확인해 별도로 정리합니다.
fio 결과는 checkpoint의 장애 후 복구나 fsync 보장을 증명하지 않습니다.
