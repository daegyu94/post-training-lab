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

현재 collector가 자동으로 받는 값은 노드 전체 CPU·메모리·NIC입니다.
TRL과 Megatron의 loss, step time, tokens/s와 rank timer를 실시간 화면에 넣으려면 같은 `run_id`를 사용하는 framework adapter가 추가로 필요합니다.

Adapter가 없어도 학습 결과는 각 backend가 생성한 rank 로그와 summary·measurement JSONL에 보존됩니다.

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
