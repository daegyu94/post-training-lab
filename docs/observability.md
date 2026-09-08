# Observability

`observability/`는 host·GPU 계측, monitoring stack 검사, 통신·저장소 baseline과 PyTorch trace 예제를 제공합니다.
먼저 상시 지표로 이상 구간을 찾고 필요한 rank만 짧게 trace합니다.
Synthetic workload와 실제 LLM 학습 결과는 구분합니다.

## CPU Checks

저장소 루트에서 아래 순서로 실행합니다.
Setup은 `.venv`를 만들고 pytest를 설치하지만 CUDA PyTorch, Docker, NCCL Tests, fio나 Python package 자체를 설치하지 않습니다.
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

## Monitoring with Docker Compose

`observability`에서 `examples/observability/targets/`의 node·GPU·application endpoint를 실제 환경에 맞춥니다.
파일 형식 검사는 네트워크나 Docker 없이 실행할 수 있습니다.

```bash
python -m profiling_lab.observability check-targets \
  --target-dir examples/observability/targets
```

Stack 실행에는 Docker 접근 권한과 Compose v2가 필요합니다.
다음 명령은 이미지를 내려받고 서비스를 시작하며 검사가 끝나도 계속 실행합니다.
제공된 Compose는 Prometheus 9090과 Grafana 3000 포트를 host에 공개하므로 접근 제어를 먼저 확인합니다.
비밀번호는 대화형으로 입력하며 저장소에 기록하지 않습니다.

```bash
read -r -s -p 'Grafana admin password: ' GRAFANA_ADMIN_PASSWORD
export GRAFANA_ADMIN_PASSWORD
bash scripts/validate_observability.sh
```

기본 검사는 target이 down이어도 stack readiness·health를 확인합니다.
Target도 모두 up이어야 하면 `REQUIRE_TARGETS_UP=1 bash scripts/validate_observability.sh`를 사용합니다.
결과는 기본 `artifacts/observability-validation/summary.json`에 기록합니다.
서비스를 중지하려면 같은 `observability` 디렉터리에서 다음을 실행합니다.
이 명령은 Compose 서비스를 중지·제거하지만 named volume은 삭제하지 않습니다.

```bash
(cd examples/observability && docker compose down)
```

## Spark Without Docker

`scripts/install_spark_tools.sh`는 ARM64 userspace 도구를 다운로드합니다.
실제 설치 경로는 `TOOLS_DIR`로 지정할 수 있으며 driver나 system package 설치를 대신하지 않습니다.
준비된 도구가 있는 각 노드의 `observability`에서 node role을 실행합니다.
이 명령은 지정 주소의 19100 포트에 node exporter를 열고 GPU sampler를 기본 900초 동안 실행합니다.

```bash
NODE_ADDR='<node-management-address>' \
  bash scripts/run_spark_observability.sh node
```

도구가 설치된 ARM64 monitoring host에서는 server role을 별도 실행합니다.
앞 절처럼 `GRAFANA_ADMIN_PASSWORD`를 설정하고 실제 두 주소를 지정합니다.

```bash
SPARK1_ADDR='<first-node-management-address>' \
SPARK2_ADDR='<second-node-management-address>' \
  bash scripts/run_spark_observability.sh server
```

이 경로는 Compose와 달리 Prometheus를 loopback 19090, Grafana를 loopback 13000에 바인딩합니다.
Server는 종료할 때까지 실행하므로 작업을 마치면 해당 세션을 종료합니다.
일부 GB10 NVML 값은 unavailable/null이며 이를 사용량 0으로 해석하지 않습니다.
시스템 메모리와 학습 process의 CUDA allocated/reserved peak도 구분합니다.

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
