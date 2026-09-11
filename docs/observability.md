# Observability

`observability/`는 host·GPU 계측, Spark monitoring, 통신·저장소 baseline과 PyTorch trace 예제를 제공합니다.
먼저 상시 지표로 이상 구간을 찾고 필요한 rank만 짧게 trace합니다.
Synthetic workload와 실제 LLM 학습 결과는 구분합니다.

## CPU Checks

저장소 루트에서 아래 순서로 실행합니다.
Setup은 `.venv`를 만들고 pytest를 설치하지만 CUDA PyTorch, NCCL Tests나 Python package 자체를 설치하지 않습니다.
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
`GRAFANA_ADMIN_PASSWORD`와 cluster 이름, 이름이 붙은 monitoring target 목록을 지정합니다.
`SPARK_TARGETS`는 `node=address` 항목을 쉼표로 연결한 값이므로 노드 수를 고정하지 않습니다.

```bash
CLUSTER_NAME='<cluster-name>' \
SPARK_TARGETS='trainer-0=<first-node-management-address>,rollout-0=<second-node-management-address>' \
GRAFANA_ADMIN_PASSWORD='<non-default-password>' \
  bash scripts/run_spark_observability.sh server
```

Prometheus는 loopback 19090, Grafana는 loopback 13000에 바인딩합니다.
Server는 종료할 때까지 실행하므로 작업을 마치면 해당 세션을 종료합니다.
기존 PoC의 `SPARK1_ADDR`와 `SPARK2_ADDR`를 함께 주는 방식도 `spark1`, `spark2` 이름으로 호환됩니다.
`SERVER_CONFIG_ONLY=1`을 지정하면 service를 시작하지 않고 generated Prometheus·Grafana provisioning 파일만 작성해 target 구성을 확인합니다.
일부 GB10 NVML 값은 unavailable/null이며 이를 사용량 0으로 해석하지 않습니다.
시스템 메모리와 학습 process의 CUDA allocated/reserved peak도 구분합니다.

Grafana가 loopback에만 열리므로 브라우저에서 보려면 SSH local forwarding이 필요합니다.
Server role을 실행한 노드(위 예시는 spark1)에서 바로 보는 경우:

```bash
ssh -NT -L 13000:127.0.0.1:13000 spark@spark1
```

Controller 자체에는 GUI(브라우저)가 없고 사용자의 실제 client 머신에서 SSH로 접속해 쓰는 host이므로, 브라우저는 그 client 머신에서 엽니다.
그래서 client → controller → spark1 순서로 SSH forwarding을 두 번 거쳐야 하며, controller는 GUI 없이 순수 jump host로만 씁니다:

```bash
ssh -NT -L 13000:127.0.0.1:13000 -J <user>@<controller-ssh-alias> spark@spark1
```

브라우저에서 `http://localhost:13000` 접속 후 `admin`과 `GRAFANA_ADMIN_PASSWORD`로 로그인합니다.

`server` role은 `examples/observability/`의 `spark-resources.json`(uid `spark-profiling`), `compute-communication.json`, `data-storage.json`을 함께 복사합니다.
Run Overview는 실행 진행과 GPU 행렬을 요약하고, Compute & Communication은 GPU health·rank timer·인터페이스별 통신을, Data & Storage는 node-local device·filesystem을 다룹니다.
Grafana의 화면 링크는 시간 범위와 cluster·node·run 선택을 전달합니다.
저장소에는 `examples/observability/grafana/dashboards/cluster-resources.json`과 `compose.yaml`도 있는데, 이는 Docker Compose로 Prometheus·Grafana를 띄우는 별도의 일반 예시이며 이 ARM64 Spark 클러스터 워크플로우에서는 쓰이지 않습니다.

## Local Viewing

`node` role을 실행할 때 같은 shell에 `FRAMEWORK_METRICS_DIR`를 학습 launcher가 쓰는 값과 똑같이 지정하면, 그 노드의 `node` role이 [`profiling_lab.framework_metrics_textfile`](../observability/profiling_lab/framework_metrics_textfile.py)을 추가로 띄웁니다.
이 process는 `<framework>-rank-<rank>.json` 스냅샷을 주기적으로 읽어 `training_loss`, `training_tokens_per_second`, `training_step_time_seconds`, `training_step`, (Megatron이면) `training_timer_seconds{timer="..."}`를 GPU sampler와 같은 textfile collector(`$output_dir/textfile/framework.prom`)에 씁니다.

```bash
NODE_ADDR='<node-management-address>' \
FRAMEWORK_METRICS_DIR='<launcher가 쓴 output-directory>/framework-metrics' \
  bash scripts/run_spark_observability.sh node
```

이 값은 새 서비스가 아니라 launcher가 이미 쓰는 그 node-local 경로를 가리키기만 하면 되므로, 별도 NFS 공유나 collector 없이 그 노드의 rank(들)이 곧바로 같은 Grafana 대시보드에 나타납니다.
결과적으로 세 대시보드에서 host·GPU·network·RDMA·학습 지표를 같은 시간 범위와 선택값으로 볼 수 있습니다.
GPU 행렬은 현재 sampler가 보고한 GPU index와 utilization이며 allocation이 아닙니다.
통신 패널은 interface·RDMA port counter이고 endpoint 쌍별 traffic matrix가 아닙니다.
Data & Storage의 local device·filesystem 지표도 특정 run의 단독 사용량이나 특정 storage 구현의 topology로 해석하지 않습니다.
`FRAMEWORK_METRICS_DIR`를 지정하지 않으면 이 process는 시작되지 않고 나머지 host/GPU 모니터링은 그대로 동작합니다.

`FRAMEWORK_METRICS_DIR`를 NFS 공유 경로로 두 노드 모두에 지정하지는 않습니다 — 그러면 두 노드의 `node` role이 같은 파일을 각자 읽어 같은 rank가 두 `instance` label로 중복 노출됩니다.
각 노드가 자신의 node-local `FRAMEWORK_METRICS_DIR`(기본값)만 읽게 두는 것이 맞습니다.

과거 run을 조회하고 싶으면 서버가 필요 없습니다 — 아래 [Run History](#run-history) 절을 씁니다.

## Run History

학습 launcher는 이미 output-dir에 파일로 기록을 남깁니다: Megatron은 `run-metadata-<stage>.json`, TRL은 `summary-<stage>.json`, 그리고 `OBSERVATORY_RUN_ID`/`FRAMEWORK_METRICS_DIR`가 설정돼 있었다면 `framework-metrics/<framework>-rank-<rank>.json`(마지막 step 값)도 남습니다.
[`profiling_lab.show_run`](../observability/profiling_lab/show_run.py)은 이 파일들을 그냥 읽어서 요약해 출력하는 CLI로, 서버나 collector가 필요 없습니다.

```bash
PYTHONPATH=observability python3 -m profiling_lab.show_run '<output-dir>'
```

output-dir가 controller에서 보이지 않는 node-local 경로면 해당 Spark 노드에서 실행합니다.
`framework-metrics`는 학습 도중에는 최신 step 값이고 학습이 끝나면 마지막 값에 고정되며, 전체 loss 추이가 필요하면 `logs/`의 학습 로그를 확인합니다.

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
