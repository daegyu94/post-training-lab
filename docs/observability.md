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

`server` role이 복사하는 대시보드는 `examples/observability/spark-resources.json`(uid `spark-profiling`)입니다.
저장소에는 `examples/observability/grafana/dashboards/cluster-resources.json`과 `compose.yaml`도 있는데, 이는 Docker Compose로 Prometheus·Grafana를 띄우는 별도의 일반 예시이며 이 ARM64 Spark 클러스터 워크플로우에서는 쓰이지 않습니다 — 대시보드를 고칠 때는 `spark-resources.json` 쪽을 수정해야 실제로 반영됩니다.

## Local Viewing

`node` role을 실행할 때 같은 shell에 `FRAMEWORK_METRICS_DIR`를 학습 launcher가 쓰는 값과 똑같이 지정하면, 그 노드의 `node` role이 [`profiling_lab.framework_metrics_textfile`](../observability/profiling_lab/framework_metrics_textfile.py)을 추가로 띄웁니다.
이 process는 `<framework>-rank-<rank>.json` 스냅샷을 주기적으로 읽어 `training_loss`, `training_tokens_per_second`, `training_step_time_seconds`, `training_step`, (Megatron이면) `training_timer_seconds{timer="..."}`를 GPU sampler와 같은 textfile collector(`$output_dir/textfile/framework.prom`)에 씁니다.

```bash
NODE_ADDR='<node-management-address>' \
FRAMEWORK_METRICS_DIR='<launcher가 쓴 output-directory>/framework-metrics' \
  bash scripts/run_spark_observability.sh node
```

이 값은 새 서비스가 아니라 launcher가 이미 쓰는 그 node-local 경로를 가리키기만 하면 되므로, 별도 NFS 공유나 collector 없이 그 노드의 rank(들)이 곧바로 같은 Grafana 대시보드에 나타납니다.
결과적으로 `spark-resources.json`의 `TCP/Ethernet network throughput`, `RDMA (InfiniBand/RoCE) throughput`, `Training loss`, `Training throughput (tokens/s)`, `Training step time`, `Training sample age` panel까지 host·GPU·network·RDMA·학습 지표가 **하나의 Grafana 대시보드**에서 보입니다.
`FRAMEWORK_METRICS_DIR`를 지정하지 않으면 이 process는 시작되지 않고 나머지 host/GPU 모니터링은 그대로 동작합니다.

`FRAMEWORK_METRICS_DIR`를 NFS 공유 경로로 바꾸는 [Lightweight Local Viewer](#lightweight-local-viewer-history-server-pattern) 패턴과 이 bridge를 같이 쓰지는 않습니다 — 그러면 두 노드의 `node` role이 같은 파일을 각자 읽어 같은 rank가 두 `instance` label로 중복 노출됩니다.
Grafana로만 보는 이 경로에서는 각 노드가 자신의 node-local `FRAMEWORK_METRICS_DIR`(기본값)만 읽게 두는 것이 맞습니다.

아래 [Controller에서 결과 수집과 표시](#controller에서-결과-수집과-표시) 절의 `profiling_lab.collector`/`node_agent`는 목적이 다른 별도 도구입니다.
Prometheus·Grafana 없이 Python 프로세스 하나로 뜨는 자체 웹 UI로, 학습 중 실시간 스트림과 과거 run 요약·비교(run history)에 초점을 두며 CPU·메모리·NIC(TCP)와 학습 지표만 다룹니다 — GPU와 RDMA는 다루지 않습니다.
지금 이 노드에서 host·GPU·network·RDMA·학습 지표를 한 화면에서 상시로 보려면 위 Grafana 대시보드를 쓰고, 과거 run을 비교하거나 Prometheus·Grafana 없이 가볍게 보고 싶을 때만 아래 절을 씁니다.

## Controller에서 결과 수집과 표시

이 절은 Prometheus·Grafana와 별개로 동작하는 자체 web UI([Local Viewing](#local-viewing) 참고)입니다 — GPU와 RDMA는 다루지 않으며, 그 대신 run history 비교와 Prometheus·Grafana 없는 가벼운 실행을 지원합니다.

각 Spark 노드의 GPU sampler는 원본 값을 JSONL로 저장하고 Node Exporter가 읽을 지표 파일도 갱신합니다.
학습 launcher는 같은 `run_id` 아래에 rank별 로그, summary와 measurement JSONL을 남깁니다.

```text
spark1 telemetry --+
                  +--> controller collector --> SQLite --> 실시간 확인
spark2 telemetry --+                            |
                                                +--> 내장 웹 UI
```

이 저장소의 `profiling_lab.collector`는 Python 표준 라이브러리로 만든 controller collector와 로컬 웹 UI를 제공합니다.
Spark 노드는 SSH reverse tunnel을 통해 측정값을 보내므로 collector port를 외부에 열지 않고도 controller 화면에서 최신 값을 확인할 수 있습니다.

저장소 루트에서 token을 한 번 만들고 collector를 실행합니다.
Token과 SQLite는 `artifacts/` 아래에 두며 Git에 포함하지 않습니다.

```bash
mkdir -p artifacts
umask 077
test -s artifacts/observatory-token || \
  python3 -c 'import secrets; print(secrets.token_hex(24))' > artifacts/observatory-token
chmod 600 artifacts/observatory-token
export OBSERVATORY_TOKEN="$(cat artifacts/observatory-token)"
PYTHONPATH=observability python3 -m profiling_lab.collector \
  --bind 127.0.0.1 --port 8001 \
  --database artifacts/observatory.sqlite3
```

원격 GUI host에서는 `ssh -N -L 8001:127.0.0.1:8001 <controller-host>`를 실행합니다.
브라우저의 `http://127.0.0.1:8001/`은 전체 profiling dashboard와 run history를, `http://127.0.0.1:8001/telemetry.html`은 실시간 collector 화면을 제공합니다.
실시간 화면에 같은 token을 입력하면 2초마다 최신 지표를 조회합니다.

공통 runner는 output 이름을 `run_id`로 사용하고 TRL과 Megatron adapter가 rank별 최신 metric을 `<output>/framework-metrics/`에 atomic JSON으로 기록합니다.
TRL은 Trainer가 집계한 loss와 실제 누적 입력 token 차이를 사용하고, Megatron은 callback loss, step wall time, 설정된 batch·sequence 상한 기반 tokens/s와 rank timer를 기록합니다.

내장 `profiling_lab.node_agent`에 같은 `run_id`와 `--framework-metrics-dir <output>/framework-metrics`를 주면 노드 전체 CPU·메모리·NIC와 framework metric이 함께 collector로 전송됩니다.
학습 process는 collector에 직접 접속하지 않으므로 collector 또는 tunnel 장애가 학습 step을 막지 않습니다.
직접 backend launcher를 실행할 때에는 `OBSERVATORY_RUN_ID`와 `FRAMEWORK_METRICS_DIR`를 함께 설정해야 adapter가 활성화됩니다. 공통 runner를 쓰면 두 값을 자동으로 설정하므로 따로 지정할 필요가 없습니다.

Spark 노드에서 controller의 loopback collector로 보내려면 controller에서 reverse tunnel과 agent를 함께 실행합니다.

```bash
run_id='<training-output-directory-name>'
{ cat artifacts/observatory-token; printf '\n'; } | \
  ssh -o ExitOnForwardFailure=yes -R 18001:127.0.0.1:8001 spark@spark1 \
    "read -r OBSERVATORY_TOKEN; export OBSERVATORY_TOKEN; \
     cd /home/spark/shared/post-training-lab; \
     PYTHONPATH=observability python3 -m profiling_lab.node_agent \
       --endpoint http://127.0.0.1:18001 --run-id '$run_id' \
       --framework-metrics-dir '<node-output>/$run_id/framework-metrics'"
```

`spark2`도 같은 명령으로 실행하며 node-local output 경로만 해당 노드 값으로 지정합니다.

### Lightweight Local Viewer (History Server Pattern)

위 push 경로는 SSH reverse tunnel과 별도 collector 프로세스가 필요합니다.
Controller와 두 Spark 노드가 이미 같은 NFS 공유 디렉터리를 보고 있으므로(`AGENTS.md`의 NFS 절 참고), `FRAMEWORK_METRICS_DIR`를 node-local NVMe 대신 그 공유 경로로 지정하면 collector나 SSH tunnel 없이도 controller가 파일을 직접 읽을 수 있습니다.
Apache Spark나 MapReduce의 History Server와 같은 pull 방식입니다: 애플리케이션은 잘 알려진 공유 경로에 쓰기만 하고, 뷰어는 그 경로를 그냥 읽습니다.

```text
Collector push 경로                           Local viewer pull 경로

spark1 --SSH tunnel--> collector --> SQLite    spark1 --+
spark2 --SSH tunnel--> collector       |                +--> NFS 공유 경로 (같은 파일, 복사 없음)
                        |                       spark2 --+          |
                        +--> 내장 웹 UI                              v
                                                     controller가 직접 읽음 (SSH·tunnel 불필요)
                                                                     |
                                                                     v
                                                     정적 HTTP server + 브라우저 polling
```

`backends/trl/scripts/run_spark_cluster.sh`와 `backends/megatron/scripts/run_spark_cluster.sh`는 `FRAMEWORK_METRICS_DIR`가 이미 설정돼 있으면 그 값을 그대로 쓰고, 없으면 기존처럼 `<output>/framework-metrics`(node-local NVMe)를 기본값으로 씁니다.
공유 경로를 가리키게 하려면 launcher 호출 전에 `FRAMEWORK_METRICS_DIR`를 NFS 경로로 export하면 됩니다. `<output>/model`처럼 checkpoint 저장 경로는 이 값의 영향을 받지 않으므로 30B NVMe 실습처럼 checkpoint 자체는 node-local NVMe에 남기고 metric만 공유 경로로 보낼 수 있습니다.
내장 collector와 `metrics_bridge`를 사용하면 그 경로의 `<framework>-rank-<rank>.json`을 주기적으로 읽어 같은 로컬 UI에서 볼 수 있습니다.
이 경로는 값을 외부에 공개하지 않으며 controller에서만 보입니다.
2026-09-09에 Qwen2.5-0.5B DDP smoke를 60 step으로 돌리며 이 방식을 실제로 검증했습니다: launcher가 공유 경로에 쓴 `trl-rank-0.json`을 controller가 SSH 없이 직접 읽었고, 약 8초 동안 step 4→60까지 13번의 서로 다른 값을 관찰했습니다.

### Metrics Bridge: NFS 지표를 내장 UI에서 보기

`observability/profiling_lab/metrics_bridge.py`는 NFS 공유 `FRAMEWORK_METRICS_DIR`를 주기적으로 읽어 내장 collector API로 전달합니다.

```text
Node agent 경로 (SSH push)                     Metrics bridge (NFS pull)

spark1 --SSH tunnel--> agent --+               spark1 --+
spark2 --SSH tunnel--> agent --+--> collector           +--> NFS 공유 경로
                                                spark2 --+          |
                                                                     v
                                              controller의 bridge poller (SSH 불필요)
                                                                     |
                                                            POST /api/framework-metrics
                                                                     v
                                                    controller의 같은 collector (동일 API)
                                                                     |
                                                                     v
                                                  내장 collector 웹 UI
```

collector와 poller 모두 controller 안에서만 통신하므로(둘 다 기본값 `127.0.0.1`), Spark 노드로의 SSH나 reverse tunnel이 전혀 없습니다.

```bash
# collector: post-training-lab 저장소 루트에서
export OBSERVATORY_TOKEN="$(cat artifacts/observatory-token)"
PYTHONPATH=observability python3 -m profiling_lab.collector \
  --database artifacts/observatory.sqlite3

# bridge: post-training-lab 저장소 루트에서
export OBSERVATORY_TOKEN="<위와 같은 값>"
PYTHONPATH=observability python3 -m profiling_lab.metrics_bridge \
  --metrics-dir /path/to/shared/observatory-demo/<run_id>/framework-metrics \
  --endpoint http://127.0.0.1:8001 --interval 2
```

`--endpoint`의 기본값은 controller local(`http://127.0.0.1:8001`)입니다.
Bridge는 NFS 공유 경로를 읽을 수 있는 곳에서 실행합니다.

2026-09-09에 Qwen2.5-0.5B DDP smoke를 80 step으로 돌리며 collector와 bridge가 쓰는 실제 `GET /api/framework-metrics` 계약을 검증했습니다: bridge가 `--once` 없이 계속 돌면서 학습이 진행되는 동안 step 1→80까지 총 22개의 서로 다른 sample을 전달했습니다.

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
