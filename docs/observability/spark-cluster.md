# Spark Cluster Profiling Walkthrough

이 브랜치의 기본 실습은 **Spark cluster에서 실행**합니다.
공통 장비 소개는 [main의 PoC Setups](../../README.md#poc-setups)를 참고하세요.
Controller는 편집·테스트·SSH 조율을 맡고, GPU 연산은 `spark1`, `spark2`에서 실행합니다.
처음에는 아래 순서로 host/GPU 관측 → 작은 collective → storage → DDP 기준 실행 → selected trace를 진행하세요.
Megatron·verl 통합은 이 기본 경로를 완료한 다음 선택하는 확장 실습입니다.

## 1. Prepare the Checkout and Python

Controller에서 실행합니다.
이미 checkout이 있다면 미완료 작업을 보존한 뒤 `profiling` 브랜치를 사용하세요.

```bash
git clone -b profiling https://github.com/daegyu94/post-training-lab.git
cd post-training-lab
../../observability/scripts/setup.sh
. .venv/bin/activate
python -m pytest -q
```

NFS의 같은 checkout은 Spark에서 `/home/spark/shared/post-training-lab`로 보입니다.
Controller의 `.venv`를 Spark에서 활성화하지 마세요. CPU architecture가 다르므로 Python 환경은 노드마다 local disk에 둡니다.
두 Spark 노드에 각각 SSH로 접속해 다음을 확인합니다.

```bash
cd /home/spark/shared/post-training-lab
hostname
uname -m
free -h
nvidia-smi
ip -br -4 addr
timedatectl show -p NTPSynchronized
```

기존 CUDA Python 환경이 있으면 `PYTHON`을 그 환경의 실행 파일로 지정합니다.
새 환경이 필요하면 각 노드에서 다음과 같이 준비합니다.

```bash
python3 -m venv "$HOME/.venvs/profiling"
export PYTHON="$HOME/.venvs/profiling/bin/python"
"$PYTHON" -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu130
"$PYTHON" -c 'import torch; x=torch.randn(64,64,device="cuda"); print(torch.__version__, torch.version.cuda, (x@x).sum().item())'
```

검증 run은 기존 node-local Torch `2.10.0+cu130` 환경을 사용했습니다.
환경의 재설치는 검증하지 않았으며, 설치 후 실제 CUDA 연산 성공까지 확인해야 합니다.
Spark1의 해당 build는 지원 capability 범위 경고를 출력했지만 이번 작은 DDP 실행은 성공했습니다.

## 2. Read Spark Memory Correctly

Spark는 CPU와 GPU가 같은 DRAM을 사용하는 unified memory architecture(UMA)입니다.
따라서 CPU tensor, GPU allocation, 다른 process와 OS cache가 같은 노드의 메모리 여유에 영향을 줍니다.
두 노드의 메모리가 하나의 pool로 합쳐지는 것은 아닙니다.

| 관측값 | 해석 |
| --- | --- |
| `free -h`의 `available`, `MemAvailable` | OS 관점의 메모리 여유 추정; GPU allocation 성공 보장은 아님 |
| swap 사용량과 `vmstat 1`의 `si`, `so` | 메모리 압박과 swap I/O를 함께 확인; swap 잔량은 GPU 성능 예산이 아님 |
| GPU 전체 `memory.used`, `memory.total` | 검증한 GB10 driver에서 미지원; 화면은 `N/A`, JSON은 `null` |
| `nvidia-smi`의 process `used_gpu_memory` | 지원되는 별도 process 관측값; 전체 process RSS나 UMA 사용량을 대체하지 않음 |
| process RSS, `VmSwap` | 해당 process의 host 상주·swap 메모리; GPU 수치와 합산하지 않음 |
| PyTorch allocated/reserved peak | 해당 rank의 CUDA allocator 관측; 전체 node 사용량이 아님 |

`free` 열만 보거나 process GPU memory만 보고 학습을 시작하지 마세요.
OS page cache 때문에 `used`가 높을 수도 있으므로 `available`, process RSS와 swap을 함께 비교합니다.
NVIDIA도 [GPU 전체 메모리 미지원과 UMA 관측 차이](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html)를 설명합니다.
미지원, exporter 장애, 실제 0은 서로 다른 상태입니다.
프로세스가 없으면 process metric이 없고, 측정된 utilization 0은 그대로 0입니다.

```bash
free -h
vmstat 1 3
ps -eo pid,rss,%mem,comm --sort=-rss | head
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
```

Launcher는 이 작은 실습에 `MemAvailable >= 8 GiB`와 다른 GPU compute process가 없는 상태를 요구합니다.
이는 대형 모델의 적합성 기준이 아니라 작은 baseline을 위한 여유 기준입니다.
한 노드에서 preflight가 실패하면 다른 노드의 launcher도 Ctrl-C로 종료하고 원인을 해결한 뒤 함께 다시 실행하세요.
각 launcher는 기본 300초 timeout으로 제한됩니다.

## 3. Start Host and GPU Telemetry

Docker 권한이나 DCGM을 기본 prerequisites로 요구하지 않습니다.
Node Exporter는 user-space binary로 실행하고, GPU는 `nvidia-smi` 값을 textfile collector로 전달합니다.
DCGM 전용 진단은 이 경로의 검증 범위에 포함되지 않습니다.

각 Spark 노드에서 도구를 설치합니다.
`curl`, `tar`, `sha256sum`과 외부 다운로드 연결이 필요합니다.
도구는 node-local `~/.local/share/profiling-lab-tools`에 설치됩니다.

```bash
# spark1: Prometheus와 Grafana도 설치
../../observability/scripts/install_spark_tools.sh server
# spark2: Node Exporter만 설치
../../observability/scripts/install_spark_tools.sh node
```

아래 명령은 **노드별 별도 SSH 터미널에서 유지**합니다.
`NODE_ADDR`에는 해당 노드의 management IP를 넣습니다.
검증 환경의 주소 예시는 `spark1=192.168.0.11`, `spark2=192.168.0.12`이며 실제 `ip` 출력으로 확인하세요.

```bash
export NODE_ADDR=192.168.0.11   # spark2에서는 192.168.0.12
DURATION=1800 ../../observability/scripts/run_spark_observability.sh node
```

Spark1의 별도 터미널에서 server를 실행합니다.
암호는 출력이나 Git 파일에 저장하지 않습니다.

```bash
export SPARK1_ADDR=192.168.0.11 SPARK2_ADDR=192.168.0.12
read -rsp 'Grafana password: ' GRAFANA_ADMIN_PASSWORD; echo
export GRAFANA_ADMIN_PASSWORD
timeout 1800 ../../observability/scripts/run_spark_observability.sh server
```

Server는 loopback에 바인딩합니다.
Dashboard를 볼 PC에서 SSH tunnel을 유지하고 `http://localhost:13000`에 접속해 `admin`과 설정한 암호로 로그인하세요.

```bash
ssh -N -L 13000:127.0.0.1:13000 -L 19090:127.0.0.1:19090 spark@spark1
```

Grafana의 **Spark Profiling Lab** dashboard에서 node별 host available memory, swap, GPU utilization·power와 process GPU memory를 확인합니다.
Prometheus `http://localhost:19090`에서 `up{job="spark"}`가 두 개 모두 1인지 확인하세요.
`profiling_gpu_sample_timestamp_seconds`의 경과 시간이 5초를 넘으면 stale sample입니다.
Node Exporter endpoint는 두 노드의 management IP TCP 19100에서 접근 가능해야 합니다.
GPU raw sample은 `artifacts/spark/monitoring-<hostname>/gpu-*.jsonl`에 저장됩니다.

## 4. Measure Collective and Storage Baselines

두 노드의 터미널에서 공통 변수를 설정합니다.
`MASTER_ADDR`는 `spark1`의 data IP이며 아래는 검증 환경 예시입니다.
`PROFILE_RUN_ID`는 controller에서 한 번 정한 값을 두 노드에 똑같이 복사합니다.

```bash
export PYTHON="$HOME/.venvs/profiling/bin/python"  # 기존 환경을 쓰면 해당 경로로 변경
export MASTER_ADDR=12.201.48.11 MASTER_PORT=29670
export PROFILE_RUN_ID=spark-$(date -u +%Y%m%dT%H%M%S)
export NCCL_SOCKET_IFNAME=enp1s0f0np0 NCCL_IB_HCA=rocep1s0f0
export NODE_RANK=0  # spark2에서는 1
../../observability/scripts/run_spark_profile.sh collective
```

반드시 두 launcher를 모두 시작하세요. 한쪽만 실행하면 상대 노드를 기다립니다.
결과는 `artifacts/spark/$PROFILE_RUN_ID/collective/rank-{0,1}.json`입니다.
4 KiB–64 MiB all-reduce correctness와 5회 warmup 후 20회 평균 latency를 기록합니다.
이는 PyTorch/NCCL probe이며 NCCL Tests의 `algbw`·`busbw` 결과가 아닙니다.
MPI와 NCCL Tests가 준비된 환경에서는 [Lab 02](labs/02-hardware-baselines.md)의 별도 benchmark를 사용하세요.
`NET/IB`를 로그에서 확인하고, 경로가 다르면 다른 transport의 결과로 기록합니다.

Storage는 GPU 실행이 끝난 뒤 `spark2`에서 실행합니다.
`gcc`, `make`로 fio를 user-space build하며 libaio가 없는 기본 환경에서는 `posixaio`를 사용합니다.

```bash
../../observability/scripts/install_spark_tools.sh fio
export FIO_BINARY="$HOME/.local/share/profiling-lab-tools/fio-fio-3.39/fio"
export FIO_IOENGINE=posixaio FIO_SIZE=256M FIO_RUNTIME=5
export FIO_DIRECTORY="$HOME/.cache/profiling-lab-fio-$PROFILE_RUN_ID"
OUTPUT_DIR="artifacts/spark/$PROFILE_RUN_ID/fio-local" ../../observability/scripts/run_fio_baseline.sh
export FIO_DIRECTORY="$PWD/artifacts/spark/$PROFILE_RUN_ID/fio-nfs-data"
OUTPUT_DIR="artifacts/spark/$PROFILE_RUN_ID/fio-nfs" ../../observability/scripts/run_fio_baseline.sh
```

各試験ではなく、各試験 is not used.
256 MiB file、4 MiB block、direct I/O、queue depth 16、write/read各5秒の短い試験です。

## 5. Run DDP, Capture, and Compare

同じではなく同じ is not used.
두 노드에서 각각 다음 명령을 실행하고, 각 단계의 두 launcher가 모두 종료된 뒤 다음 단계로 넘어갑니다.

```bash
../../observability/scripts/run_spark_profile.sh baseline
../../observability/scripts/run_spark_profile.sh capture
```

Baseline은 profiler를 끄고, capture는 rank 0·1의 loop step 6–7만 trace합니다.
각각 24 steps를 실행하고 처음 4 steps를 제외한 step 시간의 mean/p50/p95와 global synthetic tokens/s를 저장합니다.
Step 시간은 data 생성부터 optimizer 완료까지 CUDA synchronize로 측정합니다.
Profiler export와 console 출력 시간은 step 시간에서 제외되므로 capture의 전체 wall-time overhead를 의미하지 않습니다.
Global synthetic tokens/s는 전체 rank의 tokens/step을 step별 가장 느린 rank 시간의 평균으로 나눈 값입니다.
이 작은 synthetic workload는 실제 LLM 품질·처리량 benchmark가 아닙니다.

```text
artifacts/spark/<run-id>/
+-- baseline/
|   +-- node-0.log, node-1.log
|   +-- node-0-manifest.txt, node-1-manifest.txt
|   +-- rank-0.json, rank-1.json
+-- capture/
    +-- rank-0.json, rank-1.json
    +-- traces/rank-0/trace-0.json
    +-- traces/rank-1/trace-0.json
```

Trace를 [Perfetto](https://ui.perfetto.dev/)에서 열거나 [Lab 05](labs/05-selected-trace.md)의 HTA mapping으로 분석합니다.
두 trace의 CUDA kernel, NCCL kernel과 active step 개수를 확인하세요.
서로 다른 노드의 timestamp만으로 정밀한 rank skew를 단정하지 않습니다. NTP 동기화 여부도 manifest에 있습니다.

## 6. Record and Clean Up

작은 summary, command, version, 실패 이유와 분석은 `docs/results/`에 기록합니다.
Raw trace, Prometheus database와 fio test file은 `artifacts/`에 보관하고 Git에는 넣지 않습니다.
남겨둔 test file은 사용한 정확한 `FIO_DIRECTORY`를 확인한 뒤 `profiling-lab-checkpoint.bin`만 삭제합니다.
관측 터미널은 Ctrl-C 또는 timeout으로 종료하면 자신이 시작한 child process와 GPU textfile을 정리합니다.
다른 학습 작업이나 exporter는 종료하지 않습니다.

실행 결과와 해석은 제외된 runtime artifact인 `results/spark-20260908.md`에 기록된 역사적 자료를 참고하세요.
